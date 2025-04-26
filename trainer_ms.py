from importlib.metadata import requires
import time
from typing import List, Tuple
from idna import valid_contextj
from omegaconf import DictConfig
import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from data.datasets import MedicalDecathlonDataset
from utils import losses
from utils.assertions import ensure_in
from utils.callbacks import EarlyStopping
from utils.inference import call_fusion_fn, call_fusion_fn, compute_weights_depth, get_fusion_fn, weighted_softmax
from utils.losses import compute_ds_loss
from utils.metric_collecter import Agg, MetricCollector, TrainingCollector
from utils.metrics import dice_coefficient, dice_coefficient_classes
from utils.table import print_train_val_table
from utils.utils import RunManager
from utils.wandb_logger import WandBLogger
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
            self, 
            unified_cfg: DictConfig,
            model: torch.nn.Module, 
            train_dataloader: DataLoader[MedicalDecathlonDataset],
            val_dataloader: DataLoader[MedicalDecathlonDataset],
            test_dataloader: DataLoader[MedicalDecathlonDataset],
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None, # Add scheduler
            device: torch.device, 
            run_manager: RunManager,
            wandb_logger: WandBLogger | None 
        ):
            self.unified_cfg = unified_cfg
            self.model = model
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.test_dataloader = test_dataloader
            self.criterion = criterion
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.device = device
            self.logger = logging.getLogger(__name__)
            self.rm = run_manager
            self.wandb_logger = wandb_logger
            self.train_metrics = TrainingCollector()
            
            
            self.best_val_loss = float('inf')
            self.best_val_dice = float('-inf')
            self.early_stopping_counter = 0

            self.early_stopper = None
            self.early_stopping_cfg = unified_cfg.training.get('early_stopping', None)
            if self.early_stopping_cfg is not None: 
                try:
                    self.early_stopper = EarlyStopping(
                        patience=self.early_stopping_cfg.get('patience', 15),
                        delta=self.early_stopping_cfg.get('delta', 0.0),
                        criterion=self.early_stopping_cfg.get('criterion', 'loss'),
                        verbose=self.early_stopping_cfg.get('verbose', True),
                    )
                except Exception as e:
                    self.logger.error(f"Error initializing EarlyStopping: {e}. Disabling.")
                    self.early_stopper = None


            self.num_epochs = unified_cfg.training.num_epochs
            self.num_classes = unified_cfg.dataset.num_classes
            self.fusion_fn = get_fusion_fn(unified_cfg.model.deep_supervision.inference_fusion_mode)

            self.class_labels = {i: f"class_{i}" for i in range(self.num_classes)}

            self.logger.info(f"Trainer initialized with {self.num_epochs} epochs!")

            wandb_active = self.wandb_logger and self.wandb_logger.run and self.wandb_logger.is_active
            if not wandb_active:
                self.logger.info("WandB logging is not active. Skipping WandB setup.")
                return
                
            wandb_url = self.wandb_logger.run.url #type: ignore
            self.logger.info(f"WandB logging enabled: {wandb_url}")


    def train_one_epoch(self, epoch: int):
        """
        Train the model for one epoch.
        """
        self.model.train()
        tc = self.train_metrics
        tc.reset()
        tc.update({'ep': epoch})            

        try:
            for batch_idx, (images, masks) in enumerate(
                tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=False)
            ):
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass                
                segs, cons_pairs = self.model(images)

                weights = [1.0 / len(segs)] * len(segs)
                
                seg_losses: List[Tuple[float, torch.Tensor]] = []
                down_masks: List[torch.Tensor] = []
                for seg in segs:
                    gt = F.interpolate(
                        masks.unsqueeze(1).float(), size=seg.shape[2:], mode='nearest'
                    ).squeeze(1).long()
                    
                    down_masks.append(gt)

                for w, pred, gt in zip(weights, segs, down_masks):
                    loss = self.criterion(pred, gt)
                    seg_losses.append((w, loss))

                seg_loss:torch.Tensor = torch.tensor(0.0, device=self.device)
                for w, loss in seg_losses:
                    seg_loss += w * loss
                
                cons_losses: List[torch.Tensor] = [
                    F.mse_loss(ms_feats, enc_feats.detach())
                    for ms_feats, enc_feats in cons_pairs
                ]
                
                cons_loss = torch.tensor(0.0, device=self.device)
                for w, loss in zip(weights, cons_losses):
                    cons_loss += w * loss

                # network_frozen = False
                # awaken_consistency = torch.tensor(float(epoch > 60), device=self.device, requires_grad=False) #technically dont need require grad as we disable all below but :D yolo
                # if awaken_consistency and not network_frozen:
                #     network_frozen = True
                #     for param in self.model.parameters():
                #         param.requires_grad = False
                #     for param in self.model.msb_blocks.parameters():
                #         param.requires_grad = True
                #     self.logger.debug("Model parameters are frozen except for MSB blocks.")
                # self.logger.debug(f"awaken_consistency: {awaken_consistency}")
                # loss = segmentation_loss * (1.0 - awaken_consistency) + consistency_loss * awaken_consistency

                loss = seg_loss + cons_loss 

                # --- compute dice metrics per segmentation head ---
                dice_vals = []
                cls_dice_vals = []
                ignore_index = self.unified_cfg.dataset.get("ignore_index", 0)

                for pred, gt in zip(segs, down_masks):
                    pred_mask = torch.argmax(pred, dim=1)
                    dice_vals.append(dice_coefficient(pred_mask, gt, num_classes=self.num_classes, ignore_index=ignore_index))
                    cls_list = dice_coefficient_classes(
                        pred_mask, gt, self.num_classes, ignore_index=ignore_index
                    )
                    cls_dice_vals.append([x.item() for x in cls_list])
   

                # push raw lists of floats into the collector
                tc.update({
                    'loss_list'     : [l.item() for (_, l) in seg_losses],
                    'dice_list'     : dice_vals,
                    'cls_dice_list' : cls_dice_vals,
                    'cons_loss_list': [l.item() for l in cons_losses],
                })               
                
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Loss is NaN or Inf at epoch {epoch}. Skipping batch.")
                    tc.skip()
                    if self.wandb_logger:
                        self.wandb_logger.log_metrics({"train/skipped_batches": 1}, step=epoch, commit=False) 
                    continue
                
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # progress_bar.set_postfix(batch_loss=batch_loss)

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            if self.wandb_logger:
                self.wandb_logger.log_metrics({"train/skipped_batches": 1}, step=epoch)
            raise e
        
        # # --- Log epoch summary metrics to WandB ---
        # if self.wandb_logger:
        #     log_data = {
        #         'train/epoch_loss': avg_train_loss,
        #         'epoch': epoch,
        #         'learning_rate': self.optimizer.param_groups[0]['lr'] # Log LR per epoch
        #     }
        #     # current_weights = self.model.get_ds_weights()
        #     # self.wandb_logger.log_weights(current_weights, step=epoch) 
        #     self.wandb_logger.log_metrics(log_data, step=epoch)       
        #     # --- End of WandB logging ---

        # Step the LR scheduler if it exists (usually done per epoch)
        if self.lr_scheduler:
            # Some schedulers need metrics (e.g., ReduceLROnPlateau)
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Need validation loss, so step after validation
                pass 
            else:
                self.lr_scheduler.step()
        
        return tc.summary()

    
    def train(self):
        """
        Main training loop.
        """
        start_time = time.time()
        
        epoch = -1 
        pbar = trange(self.num_epochs, desc="Training", leave=True)

        try: 
            for epoch in pbar:
                epoch_start_time = time.time()

                # train_dict = self.train_one_epoch(epoch)
                # val_dict = self.validate(epoch)

                train_dict = self.train_one_epoch(epoch)
                table = print_train_val_table(train_dict, {})
                pbar.write("")          
                pbar.write(table)        
                
                self.logger.info(f"Epoch {epoch+1} Training Summary:\n{table}") 
               
                val_loss, val_dice, val_dice_per_class = self.validate(epoch)
                
                epoch_duration = time.time() - epoch_start_time

                # train_dict['time'] = val_dict['time'] = epoch_duration

                # self.logger.info(
                #     f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_duration:.2f}s | "
                #     f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
                #     f"Val Dice per Class: {val_dice_per_class}"
                # )

                # --- Log Epoch Timing ---
                if self.wandb_logger:
                    self.wandb_logger.log_metrics({'epoch_duration_sec': epoch_duration}, step=epoch, commit=False)
                # ------------------------

                # Step ReduceLROnPlateau scheduler if used
                if self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)      

                # --- Early Stopping and Model Saving ---
                if self.early_stopper is not None:
                    stop_training, is_improved = self.early_stopper(val_loss, val_dice)

                    if self.wandb_logger:
                        self.wandb_logger.log_metrics({
                            'val/best_loss': self.early_stopper.best_loss,
                            'val/best_dice': self.early_stopper.best_dice
                        }, step=epoch, commit=True)

                    metric = self.early_stopper.best_dice if self.early_stopper.criterion != 'loss' else self.early_stopper.best_loss
                    
                    if is_improved:
                        model_save_path = self.rm.save_model(
                            model=self.model,
                            optimizer=self.optimizer, # Pass optimizer
                            scheduler=self.lr_scheduler, # Pass scheduler (can be None)
                            epoch=epoch,
                            metric=metric
                        )
                        self.logger.info(f"New best model checkpoint saved to {model_save_path}")  
                    if stop_training:
                        break
                    
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds.")
            
            if self.wandb_logger and self.wandb_logger.run:
                self.wandb_logger.run.summary["total_training_time_sec"] = total_time
                self.wandb_logger.run.summary["stopped_epoch"] = epoch + 1

                if self.early_stopper is not None:
                    self.wandb_logger.run.summary["best_val_loss"] = self.early_stopper.best_loss
                    self.wandb_logger.run.summary["best_val_dice"] = self.early_stopper.best_dice

        except Exception as e:
            self.logger.error(f"Unhandled exception during training: {e}")
            if self.wandb_logger:
                self.wandb_logger.finalize(exit_code=1) 
            raise e 


    def validate(self, epoch: int):
        """Evaluates the model on the validation set."""
        self.model.eval()
        ignore_index = self.unified_cfg.dataset.get("ignore_index", 0)

        logged_images = False # Flag to log images only once per epoch  

        progress_bar = 

        with torch.no_grad():
            try:
                for images, masks in tqdm(self.val_dataloader, 
                                          desc=f"Epoch {epoch+1}/{self.num_epochs} [Validate]", 
                                          leave=False
                    ):
                    images, masks = images.to(self.device), masks.to(self.device)
                    
                    # print(f"[DEBUG validate] Batch {batch_idx} Image Shape: {images.shape}") # -- ADD
                    pred_logits = self.model(images)
                    
                    loss = self.criterion(pred_logits, masks)

                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf validation loss encountered at epoch {epoch}, batch {batch_idx}. Skipping batch metrics.")
                        continue 

                    val_loss += loss.item()

                    pred_masks = torch.argmax(pred_logits, dim=1)

                    dice_scores_batch_per_class = dice_coefficient_classes(
                        pred_masks,
                        masks,
                        self.num_classes,
                        ignore_index=ignore_index
                    )

                    dice_scores_sum_per_class += torch.stack(dice_scores_batch_per_class)
                    dice_per_batch_avg = torch.mean(torch.stack(dice_scores_batch_per_class)).item()
                    dice_score_total += dice_per_batch_avg 
                    
                    # --- Log segmentation examples to WandB (once per epoch) ---
                    is_first_batch_and_not_logged = self.wandb_logger is not None and not logged_images and batch_idx == 0
                    if is_first_batch_and_not_logged: 
                        self.wandb_logger.log_segmentation_masks( # type: ignore
                            images=images, 
                            true_masks=masks, 
                            pred_masks=pred_logits,
                            step=epoch,
                            class_labels=self.class_labels,
                        ) 
                        logged_images = True
                    # ---------------------------------------------------------

            except Exception as e:
                self.logger.error(f"Error during validation epoch {epoch}: {str(e)}")
                if self.wandb_logger:
                    self.wandb_logger.finalize(exit_code=1)
                raise e
            

        avg_val_loss = val_loss / num_batches
        avg_dice_score = dice_score_total / num_batches
        avg_dice_per_class = dice_scores_sum_per_class / num_batches

        # --- Log validation summary metrics to WandB ---
        if self.wandb_logger:
            log_data = {
                'val/loss': avg_val_loss,
                'val/dice': avg_dice_score,
            }

            # Add per-class dice scores
            for class_idx, score in zip(active_classes, avg_dice_per_class):
                log_data[f'val/dice_class_{self.class_labels.get(class_idx, class_idx)}'] = score.item() # Use label if available

            self.wandb_logger.log_metrics(log_data, step=epoch, commit=False) # Commit validation metrics + images here
            # -----------------------------------------------

        class_dice_summary = ', '.join(
            [f"Class {class_idx}: {score.item():.4f}" 
            for class_idx, score in zip(active_classes, avg_dice_per_class)]
        )

        log_message = (
            f" Validation Loss: {avg_val_loss:.4f} | Dice Score: {avg_dice_score:.4f}\n"
            f"  ↳ Per-Class: [ {class_dice_summary} ]"
        )
        self.logger.info(log_message)


        return val_loss, avg_dice_score, avg_dice_per_class



    # def validate(self, epoch: int):
    #     """Evaluates the model on the validation set."""
    #     self.model.eval()
    #     val_loss = 0.0
    #     dice_score_total = 0.0
    #     num_batches = len(self.val_dataloader)

    #     # For per-class Dice
    #     ignore_index = self.unified_cfg.dataset.get("ignore_index", 0)
    #     active_classes = [c for c in range(self.num_classes) if c != ignore_index]
    #     num_active = len(active_classes)
    #     dice_scores_sum_per_class = torch.zeros(num_active, device=self.device)

    #     logged_images = False # Flag to log images only once per epoch  

    #     progress_bar = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Validate]", leave=False)

    #     with torch.no_grad():
    #         try:
    #             for batch_idx, (images, masks) in enumerate(progress_bar):
    #                 images, masks = images.to(self.device), masks.to(self.device)
                    
    #                 # print(f"[DEBUG validate] Batch {batch_idx} Image Shape: {images.shape}") # -- ADD
    #                 pred_logits = self.model(images)
                    
    #                 loss = self.criterion(pred_logits, masks)

    #                 if torch.isnan(loss) or torch.isinf(loss):
    #                     self.logger.warning(f"NaN/Inf validation loss encountered at epoch {epoch}, batch {batch_idx}. Skipping batch metrics.")
    #                     continue 

    #                 val_loss += loss.item()

    #                 pred_masks = torch.argmax(pred_logits, dim=1)

    #                 dice_scores_batch_per_class = dice_coefficient_classes(
    #                     pred_masks,
    #                     masks,
    #                     self.num_classes,
    #                     ignore_index=ignore_index
    #                 )

    #                 dice_scores_sum_per_class += torch.stack(dice_scores_batch_per_class)
    #                 dice_per_batch_avg = torch.mean(torch.stack(dice_scores_batch_per_class)).item()
    #                 dice_score_total += dice_per_batch_avg 
                    
    #                 # --- Log segmentation examples to WandB (once per epoch) ---
    #                 is_first_batch_and_not_logged = self.wandb_logger is not None and not logged_images and batch_idx == 0
    #                 if is_first_batch_and_not_logged: 
    #                     self.wandb_logger.log_segmentation_masks( # type: ignore
    #                         images=images, 
    #                         true_masks=masks, 
    #                         pred_masks=pred_logits,
    #                         step=epoch,
    #                         class_labels=self.class_labels,
    #                     ) 
    #                     logged_images = True
    #                 # ---------------------------------------------------------

    #         except Exception as e:
    #             self.logger.error(f"Error during validation epoch {epoch}: {str(e)}")
    #             if self.wandb_logger:
    #                 self.wandb_logger.finalize(exit_code=1)
    #             raise e
            

    #     avg_val_loss = val_loss / num_batches
    #     avg_dice_score = dice_score_total / num_batches
    #     avg_dice_per_class = dice_scores_sum_per_class / num_batches

    #     # --- Log validation summary metrics to WandB ---
    #     if self.wandb_logger:
    #         log_data = {
    #             'val/loss': avg_val_loss,
    #             'val/dice': avg_dice_score,
    #         }

    #         # Add per-class dice scores
    #         for class_idx, score in zip(active_classes, avg_dice_per_class):
    #             log_data[f'val/dice_class_{self.class_labels.get(class_idx, class_idx)}'] = score.item() # Use label if available

    #         self.wandb_logger.log_metrics(log_data, step=epoch, commit=False) # Commit validation metrics + images here
    #         # -----------------------------------------------

    #     class_dice_summary = ', '.join(
    #         [f"Class {class_idx}: {score.item():.4f}" 
    #         for class_idx, score in zip(active_classes, avg_dice_per_class)]
    #     )

    #     log_message = (
    #         f" Validation Loss: {avg_val_loss:.4f} | Dice Score: {avg_dice_score:.4f}\n"
    #         f"  ↳ Per-Class: [ {class_dice_summary} ]"
    #     )
    #     self.logger.info(log_message)

    #     val_dict = {
    #         'epoch': epoch,
    #         'losses': None,
    #         'dices': None,
    #         'dices_per_class': None,
    #     }

    #     return val_loss, avg_dice_score, avg_dice_per_class


    def test(self):
        """
        Tests the model on the test set. Test accurracy and metrics only. We don't care about loss.
        """
        self.rm.load_model(self.model)  
        self.model.eval()

        ignore_index = 0
        active_classes = [c for c in range(self.num_classes) if c != ignore_index]
        num_active = len(active_classes)
        dice_scores_sum = torch.zeros(num_active, device=self.device)
        
        progress_bar = tqdm(self.test_dataloader, desc="Testing Model")
        with torch.no_grad():
            try: 
                for images, masks in progress_bar:
                    images, masks = images.to(self.device), masks.to(self.device)
                    # TODO simon husk at lav wandb med batches og ikke epochs her! :D 
                    # outputs = self.model(images, "inference")

                    pred_logits = self.model(images)

                    pred_masks = torch.argmax(pred_logits, dim=1)
                    # current_weights = self.model.get_ds_weights()
                    # fused_output = call_fusion_fn(self.fusion_fn, outputs=outputs, weights=current_weights)
                    dice_scores_batch = dice_coefficient_classes(
                        pred_masks, 
                        masks, 
                        self.num_classes, 
                        ignore_index=ignore_index
                    )
                    dice_scores_sum += torch.stack(dice_scores_batch)
                    
                    batch_avg = torch.mean(torch.stack(dice_scores_batch)).item()
                    progress_bar.set_postfix(dice_score=f"{batch_avg:.4f}")
            except Exception as e:
                self.logger.error(f"Error during testing: {str(e)}")
                raise e
            
        avg_per_class = dice_scores_sum / len(self.test_dataloader)
        overall_avg = avg_per_class.mean().item()

        self.logger.info(f"Test | Overall Dice: {overall_avg:.4f}")
        for class_idx, score in zip(active_classes, avg_per_class):
            self.logger.info(
                f"     | Class {class_idx} Dice: {score.item():.4f}" 
            )
            
def should_log(epoch:int, batch_idx:int, freq=10) -> bool:
    "Logs at every `freq` epoch, and at the first batch of each epoch."
    return batch_idx == 0 and epoch % freq == 0

def log_consistency_stats(logger, ms_feats, enc_feats, prefix="Consistency Check"):
    ms_mean, enc_mean = ms_feats.mean().item(), enc_feats.mean().item()
    ms_std, enc_std = ms_feats.std().item(), enc_feats.std().item()
    ms_min, enc_min = ms_feats.min().item(), enc_feats.min().item()
    ms_max, enc_max = ms_feats.max().item(), enc_feats.max().item()
    mae = (ms_feats - enc_feats).abs().mean().item()
    mse_val = F.mse_loss(ms_feats, enc_feats).item()
    cos = F.cosine_similarity(ms_feats.flatten(), enc_feats.flatten(), dim=0).item()
    diff = (ms_feats - enc_feats).abs()
    pct_90 = torch.quantile(diff, 0.9).item()
    max_diff = diff.max().item()

    logger.debug(
        f"{prefix} | Shape: {ms_feats.shape} | "
        f"Mean(msb/enc): {ms_mean:.4f}/{enc_mean:.4f}, "
        f"Std(msb/enc): {ms_std:.4f}/{enc_std:.4f}, "
        f"Min(msb/enc): {ms_min:.4f}/{enc_min:.4f}, "
        f"Max(msb/enc): {ms_max:.4f}/{enc_max:.4f}, "
        f"MAE: {mae:.6f}, MSE: {mse_val:.6f}, CosSim: {cos:.4f}, "
        f"90% Diff: {pct_90:.4f}, Max Diff: {max_diff:.4f}"
    )
