import logging
import time
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
from utils.metrics import dice_coefficient, dice_coefficient_classes
from utils.utils import RunManager
from utils.wandb_logger import WandBLogger
from torch.nn.utils.clip_grad import clip_grad_norm_

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
            if self.wandb_logger and self.wandb_logger.run and self.wandb_logger.is_active:
                self.logger.info(f"WandB logging enabled: {self.wandb_logger.run.url}")
            else:
                self.logger.info("WandB logging disabled.")


    def train_one_epoch(self, epoch: int):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=False)

        try:
            for batch_idx, (images, masks) in enumerate(progress_bar):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
            
                outputs = self.model(images, "train") 

                current_weights = self.model.get_ds_weights()
                loss = compute_ds_loss(self.criterion, outputs, masks, current_weights, self.device)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Loss is NaN or Inf at epoch {epoch}. Skipping batch.")
                
                    # Optionally log this event to WandB, so we can see how often it happens
                    if self.wandb_logger:
                        self.wandb_logger.log_metrics({"train/skipped_batches": 1}, step=epoch, commit=False) 
                    continue
                
                loss.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                batch_loss = loss.item()
                total_loss += batch_loss
                progress_bar.set_postfix(batch_loss=batch_loss)

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            if self.wandb_logger:
                self.wandb_logger.log_metrics({"train/skipped_batches": 1}, step=epoch)
            raise e

        
        avg_train_loss = total_loss / num_batches

        # --- Log epoch summary metrics to WandB ---
        if self.wandb_logger:
            log_data = {
                'train/epoch_loss': avg_train_loss,
                'epoch': epoch,
                'learning_rate': self.optimizer.param_groups[0]['lr'] # Log LR per epoch
            }
            current_weights = self.model.get_ds_weights()
            self.wandb_logger.log_weights(current_weights, step=epoch) 
            self.wandb_logger.log_metrics(log_data, step=epoch)       
            # --- End of WandB logging ---

        # Step the LR scheduler if it exists (usually done per epoch)
        if self.lr_scheduler:
            # Some schedulers need metrics (e.g., ReduceLROnPlateau)
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Need validation loss, so step after validation
                pass 
            else:
                self.lr_scheduler.step()

        return avg_train_loss
    
    def train(self):
        """
        Main training loop.
        """
        start_time = time.time()
        num_epochs = self.num_epochs
        
        epoch = -1 

        try: 
            for epoch in trange(num_epochs, desc="Training Progress"):
                epoch_start_time = time.time()

                # --- Train one epoch ---
                train_loss = self.train_one_epoch(epoch)

                # --- Validate after each epoch ---
                val_loss, val_dice, val_dice_per_class = self.validate(epoch)
                
                epoch_duration = time.time() - epoch_start_time

                self.logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_duration:.2f}s | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
                    f" Val Dice per Class: {val_dice_per_class}"
                )


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
        val_loss = 0.0
        dice_score_total = 0.0
        num_batches = len(self.val_dataloader)

        # For per-class Dice
        ignore_index = self.unified_cfg.dataset.get("ignore_index", 0)
        active_classes = [c for c in range(self.num_classes) if c != ignore_index]
        num_active = len(active_classes)
        dice_scores_sum_per_class = torch.zeros(num_active, device=self.device)

        logged_images = False # Flag to log images only once per epoch  

        progress_bar = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Validate]", leave=False)

        with torch.no_grad():
            try:
                for batch_idx, (images, masks) in enumerate(progress_bar):
                    images, masks = images.to(self.device), masks.to(self.device)

                    outputs = self.model(images, "inference")
                    current_weights = self.model.get_ds_weights()
                    loss = compute_ds_loss(self.criterion, outputs, masks, current_weights, self.device)

                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf validation loss encountered at epoch {epoch}, batch {batch_idx}. Skipping batch metrics.")
                        continue 

                    val_loss += loss.item()

                    fused_output = call_fusion_fn(self.fusion_fn, outputs=outputs, weights=current_weights)  
                    dice_scores_batch_per_class = dice_coefficient_classes(
                        fused_output,
                        masks,
                        self.num_classes,
                        ignore_index=ignore_index
                    )

                    dice_scores_sum_per_class += torch.stack(dice_scores_batch_per_class)
                    dice_per_batch_avg = torch.mean(torch.stack(dice_scores_batch_per_class)).item()
                    dice_score_total += dice_per_batch_avg 

                    # --- Log segmentation examples to WandB (once per epoch) ---
                    if self.wandb_logger and not logged_images and batch_idx == 0: # Log first batch
                        self.wandb_logger.log_segmentation_masks(
                            images=images, 
                            true_masks=masks, 
                            pred_masks=fused_output,
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
        return avg_val_loss, avg_dice_score, avg_dice_per_class

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
                    outputs = self.model(images, "inference")
                    
                    current_weights = self.model.get_ds_weights()
                    fused_output = call_fusion_fn(self.fusion_fn, outputs=outputs, weights=current_weights)

                    dice_scores_batch = dice_coefficient_classes(
                        fused_output, 
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