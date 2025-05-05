import time
from typing import Any, Dict, List, Optional, Tuple
from utils import metrics  # Assuming your metrics live here
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from omegaconf import DictConfig
from data.datasets import MedicalDecathlonDataset
from utils.assertions import ensure_in
from utils.metric_collecter import Agg, MetricCollector
from utils.metrics import dice_coefficient_classes
from utils.wandb_logger import WandBLogger
from utils.utils import RunManager
from trainers.callbacks.early_stopping import EarlyStopping
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils.table import print_train_val_table

import logging


class BaseTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        run_manager: RunManager,
        wandb_logger: Optional[WandBLogger],
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.weights = [1 / cfg.architecture.depth] * cfg.architecture.depth

        # Performance Optimization
        self.use_amp = cfg.training.get("use_amp", False)
        # self.scaler = torch.GradScaler("cuda", enabled=self.use_amp)
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.rm = run_manager
        self.wandb_logger = wandb_logger

        self.logger = logging.getLogger(__name__)
        self.num_epochs = cfg.training.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.class_labels = {i: f"class_{i}" for i in range(self.num_classes)}
        self.ignore_index = cfg.dataset.get("ignore_index", 0)
        # Setup early stopping if configured
        self.early_stopper = None
        early_cfg = cfg.training.get("early_stopper", None)
        if early_cfg:
            self.early_stopper = EarlyStopping(
                patience=early_cfg.get("patience", 15),
                delta=early_cfg.get("delta", 0.0),
                criterion=early_cfg.get("criterion", "loss"),
                verbose=early_cfg.get("verbose", True),
            )
        if not self.early_stopper:
            self.logger.info("No early stopping configured.")

        # Setup metric collector
        self.metric_collector = MetricCollector()
        self._setup_metric_collector_rules()

        self.logger.info(f"Initialized BaseTrainer for {self.num_epochs} epochs.")

    def _setup_metric_collector_rules(self):
        """Sets up the basic rules for the metric collector. Can be overridden."""
        self.metric_collector.set_rule("loss", Agg.MEAN)  # Overall loss for backprop
        for metric in ["dice", "iou", "precision", "recall", "f1"]:
            # Ensure 'dice' key here matches early stopping criterion if used
            self.metric_collector.set_rule(f"avg_{metric}", Agg.MEAN)
            self.metric_collector.set_rule(f"cls_{metric}", Agg.LIST_MEAN)
        self.logger.debug("Base metric collector rules set.")

    def train(self):
        start_time = time.time()

        for epoch in trange(self.num_epochs, desc="Training"):
            epoch_start_time = time.time()
            train_dict = self.train_one_epoch(epoch)
            train_dict["epoch_time"] = time.time() - epoch_start_time

            epoch_start_time = time.time()
            val_dict = self.validate(epoch)
            val_dict["epoch_time"] = time.time() - epoch_start_time

            table_str = print_train_val_table(train_dict, val_dict)
            tqdm.write(f"\nEpoch {epoch + 1} Results:\n{table_str}")

            if self.early_stopper:
                for key in ("loss", "avg_dice"):
                    ensure_in(key, val_dict, KeyError)

                stop, improved = self.early_stopper(
                    val_dict["loss"], val_dict["avg_dice"]
                )

                metric = (
                    self.early_stopper.best_dice
                    if self.early_stopper.criterion != "loss"
                    else self.early_stopper.best_loss
                )

                if improved:
                    print("BETTER:)")
                    model_save_path = self.rm.save_model(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.lr_scheduler,
                        epoch=epoch,
                        metric=metric,
                    )
                    print("BETTER:)")
                    self.logger.info(
                        f"New best model checkpoint saved to {model_save_path}"
                    )
                if stop:
                    break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes.")

    def train_one_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        self.metric_collector.reset()

        for images, masks in tqdm(
            self.train_dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False
        ):
            images, masks = images.to(self.device), masks.to(self.device)

            # Forward (and backward ops) run in FP16 (where safe) under autocast, so your convolutions, linears, etc. execute faster and use less memory.
            # GradScaler “scales” your loss by a big factor so that the FP16 gradients don’t underflow to zero.
            # After backward(), GradScaler “unscales” those gradients back down to their true FP32 magnitude.
            # So you get the speed & memory wins of FP16 math, but all weight‐updates happen in FP32 accuracy.
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self._compute_loss(outputs, masks)
            loss.backward()
            clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.training.get("grad_clip_norm", 1.0),
            )
            self.optimizer.step()

            # preds = torch.argmax(outputs, dim=1)
            batch_metrics = self._compute_metrics(outputs, masks)
            batch_metrics["loss"] = loss.item()
            self.metric_collector.update(batch_metrics)
        result = self.metric_collector.aggregate()
        result["ep"] = epoch
        return result

    def validate(self, epoch: int) -> Dict[str, Any]:
        self.model.eval()
        self.metric_collector.reset()

        with torch.no_grad():
            for images, masks in tqdm(
                self.val_dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False
            ):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self._compute_loss(outputs, masks)
                # preds = torch.argmax(outputs, dim=1)
                batch_metrics = self._compute_metrics(outputs, masks)
                batch_metrics["loss"] = loss.item()
                self.metric_collector.update(batch_metrics)
        result = self.metric_collector.aggregate()
        result["ep"] = epoch
        return result

    def test(self) -> Dict[str, Any]:
        self.model.eval()
        self.rm.load_model(self.model)
        self.metric_collector.reset()

        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Testing"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                # preds = torch.argmax(outputs, dim=1)
                batch_metrics = self._compute_metrics(outputs, masks)
                batch_metrics["loss"] = self._compute_loss(outputs, masks).item()
                self.metric_collector.update(batch_metrics)
        result = self.metric_collector.aggregate()
        return result

    def _compute_loss(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return self.criterion(outputs, masks)

    def _compute_metrics(
        self, outputs: torch.Tensor, masks: torch.Tensor
    ) -> Dict[str, Any]:
        preds = torch.argmax(outputs, dim=1)

        # Compute all per-class metrics as tensors
        dice_scores_cls = metrics.dice_coefficient_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        iou_scores_cls = metrics.iou_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        precision_scores_cls = metrics.precision_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        recall_scores_cls = metrics.recall_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        f1_scores_cls = metrics.f1_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )

        # Average metrics
        avg_dice = (
            torch.stack(dice_scores_cls).mean().item() if dice_scores_cls else 0.0
        )
        avg_iou = torch.stack(iou_scores_cls).mean().item() if iou_scores_cls else 0.0
        avg_precision = (
            torch.stack(precision_scores_cls).mean().item()
            if precision_scores_cls
            else 0.0
        )
        avg_recall = (
            torch.stack(recall_scores_cls).mean().item() if recall_scores_cls else 0.0
        )
        avg_f1 = torch.stack(f1_scores_cls).mean().item() if f1_scores_cls else 0.0

        return {
            "avg_dice": avg_dice,
            "avg_iou": avg_iou,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "cls_dice": [x.item() for x in dice_scores_cls],
            "cls_iou": [x.item() for x in iou_scores_cls],
            "cls_precision": [x.item() for x in precision_scores_cls],
            "cls_recall": [x.item() for x in recall_scores_cls],
            "cls_f1": [x.item() for x in f1_scores_cls],
        }
