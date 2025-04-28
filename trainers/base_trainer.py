import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from omegaconf import DictConfig
from data.datasets import MedicalDecathlonDataset
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
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.rm = run_manager
        self.wandb_logger = wandb_logger

        self.logger = logging.getLogger(__name__)
        self.num_epochs = cfg.training.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.class_labels = {i: f"class_{i}" for i in range(self.num_classes)}
        self.ignore_index = cfg.dataset.get("ignore_index", 0)

        self.early_stopper = None
        early_cfg = cfg.training.get("early_stopping", None)
        if early_cfg:
            self.early_stopper = EarlyStopping(
                patience=early_cfg.get("patience", 15),
                delta=early_cfg.get("delta", 0.0),
                criterion=early_cfg.get("criterion", "loss"),
                verbose=early_cfg.get("verbose", True),
            )

        self.logger.info(f"Initialized BaseTrainer for {self.num_epochs} epochs.")

    def train(self):
        start_time = time.time()

        for epoch in trange(self.num_epochs, desc="Training"):
            epoch_start_time = time.time()
            train_dict = self.train_one_epoch(epoch)
            train_dict["epoch_time"] = time.time() - epoch_start_time

            epoch_start_time = time.time()
            val_dict = self.validate(epoch)
            val_dict["epoch_time"] = time.time() - epoch_start_time

            table = print_train_val_table(train_dict, val_dict)

            self.logger.info("\n " + table + "\n")

            if self.early_stopper:
                stop, improved = self.early_stopper(val_dict["loss"], val_dict["dice"])

                metric = (
                    self.early_stopper.best_dice
                    if self.early_stopper.criterion != "loss"
                    else self.early_stopper.best_loss
                )

                if improved:
                    model_save_path = self.rm.save_model(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.lr_scheduler,
                        epoch=epoch,
                        metric=metric,
                    )
                    self.logger.info(
                        f"New best model checkpoint saved to {model_save_path}"
                    )
                if stop:
                    break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes.")

    def train_one_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        cls_dice_sum = torch.zeros(self.num_classes, device=self.device)
        num_batches = len(self.train_dataloader)

        for images, masks in tqdm(
            self.train_dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False
        ):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self._compute_loss(outputs, masks)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            avg_dice, dice_scores = self._compute_dice(preds, masks)
            cls_dice_sum += torch.tensor(dice_scores, device=self.device)
            total_dice += avg_dice

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_cls_dice = (cls_dice_sum / num_batches).tolist()

        return {
            "ep": epoch,
            "loss": avg_loss,
            "dice": avg_dice,
            "cls_dice": avg_cls_dice,
        }

    def validate(self, epoch: int) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        cls_dice_sum = torch.zeros(self.num_classes, device=self.device)
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            for images, masks in tqdm(
                self.val_dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False
            ):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self._compute_loss(outputs, masks)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                avg_dice, dice_scores = self._compute_dice(preds, masks)
                cls_dice_sum += torch.tensor(dice_scores, device=self.device)
                total_dice += avg_dice

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_cls_dice = (cls_dice_sum / num_batches).tolist()

        return {
            "ep": epoch,
            "loss": avg_loss,
            "dice": avg_dice,
            "cls_dice": avg_cls_dice,
        }

    def test(self) -> Dict[str, Any]:
        self.model.eval()
        self.rm.load_model(self.model)

        total_dice = 0.0
        cls_dice_sum = torch.zeros(self.num_classes, device=self.device)
        num_batches = len(self.test_dataloader)

        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Testing"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                avg_dice, dice_scores = self._compute_dice(preds, masks)
                cls_dice_sum += torch.tensor(dice_scores, device=self.device)
                total_dice += avg_dice

        avg_dice = total_dice / num_batches
        avg_cls_dice = (cls_dice_sum / num_batches).tolist()

        self.logger.info(f"Test: Dice={avg_dice:.4f}")
        for i, score in enumerate(avg_cls_dice):
            if i != self.ignore_index:
                self.logger.info(f"Class {i} Dice: {score:.4f}")

        return {
            "ep": None,
            "loss": None,
            "dice": avg_dice,
            "cls_dice": avg_cls_dice,
        }

    def _compute_loss(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs, masks)

    def _compute_dice(
        self, preds: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[float, List[float]]:
        dice_scores = dice_coefficient_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        avg_dice = torch.mean(torch.tensor(dice_scores)).item()
        return avg_dice, dice_scores
