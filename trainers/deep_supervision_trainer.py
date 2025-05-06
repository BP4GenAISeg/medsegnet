from typing import Any, Dict
import torch
import torch.nn.functional as F

from fusion.fuser import OutputFuser
from trainers.base_trainer import BaseTrainer
from utils.utils import resize_masks_to
import time
from typing import Any, Dict, List, Optional, Tuple
from utils import metrics  # Assuming your metrics live here
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from omegaconf import DictConfig
from data.datasets import MedicalDecathlonDataset
from utils.metric_collecter import Agg, MetricCollector
from utils.metrics import dice_coefficient_classes
from utils.wandb_logger import WandBLogger
from utils.utils import RunManager
from trainers.callbacks.early_stopping import EarlyStopping
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils.table import print_train_val_table

import logging


class DeepSupervisionTrainer(BaseTrainer):
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
        super().__init__(
            cfg,
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            lr_scheduler,
            device,
            run_manager,
            wandb_logger,
        )
        self.fusion_mode = cfg.architecture.get("fusion", "no_fuse_only_final")
        self.fuser = OutputFuser(self.fusion_mode, self.weights)

    def _compute_loss(self, outputs, masks) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for w, output in zip(self.weights, outputs):
            loss += w * super()._compute_loss(output, masks)
        return loss

    def _compute_metrics(self, outputs, masks) -> Dict[str, Any]:
        outputs = self.fuser(outputs)
        return super()._compute_metrics(outputs, masks)

