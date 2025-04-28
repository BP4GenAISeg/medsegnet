import string
from networkx import omega
from numpy import divide
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Optional, Tuple
import os
from datetime import datetime
import torch
import yaml
import json
from omegaconf import OmegaConf
import re
import logging
import sys
import numpy as np
import random

from data.datasets import (
    VALID_TASKS,
    BrainTumourDataset,
    MedicalDecathlonDataset,
    ProstateDataset,
)


def setup_seed(seed: Optional[int]):
    if not seed:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import logging, os, sys
from omegaconf import DictConfig

DEFAULT_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_is_logging_configured = False


# module‐level guard
def setup_logging(logging_cfg: DictConfig, run_exp_dir: str):
    if logging_cfg is None:
        raise ValueError("Logging configuration is required.")

    global _is_logging_configured
    if _is_logging_configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    root.propagate = False

    # ---- If we don't have this, we will get default logging that logging provides us with, we dont :)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    file_cfg = logging_cfg.get("file", {})
    console_cfg = logging_cfg.get("console", {})

    file_enabled = file_cfg.get("level") is not None
    console_enabled = console_cfg.get("level") is not None

    if file_enabled:
        lvl = getattr(logging, file_cfg["level"].upper(), logging.INFO)
        fh = logging.FileHandler(os.path.join(run_exp_dir, "output.log"))
        fh.setLevel(lvl)
        fh.setFormatter(
            logging.Formatter(
                file_cfg.get("format", DEFAULT_FMT),
                datefmt=file_cfg.get("datefmt", DEFAULT_DATEFMT),
            )
        )

        root.addHandler(fh)
        root.debug(f"File logging enabled at {file_cfg['level']}")

    if console_enabled:
        lvl = getattr(logging, console_cfg["level"].upper(), logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(lvl)
        ch.setFormatter(
            logging.Formatter(
                console_cfg.get("format", DEFAULT_FMT),
                datefmt=console_cfg.get("datefmt", DEFAULT_DATEFMT),
            )
        )
        root.addHandler(ch)
        root.debug(f"Console logging enabled at {console_cfg['level']}")

    _is_logging_configured = True
    root.debug("Root logger configuration complete.")


# TODO: Consider a generalized dataset factory
# def get_dataset(task_name, *args, **kwargs):
#    if task_name not in DATASET_MAPPING:
#        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {list(DATASET_MAPPING.keys())}")
#
#    DatasetClass = DATASET_MAPPING[task_name]
#    return DatasetClass(*args)


class RunManager:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_name = cfg.get("active_architecture", {})
        self.task_name = cfg.get("active_dataset", {})

        if not self.model_name or not self.task_name:
            raise ValueError(
                "Model name and task name must be specified in the configuration."
            )

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_exp_dir = os.path.join(
            "trained_models", self.model_name, self.task_name
        )
        self.run_exp_dir = os.path.join(self.base_exp_dir, self.timestamp)
        self.best_model_path = os.path.join(self.run_exp_dir, "best_model.pth")
        self.logger = logging.getLogger(f"{self.task_name}_{self.timestamp}")

        os.makedirs(self.run_exp_dir, exist_ok=True)
        self.__save_config()
        self.logger.info(
            f"RunManager initialized for {self.model_name} on {self.task_name}."
        )
        self.logger.info(f"Experiment directory: {self.run_exp_dir}")

    def __save_config(self):
        config_path = os.path.join(self.run_exp_dir, "config.yaml")
        try:
            OmegaConf.save(self.cfg, config_path)
            self.logger.debug(f"Config saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed saving config: {e}")

    def save_model(
        self,
        model: torch.nn.Module,
        epoch: int,
        metric: float,
        optimizer=None,
        scheduler=None,
    ) -> str:
        checkpoint = {
            "epoch": epoch,
            "metric": metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(checkpoint, self.best_model_path)
        self.logger.debug(f"Saved model at epoch={epoch}, metric={metric}")
        return self.best_model_path

    def load_model(self, model, optimizer=None, scheduler=None, device: str = "cpu"):
        if not os.path.exists(self.best_model_path):
            self.logger.error(f"No checkpoint at {self.best_model_path}")
            raise FileNotFoundError(f"Checkpoint not found: {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint.get("epoch")
        metric = checkpoint.get("metric")
        self.logger.debug(f"Loaded model at epoch={epoch}, metric={metric}")
        return checkpoint

    def get_best_model_path(self) -> Optional[str]:
        return self.best_model_path if os.path.exists(self.best_model_path) else None

    # sim
