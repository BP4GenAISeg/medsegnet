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


def setup_seed(seed: Optional[int]):
  if not seed: return
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  
  np.random.seed(seed)
  random.seed(seed)
  torch.use_deterministic_algorithms(True, warn_only=True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False  


def old_prepare_dataset_config(cfg: DictConfig) -> DictConfig:
    """
    Prepares a unified configuration by merging architecture-specific defaults,
    training defaults, and dataset-specific settings for the active dataset.
    """
    if "active_dataset" not in cfg or cfg.active_dataset not in cfg.datasets:
        raise KeyError(f"Active dataset '{cfg.active_dataset}' not found in datasets.")
    
    if "active_architecture" not in cfg or cfg.active_architecture not in cfg.architectures:
        raise KeyError(f"Active architecture '{cfg.active_architecture}' not found in architectures.")
    
    dataset_cfg: DictConfig = cfg.datasets[cfg.active_dataset]
    arch_cfg: DictConfig = cfg.architectures[cfg.active_architecture]

    # Merge model defaults with any dataset-specific model overrides
    model_overrides = dataset_cfg.get(f"{cfg.active_architecture}_overrides", {}).get("model", {})
    merged_model = OmegaConf.merge(arch_cfg.get("model_defaults", {}), model_overrides)

    training_overrides = dataset_cfg.get(f"{cfg.active_architecture}_overrides", {}).get("training", {})
    merged_training = OmegaConf.merge(arch_cfg.get("training_defaults", {}), training_overrides)

    unified_cfg = OmegaConf.create({
        "model": merged_model,
        "training": merged_training,
        "dataset": dataset_cfg
    })

    # Clean the dataset config by removing architecture-specific overrides
    dataset_cleaned = OmegaConf.create({
        k: v for k, v in dataset_cfg.items()
        if k != f"{cfg.active_architecture}_overrides"
    })
    unified_cfg.dataset = dataset_cleaned

    if not isinstance(unified_cfg, DictConfig):
        raise TypeError("Unified config is not a DictConfig. Check your configuration structure.")

    return unified_cfg


def prepare_dataset_config(cfg: DictConfig) -> DictConfig:
    """
    Prepares a unified configuration by merging architecture-specific defaults,
    training defaults, and dataset-specific settings for the active dataset.
    """
    if "active_dataset" not in cfg or cfg.active_dataset not in cfg.datasets:
        raise KeyError(f"Active dataset '{cfg.get('active_dataset', 'N/A')}' not found or specified in datasets config.")

    if "active_architecture" not in cfg or cfg.active_architecture not in cfg.architectures:
        raise KeyError(f"Active architecture '{cfg.get('active_architecture', 'N/A')}' not found or specified in architectures config.")

    dataset_name = cfg.active_dataset
    arch_name = cfg.active_architecture

    if dataset_name not in cfg.datasets:
         raise KeyError(f"Configuration for dataset '{dataset_name}' not found under 'datasets'.")
    if arch_name not in cfg.architectures:
         raise KeyError(f"Configuration for architecture '{arch_name}' not found under 'architectures'.")

    dataset_cfg = cfg.datasets[dataset_name]
    arch_cfg = cfg.architectures[arch_name]

    # Start with base defaults
    merged_model = arch_cfg.get("model_defaults", OmegaConf.create({}))
    merged_training = arch_cfg.get("training_defaults", OmegaConf.create({}))

    # Merge dataset-specific overrides if they exist
    dataset_overrides = dataset_cfg.get("overrides", OmegaConf.create({}))
    model_overrides = dataset_overrides.get("model", OmegaConf.create({}))
    training_overrides = dataset_overrides.get("training", OmegaConf.create({}))

    merged_model = OmegaConf.merge(merged_model, model_overrides)
    merged_training = OmegaConf.merge(merged_training, training_overrides)

    # Create the final structure
    # Make sure dataset config doesn't include the 'overrides' structure itself
    cleaned_dataset_cfg = OmegaConf.create({k: v for k, v in dataset_cfg.items() if k != 'overrides'})

    unified_cfg = OmegaConf.create({
        "model": merged_model,
        "training": merged_training,
        "dataset": cleaned_dataset_cfg,
        "tracking": cfg.get("tracking", OmegaConf.create({})),
        "logging": cfg.get("logging", OmegaConf.create({})), 
        "active_dataset": dataset_name,
        "active_architecture": arch_name
    })

    # Optional: Add back some top-level keys if needed elsewhere, like tracking
    # unified_cfg.tracking = cfg.get("tracking", {})

    if not isinstance(unified_cfg, DictConfig):
        raise TypeError("Unified config generation failed. Check configuration structure.")

    return unified_cfg

# def prepare_dataset_config(cfg: DictConfig) -> DictConfig:
#     """
#     Prepares a unified configuration by merging architecture-specific defaults,
#     training defaults, and dataset-specific settings for the active dataset.
#     """
#     if "active_dataset" not in cfg or cfg.active_dataset not in cfg.datasets:
#         raise KeyError(f"Active dataset '{cfg.active_dataset}' not found in datasets.")
    
#     if "active_architecture" not in cfg or cfg.active_architecture not in cfg.architectures:
#         raise KeyError(f"Active architecture '{cfg.active_architecture}' not found in architectures.")
    
#     dataset_cfg: DictConfig = cfg.datasets[cfg.active_dataset]
#     arch_cfg: DictConfig = cfg.architectures[cfg.active_architecture]

#     # Merge model defaults with any dataset-specific model overrides
#     # model_overrides = dataset_cfg.get(f"{cfg.active_architecture}_overrides", {}).get("model", {})
#     merged_model = OmegaConf.merge(arch_cfg.model_defaults, dataset_cfg.overrides.model)

#     merged_training = OmegaConf.merge(arch_cfg.training_defaults, dataset_cfg.overrides.training)
#     # training_overrides = dataset_cfg.get(f"overrides", {}).get("training", {})
#     # merged_training = OmegaConf.merge(arch_cfg.get("training_defaults", {}), training_overrides)

#     unified_cfg = OmegaConf.create({
#         "model": merged_model,
#         "training": merged_training,
#         "dataset": OmegaConf.create({
#             # Clean the dataset config by removing architecture-specific overrides
#             k: v for k, v in dataset_cfg.items()
#             if k != f"overrides"
#         })
#     })

#     if not isinstance(unified_cfg, DictConfig):
#         raise TypeError("Unified config is not a DictConfig. Check your configuration structure.")

#     return unified_cfg

# logging_setup.py

import logging, os, sys
from omegaconf import DictConfig

DEFAULT_FMT     = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# module‐level guard
_is_logging_configured = False
def setup_logging(logging_cfg: DictConfig, run_exp_dir: str):
    """
    Call this exactly once per process (e.g. from RunManager.__init__).
    Subsequent calls are no‐ops.
    """
    global _is_logging_configured
    if _is_logging_configured:
        return
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    root.propagate = False

    file_cfg    = logging_cfg.get("file", {})
    console_cfg = logging_cfg.get("console", {})

    file_enabled    = file_cfg.get("level")    is not None
    console_enabled = console_cfg.get("level") is not None

    # File handler
    if file_enabled:
        lvl = getattr(logging, file_cfg["level"].upper(), logging.INFO)
        fh  = logging.FileHandler(os.path.join(run_exp_dir, "output.log"))
        fh.setLevel(lvl)
        fh.setFormatter(logging.Formatter(
            file_cfg.get("format", DEFAULT_FMT),
            datefmt=file_cfg.get("datefmt", DEFAULT_DATEFMT)
        ))
        root.addHandler(fh)
        root.debug(f"File logging enabled at {file_cfg['level']}")

    # Console handler
    if console_enabled:
        lvl = getattr(logging, console_cfg["level"].upper(), logging.INFO)
        ch  = logging.StreamHandler(sys.stdout)
        ch.setLevel(lvl)
        ch.setFormatter(logging.Formatter(
            console_cfg.get("format", DEFAULT_FMT),
            datefmt=console_cfg.get("datefmt", DEFAULT_DATEFMT)
        ))
        root.addHandler(ch)
        root.debug(f"Console logging enabled at {console_cfg['level']}")

    _is_logging_configured = True
    root.debug("Root logger configuration complete.")
    
class RunManager:
    def __init__(self, unified_cfg: DictConfig):
        self.unified_cfg = unified_cfg
        self.model_name = unified_cfg.get("active_architecture", "unknown_model")
        self.task_name = unified_cfg.get("active_dataset", "unknown_dataset")

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_exp_dir = os.path.join("trained_models", self.model_name, self.task_name)
        self.run_exp_dir = os.path.join(self.base_exp_dir, self.timestamp)
        self.best_model_path = os.path.join(self.run_exp_dir, "best_model.pth")
        self.logger = logging.getLogger(f"{self.task_name}_{self.timestamp}")
        
        os.makedirs(self.run_exp_dir, exist_ok=True)
        self.__save_config()
        self.logger.info(f"RunManager initialized for {self.model_name} on {self.task_name}.")
        self.logger.info(f"Experiment directory: {self.run_exp_dir}")

    def __save_config(self):
        config_path = os.path.join(self.run_exp_dir, "config.yaml")
        try:
            OmegaConf.save(self.unified_cfg, config_path)
            self.logger.debug(f"Config saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed saving config: {e}")

    def save_model(self, model: torch.nn.Module, epoch: int, metric: float, optimizer=None, scheduler=None) -> str:
        checkpoint = {
            'epoch': epoch,
            'metric': metric,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': OmegaConf.to_container(self.unified_cfg, resolve=True)
        }
        torch.save(checkpoint, self.best_model_path)
        self.logger.debug(f"Saved model at epoch={epoch}, metric={metric}")
        return self.best_model_path

    def load_model(self, model, optimizer=None, scheduler=None, device: str = 'cpu'):
        if not os.path.exists(self.best_model_path):
            self.logger.error(f"No checkpoint at {self.best_model_path}")
            raise FileNotFoundError(f"Checkpoint not found: {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint.get('epoch')
        metric = checkpoint.get('metric')
        self.logger.debug(f"Loaded model at epoch={epoch}, metric={metric}")
        return checkpoint

    def get_best_model_path(self) -> Optional[str]:
        return self.best_model_path if os.path.exists(self.best_model_path) else None


    # sim