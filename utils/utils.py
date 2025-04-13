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

class RunManager:
    def __init__(self,
                 unified_cfg: DictConfig, 
    ):
        self.unified_cfg = unified_cfg 
        # Derive names from the unified config if they exist
        self.model_name = unified_cfg.get("active_architecture", "unknown_model")
        self.task_name = unified_cfg.get("active_dataset", "unknown_dataset")

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Define base experiment directory
        self.base_exp_dir = os.path.join("trained_models", self.model_name, self.task_name)
        # Define specific run directory using timestamp
        self.run_exp_dir = os.path.join(self.base_exp_dir, self.timestamp)
        # Define the path for the best model checkpoint early
        self.best_model_path = os.path.join(self.run_exp_dir, "best_model.pth")

        os.makedirs(self.run_exp_dir, exist_ok=True)
        self.__setup_logging()
        self.__save_config()
        self.logger.info(f"RunManager initialized for {self.model_name} on {self.task_name}.")
        self.logger.info(f"Experiment directory: {self.run_exp_dir}")

    def __setup_logging(self):
        """Set up logging to output.log file within the run directory"""
        self.logger = logging.getLogger(f"{self.task_name}_{self.timestamp}") # Unique logger name per run
        self.logger.setLevel(logging.INFO)
        # Prevent logs from propagating to the root logger (important for multiple runs)
        self.logger.propagate = False

        # Remove existing handlers if any (e.g., during re-runs in notebooks)
        while self.logger.handlers:
             self.logger.removeHandler(self.logger.handlers[0])

        # File Handler
        log_file = os.path.join(self.run_exp_dir, "output.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info("Logging setup complete.")
    
    def __save_config(self):
        """
        Saves the unified configuration to the run's experiment directory.
        """
        config_path = os.path.join(self.run_exp_dir, "config.yaml")
        try:
            with open(config_path, "w") as f:
                OmegaConf.save(self.unified_cfg, f)
            self.logger.info(f"Unified configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")


    def save_model(self,
                   model: torch.nn.Module,
                   epoch: int,
                   metric: float,
                   optimizer: torch.optim.Optimizer | None = None, 
                   scheduler: torch.optim.lr_scheduler._LRScheduler | None = None 
        ) -> str:
        """
        Save the current best model checkpoint to disk (overwrites previous best_model.pth).
        Includes model state, epoch, metric, and optionally optimizer/scheduler states.

        Args:
            model: The PyTorch model to save.
            optimizer: The optimizer instance (optional).
            scheduler: The learning rate scheduler instance (optional).
            epoch: The epoch number at which this checkpoint is saved.
            metric: The validation metric value (e.g., Dice score or loss) that makes this the best model so far.

        Returns:
            str: The path where the best model checkpoint was saved ('best_model.pth').
        """
        checkpoint = {
            'epoch': epoch,
            'metric': metric,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': OmegaConf.to_container(self.unified_cfg, resolve=True) 
        }

        try:
            torch.save(checkpoint, self.best_model_path)
            self.logger.info(f"Best model checkpoint saved at epoch {epoch} with metric {metric:.4f} to: {self.best_model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {self.best_model_path}: {e}")
            raise e # Re-raise the exception so the Trainer knows saving failed

        return self.best_model_path

    def load_model(self,
                   model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer | None = None, # Pass if resuming
                   scheduler: torch.optim.lr_scheduler._LRScheduler | None = None, # Pass if resuming
                   device: torch.device | str = 'cpu' # Device to load onto
                   ) -> dict:
        """
        Load the best model checkpoint ('best_model.pth') from the run's directory.
        Loads model state dict and optionally optimizer/scheduler states.

        Args:
            model: The PyTorch model instance (initialized architecture).
            optimizer: The optimizer instance (optional, for resuming).
            scheduler: The LR scheduler instance (optional, for resuming).
            device: The device to map the loaded tensors to.

        Returns:
            dict: The loaded checkpoint dictionary (useful for retrieving epoch, metric etc.).

        Raises:
            FileNotFoundError: If the best_model.pth file does not exist.
            KeyError: If the checkpoint dictionary is missing expected keys.
            Exception: For other potential loading errors.
        """
        if not os.path.exists(self.best_model_path):
            self.logger.error(f"Checkpoint file not found: {self.best_model_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {self.best_model_path}")

        try:
            # Load checkpoint onto the specified device directly
            checkpoint = torch.load(self.best_model_path, map_location=device)
            self.logger.info(f"Loading checkpoint from: {self.best_model_path}")

            # Load model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("Model state_dict loaded successfully.")
            else:
                raise KeyError("Checkpoint dictionary missing 'model_state_dict'.")

            # Load optimizer state if provided and available
            if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("Optimizer state_dict loaded successfully.")
            elif optimizer:
                self.logger.warning("Optimizer state_dict not found or empty in checkpoint, optimizer not loaded.")

            # Load scheduler state if provided and available
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("Scheduler state_dict loaded successfully.")
            elif scheduler:
                self.logger.warning("Scheduler state_dict not found or empty in checkpoint, scheduler not loaded.")

            epoch = checkpoint.get('epoch', -1)
            metric = checkpoint.get('metric', float('nan'))
            self.logger.info(f"Checkpoint loaded from epoch {epoch} with metric {metric:.4f}")

            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {self.best_model_path}: {e}")
            raise e # Re-raise the exception

    def get_best_model_path(self) -> str | None:
        """
        Returns the path to the best model checkpoint file ('best_model.pth').
        Checks if the file actually exists.

        Returns:
            str | None: The path if the file exists, otherwise None.
        """
        if os.path.exists(self.best_model_path):
            return self.best_model_path
        else:
            self.logger.warning(f"Best model path requested, but file not found: {self.best_model_path}")
            return None

    # --- Logging methods ---
    def info(self, message: str, stdout: bool = False):
        """Log a message"""
        if stdout:
            print(message)
        self.logger.info(message)

    def warning(self, message: str, stdout: bool = False):
        """Log a warning"""
        if stdout:
            print(f"WARNING: {message}") # Make warnings more visible on console
        self.logger.warning(message)

    def error(self, message: str, stdout: bool = False):
        """Log an error"""
        if stdout:
            print(f"ERROR: {message}", file=sys.stderr) # Print errors to stderr
        self.logger.error(message)

    def close_loggers(self):
        """Close all logging handlers."""
        self.logger.info("Closing log handlers.")
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)