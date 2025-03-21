import string
from networkx import omega
from numpy import divide
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Tuple
import os
from datetime import datetime
import torch
import yaml
import json
from omegaconf import OmegaConf
import re
import logging

def prepare_dataset_config(cfg: DictConfig) -> DictConfig:
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


class RunManager:
    def __init__(self, 
                 arch_cfg: DictConfig, 
                 model_name: str, 
                 task_name: str
        ):
        self.arch_cfg = arch_cfg
        self.model_name = model_name
        self.task_name = task_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = os.path.join("trained_models", model_name, task_name, self.timestamp)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.__save_config()
        self.__setup_logging()

    def __setup_logging(self):
        """Set up logging to output.log file"""
        self.logger = logging.getLogger(self.task_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        fh = logging.FileHandler(os.path.join(self.exp_dir, "output.log"))
        fh.setLevel(logging.INFO)
            
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)

    def __save_config(self):
        """
        Saves the configuration to disk. This is called internally when the RunManager is created.
        Usually at the start of the training process. Such that, if you run multiple experiments, the 
        configuration is stored at the beginning of each experiment, and don't get overwritten.
        """
        with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
            OmegaConf.save(self.arch_cfg, f)

    def save_model(self, model: torch.nn.Module):
        """
        Save the current best model to disk. Overwrites the previous best model.
        """
        torch.save(model.state_dict(), os.path.join(self.exp_dir, "best_model.pth"))

    def load_model(self, model: torch.nn.Module):
        """
        Load the best model from disk.
        """
        model.load_state_dict(torch.load(os.path.join(self.exp_dir, "best_model.pth")))

    def info(self, message: str, stdout: bool = False):
        """Log a message to the output.log file"""
        if stdout:
            print(message)

        self.logger.info(message)
    
    def warning(self, message: str, stdout: bool = False):
        """Log a warning to the output.log file"""
        if stdout:
            print(message)
        self.logger.warning(message)
    
    def error(self, message: str, stdout: bool = False):
        if stdout:
            print(message)
        """Log an error to the output.log file"""
        self.logger.error(message)

def inference(model, data, device):
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))  # Single output (logits)
        probabilities = torch.softmax(output, dim=1)  # Convert to probabilities if needed
        predictions = torch.argmax(output, dim=1)  # Get class predictions
    return predictions, probabilities