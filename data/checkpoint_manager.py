import os
from datetime import datetime
import torch
import yaml
import json
from omegaconf import OmegaConf

class CheckpointManager:
    def __init__(self, base_dir:str, model_name:str, task_name:str):
        self.base_dir = base_dir
        self.model_name = model_name
        self.task_name = task_name
        self.exp_dir = None  # Defer directory creation until first save

    def _ensure_exp_dir(self):
        """Create experiment directory with timestamp if not already set."""
        if self.exp_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.exp_dir = os.path.join(self.base_dir, self.model_name, self.task_name, timestamp)
            os.makedirs(self.exp_dir, exist_ok=True)

    def save_checkpoint(self, model, epoch, val_loss, is_best=False):
        """Save model checkpoint with epoch information"""
        self._ensure_exp_dir()  # Create directory on first save if not already done
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, os.path.join(self.exp_dir, 'last_checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(self.exp_dir, 'best_model.pth'))

    def save_config(self, cfg):
        """Save experiment configuration"""
        self._ensure_exp_dir()  # Create directory when config is saved
        with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    def save_metrics(self, metrics):
        """Save training metrics"""
        self._ensure_exp_dir()  # Just in case, though this should already exist by now
        with open(os.path.join(self.exp_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def get_exp_dir(self):
        """Return experiment directory, raising an error if not yet created."""
        if self.exp_dir is None:
            raise ValueError("Experiment directory not yet created; call a save method first.")
        return self.exp_dir