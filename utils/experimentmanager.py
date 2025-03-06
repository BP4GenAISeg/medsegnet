import os
import json
import yaml
import datetime
import torch
from omegaconf import OmegaConf
from omegaconf import DictConfig  # Just in case you need to type hint

class ExperimentManager:
    def __init__(self, cfg: DictConfig, model_name, task_name):
        """
        Handles experiment organization, config saving, and model checkpointing.
        """
        self.cfg = cfg
        self.base_dir = cfg.trained_models.base_dir
        def __create_experiment_dir():
            self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.exp_dir = os.path.join(self.base_dir, model_name, task_name, self.timestamp)
            os.makedirs(self.exp_dir, exist_ok=True)
        self.create_experiment_dir = __create_experiment_dir

    def __save_config(self):
        """Saves Hydra config as a YAML file."""
        with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
            yaml.dump(OmegaConf.to_container(self.cfg, resolve=True), f)

    def __save_model(self, model):
        """Saves model weights."""
        torch.save(model.state_dict(), os.path.join(self.exp_dir, "model.pth"))

    def __save_log(self, metrics_dict):
        """Saves training metrics like loss & dice scores."""
        with open(os.path.join(self.exp_dir, "output.log"), "w") as f:
            json.dump(metrics_dict, f, indent=4)

    def save(self, model, log_file=None):
        """Saves the model, config, and training metrics."""
        self.create_experiment_dir()
        self.__save_config()
        self.__save_model(model)
        if log_file is not None:
          self.__save_log(log_file)
    
    def get_experiment_path(self):
        """Returns the directory where this experiment is saved."""
        return self.exp_dir
