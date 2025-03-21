import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from data.data_manager import DataManager
from models.test_unet3d import UNet3DDynamic
from models.unet3d import UNet3D
from trainer import Trainer
from data.datasets import MedicalDecathlonDataset, VALID_TASKS, ProstateDataset, BrainTumourDataset
from utils.losses import get_loss_from_config
from utils.utils import RunManager, prepare_dataset_config
import random
import numpy as np

EXCLUDED_TASKS = {"Task01_BrainTumour"}
DATASET_MAPPING = {task: MedicalDecathlonDataset for task in VALID_TASKS - EXCLUDED_TASKS}
DATASET_MAPPING["Task01_BrainTumour"] = BrainTumourDataset
DATASET_MAPPING["Task05_Prostate"] = ProstateDataset

def setup_seed():
  seed = 42
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  
  np.random.seed(seed)
  random.seed(seed)
  torch.use_deterministic_algorithms(True, warn_only=True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False  

def get_config_key_or_throw(arch_cfg, key):
  return 


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
  arch_cfg = prepare_dataset_config(cfg)
  task_name = cfg.active_dataset    
  assert task_name in DATASET_MAPPING , f"Unknown dataset: {task_name}"

  run_manager = RunManager(arch_cfg, model_name="unet3d", task_name=task_name)

  gpu_device = cfg.gpu.devices[0] #TODO: Handle multiple GPUs
  device = torch.device(f"cuda:{gpu_device}") if torch.cuda.is_available() else torch.device("cpu")
  
  model = UNet3D(
      in_channels=1,
      num_classes=arch_cfg.dataset.num_classes,
      n_filters=arch_cfg.model.n_filters,
      dropout=arch_cfg.training.dropout,
      batch_norm=True,
  ).to(device)

  criterion = get_loss_from_config(arch_cfg, run_manager)

  optimizer = optim.Adam(
    model.parameters(), 
    lr=arch_cfg.training.learning_rate, 
    weight_decay=arch_cfg.training.weight_decay
  )

  dataset_class = DATASET_MAPPING[task_name]
  full_dataset = dataset_class(arch_cfg)

  data_manager = DataManager(full_dataset, arch_cfg, split_ratios=(0.80, 0.05, 0.15), seed=42)
  train_dataloader, val_dataloader, test_dataloader = data_manager.get_dataloaders()

  trainer = Trainer(
      arch_cfg, model, train_dataloader, val_dataloader, test_dataloader,
      criterion, optimizer, device, run_manager
  )
  trainer.train()
  
  trainer.test()

if __name__ == "__main__":
  setup_seed()
  main()