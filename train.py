import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from data.data_manager import DataManager
from models.unet3d import UNet3D
from trainer import Trainer
from utils.metrics import CombinedLoss, FocalDiceLoss
from data.datasets import MedicalDecathlonDataset, VALID_TASKS, ProstateDataset, BrainTumourDataset
from utils.utils import divide_dataset_config, RunManager
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
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False  

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
  setup_seed()
  dataset_cfg, training_cfg = divide_dataset_config(cfg)
  task_name = cfg.active_dataset    
  assert task_name in DATASET_MAPPING , f"Unknown dataset: {task_name}"

  # exp_manager = ExperimentManager(cfg, model_name="unet3d", task_name=task_name)
  gpu_device = cfg.gpu.devices[0] #TODO: Handle multiple GPUs
  device = torch.device(f"cuda:{gpu_device}") if torch.cuda.is_available() else torch.device("cpu")
  model = UNet3D(
      in_channels=1,
      num_classes=dataset_cfg.num_classes,
      n_filters=training_cfg.n_filters,
      dropout=training_cfg.dropout,
      batch_norm=True,
  ).to(device)

  criterion = CombinedLoss(alpha=0.4)

  optimizer = optim.Adam(
    model.parameters(), 
    lr=training_cfg.learning_rate, 
    weight_decay=training_cfg.weight_decay
  )

  dataset_class = DATASET_MAPPING[task_name]
  full_dataset = dataset_class(dataset_cfg)

  data_manager = DataManager(full_dataset, training_cfg, split_ratios=(0.80, 0.05, 0.15), seed=42)
  train_dataloader, val_dataloader, test_dataloader = data_manager.get_dataloaders()

  exp_manager = RunManager(dataset_cfg, training_cfg, model_name="unet3d", task_name=task_name)
  trainer = Trainer(
      dataset_cfg, training_cfg, model, train_dataloader, val_dataloader, test_dataloader,
      criterion, optimizer, device, exp_manager
  )
  trainer.train()
  
  trainer.test()

if __name__ == "__main__":
  main()