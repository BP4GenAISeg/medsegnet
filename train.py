import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from utils.experimentmanager import ExperimentManager
from models.unet3d import UNet3D
from trainer import Trainer
from utils.metrics import CombinedLoss
import os
from data.datasets import MedicalDecathlonDataset, VALID_TASKS, ProstateDataset, BrainTumourDataset

EXCLUDED_TASKS = {"Task01_BrainTumour"}
DATASET_MAPPING = {task: MedicalDecathlonDataset for task in VALID_TASKS - EXCLUDED_TASKS}
DATASET_MAPPING["Task01_BrainTumour"] = BrainTumourDataset
DATASET_MAPPING["Task05_Prostate"] = ProstateDataset

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
  task_name = cfg.dataset.task_name  # e.g., "Task08_HepaticVessel"
  assert task_name in DATASET_MAPPING, f"Unknown dataset: {task_name}"

  exp_manager = ExperimentManager(cfg, model_name="unet3d", task_name=task_name)
  gpu_device = cfg.gpu.devices[0] #TODO: Handle multiple GPUs
  device = torch.device(f"cuda:{gpu_device}") if torch.cuda.is_available() else torch.device("cpu")

  model = UNet3D(
      in_channels=1,
      num_classes=cfg.training.num_classes,
      n_filters=cfg.training.n_filters,
      dropout=cfg.training.dropout,
      batch_norm=True,
  ).to(device)

  criterion = nn.CrossEntropyLoss(ignore_index=0)
  
  optimizer = optim.Adam(
    model.parameters(), 
    lr=cfg.training.learning_rate, 
    weight_decay=cfg.training.weight_decay
  )

  dataset_path = f"{cfg.dataset.base_path}{task_name}/"
  images_path = f"{dataset_path}{cfg.dataset.images_subdir}"
  labels_path = f"{dataset_path}{cfg.dataset.labels_subdir}"

  assert os.path.exists(images_path), f"Images path not found: {images_path}"
  assert os.path.exists(labels_path), f"Labels path not found: {labels_path}"

  dataset_class = DATASET_MAPPING[task_name]
  dataset = dataset_class(
    cfg, 
    task_name, 
    images_path=images_path, 
    labels_path=labels_path,
    target_shape=cfg.dataset.target_shape
  )
  dataloader = DataLoader(
    dataset, 
    batch_size=cfg.training.batch_size, 
    shuffle=True, 
    drop_last=cfg.training.drop_last
)

  trainer = Trainer(cfg, model, dataloader, criterion, optimizer, dataset, device)
  trainer.train()

  exp_manager.save(model)

if __name__ == "__main__":
  main()