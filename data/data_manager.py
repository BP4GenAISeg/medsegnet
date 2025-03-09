from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split

from data.datasets import MedicalDecathlonDataset
from typing import Tuple

class DataManager:
    def __init__(
            self,
            dataset: MedicalDecathlonDataset,
            training_cfg: DictConfig,
            split_ratios: Tuple[float, float, float] = (0.80, 0.05, 0.15), 
            seed: int = 42,
        ):
            """
            Initialize the DataManager with a dataset and configuration.
            """
            if (abs(sum(split_ratios) - 1) >= 1e-6):
                raise ValueError("Split ratios must sum to 1.")
                    
            self.dataset = dataset
            self.training_cfg = training_cfg
            self.split_ratios = split_ratios
            self.seed = seed

            self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset()
            self.train_dataloader, self.val_dataloader, self.test_dataloader = self._create_dataloaders()

    def _split_dataset(self):
        """
        Split the dataset into training, validation, and test sets.
        """
        dataset_size = len(self.dataset)
        if dataset_size == 0:
            raise ValueError("Dataset is empty. Cannot split an empty dataset.")
        
        train_ratio, val_ratio, _ = self.split_ratios

        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        return random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def _create_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_cfg.batch_size,
            shuffle=True,
            drop_last=self.training_cfg.drop_last
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.training_cfg.batch_size,
            shuffle=False,
            drop_last=False
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.training_cfg.batch_size,
            shuffle=False,
            drop_last=False
        )
        return train_dataloader, val_dataloader, test_dataloader

    def get_dataloaders(self):
        """Return the train, val, and test DataLoaders."""
        return self.train_dataloader, self.val_dataloader, self.test_dataloader
    


# def create_data_splits_and_loaders(dataset, cfg, split_ratios=(0.80, 0.05, 0.15), seed=42):
#     """
#     Split a dataset into train, validation, and test sets and return corresponding DataLoaders.
    

#     Maybe switch to this instead of class and put inside utils.py??
#     """
#     assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
#     total_size = len(dataset)
#     train_size = int(split_ratios[0] * total_size)
#     val_size = int(split_ratios[1] * total_size)
#     test_size = total_size - train_size - val_size  # Avoid rounding errors
    
#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset,
#         [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(seed)
#     )
    
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=cfg.training.batch_size,
#         shuffle=True,
#         drop_last=cfg.training.drop_last
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=cfg.training.batch_size,
#         shuffle=False,
#         drop_last=False
#     )
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=cfg.training.batch_size,
#         shuffle=False,
#         drop_last=False
#     )
    
#     return train_dataloader, val_dataloader, test_dataloader