from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split
import os
from data.datasets import MedicalDecathlonDataset
from typing import Tuple, Type
import numpy as np


class DataManager:
    def __init__(
            self,
            dataset_class: Type[MedicalDecathlonDataset],
            arch_cfg: DictConfig,
                seed: int,
                split_ratios: Tuple[float, float, float] = (0.80, 0.05, 0.15), 
        ):
            """
            Initialize the DataManager with a dataset and configuration.
            """
            if (abs(sum(split_ratios) - 1) >= 1e-6):
                raise ValueError("Split ratios must sum to 1.")
                    
            self.dataset_class = dataset_class
            self.arch_cfg = arch_cfg
            self.split_ratios = split_ratios
            self.seed = seed
            self.train_dataset, self.val_dataset, self.test_dataset = self._create_datasets()
            self.train_dataloader, self.val_dataloader, self.test_dataloader = self._create_dataloaders()
        
    def _create_datasets(self):
        """
        Split the dataset into training, validation, and test sets,
        keeping image and mask filenames aligned.
        """
        base = self.arch_cfg.dataset.base_path
        images_path = f"{base}{self.arch_cfg.dataset.images_subdir}"
        masks_path  = f"{base}{self.arch_cfg.dataset.labels_subdir}"

        # List them once, in sorted order
        image_names = sorted(os.listdir(images_path))
        mask_names  = sorted(os.listdir(masks_path))
        N = len(image_names)
        if N == 0:
            raise ValueError("Dataset is empty. Cannot create datasets.")
        assert N == len(mask_names), "Mismatch between total images and masks!"

        # Shuffle indices deterministically
        np.random.seed(self.seed)
        perm = np.random.permutation(N)

        # Compute split sizes
        n_train = int(self.split_ratios[0] * N)
        n_val   = int(self.split_ratios[1] * N)
        # whatever remains is test

        # Build index s
        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]

        # Subset filenames
        train_images = [image_names[i] for i in train_idx]
        train_masks  = [mask_names[i]  for i in train_idx]
        val_images   = [image_names[i] for i in val_idx]
        val_masks    = [mask_names[i]  for i in val_idx]
        test_images  = [image_names[i] for i in test_idx]
        test_masks   = [mask_names[i]  for i in test_idx]

        # Instantiate three datasets, each with its own file lists
        train_ds = self.dataset_class(
            self.arch_cfg,
            phase="train",
            image_files=train_images,
            mask_files=train_masks,
        )
        val_ds = self.dataset_class(
            self.arch_cfg,
            phase="val",
            image_files=val_images,
            mask_files=val_masks,
        )
        test_ds = self.dataset_class(
            self.arch_cfg,
            phase="test",
            image_files=test_images,
            mask_files=test_masks,
        )

        return train_ds, val_ds, test_ds

        
    def _create_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.arch_cfg.training.batch_size,
            shuffle=True,
            drop_last=self.arch_cfg.training.drop_last, 
        )
        #TODO maybe make arch_cfg.validation.batch_size
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.arch_cfg.training.batch_size,
            shuffle=False,
            drop_last=False
        )
        #TODO maybe make arch_cfg.test.batch_size
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.arch_cfg.training.batch_size,
            shuffle=False,
            drop_last=False
        )
        return train_dataloader, val_dataloader, test_dataloader

    def get_dataloaders(self):
        """Return the train, val, and test DataLoaders."""
        return self.train_dataloader, self.val_dataloader, self.test_dataloader
    