import os
import numpy as np
from omegaconf import DictConfig
import nibabel as nib
import torch
from torch.utils.data import Dataset
from preprocessing.normalization import normalize_image
import numpy as np
import torch.nn.functional as F
import torch
from preprocessing.dimensions import resize_nd_image
import re
from preprocessing.normalization import is_image_normalized

VALID_TASKS = {
    "Task01_BrainTumour",
    "Task02_Heart",
    "Task03_Liver",
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task06_Lung",
    "Task07_Pancreas",
    "Task08_HepaticVessel",
    "Task09_Spleen",
    "Task10_Colon",
}

class MedicalDecathlonDataset(Dataset):
    def __init__(self, arch_cfg: DictConfig):
        self.images_path = f"{arch_cfg.dataset.base_path}{arch_cfg.dataset.images_subdir}"
        self.masks_path = f"{arch_cfg.dataset.base_path}{arch_cfg.dataset.labels_subdir}"

        assert os.path.exists(self.images_path), f"Images path not found: {self.images_path}"
        assert os.path.exists(self.masks_path), f"Labels path not found: {self.masks_path}"
        
        self.target_shape = arch_cfg.dataset.target_shape # [W, H, D]
        self.num_classes = arch_cfg.dataset.num_classes
        self.image_files = sorted(os.listdir(self.images_path))
        self.label_files = sorted(os.listdir(self.masks_path))

    def __len__(self):
        return len(self.image_files)
    
    def load_img_and_gts(self, idx):
        image_path = os.path.join(self.images_path, self.image_files[idx])
        image = nib.load(image_path).get_fdata()                     # [W, H, D]

        mask_path = os.path.join(self.masks_path, self.label_files[idx])
        mask = nib.load(mask_path).get_fdata()                       # [W, H, D]

        return image, mask

    def __getitem__(self, idx):
        
        image, mask = self.load_img_and_gts(idx)

        image = resize_nd_image(image, self.target_shape, is_mask=False)

        image = normalize_image(image)

        image = image.float().permute(2, 1, 0)       # (D, H, W)
        image = image.unsqueeze(0)                                     # (C=1, D, H, W), add channel dimension (greyscale) now ready for model
        
        assert mask.min() >= 0 and mask.max() < self.num_classes, "Invalid mask values!"
        mask = resize_nd_image(mask, self.target_shape, is_mask=True)

        mask = torch.from_numpy(mask).permute(2, 1, 0).long()          # (D, H, W)
        return image, mask

class BrainTumourDataset(MedicalDecathlonDataset):
    """
    Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd, T2w)
    """
    def load_img_and_gts(self, idx, mod_idx=1):
        image, mask = super().load_img_and_gts(idx)                    # (W, H, D, Modalities)
        image = image[:, :, :, mod_idx]                               # (W, H, D)
        return image, mask
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image, label
    
    

class ProstateDataset(MedicalDecathlonDataset):
    """
    Modality: Multimodal MR (T2, ADC)
    """
    def load_img_and_gts(self, idx, mod_idx=0):
        image, mask = super().load_img_and_gts(idx)                    # (W, H, D, Modalities)
        image = image[:, :, :, mod_idx]                               # (W, H, D)
        
        return image, mask
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image, label
    
    