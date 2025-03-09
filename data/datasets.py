import os
import numpy as np
from omegaconf import DictConfig
import nibabel as nib
import torch
from torch.utils.data import Dataset
from preprocessing.normalization import normalize
import scipy.ndimage as ndimage
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

def extract_task_name(path: str) -> str:
    """
    Extract the task name from a file path.
    Args:
        path (str): The file path.
    Returns:
        str: The extracted task name.
    """
    pattern = re.compile(r'(Task\d+_[A-Za-z]+)', re.IGNORECASE)
    match = pattern.search(path)
    return match.group(1).lower() if match else None


class MedicalDecathlonDataset(Dataset):
    def __init__(self, dataset_cfg: DictConfig):
        self.images_path = f"{dataset_cfg.base_path}{dataset_cfg.images_subdir}"
        self.labels_path = f"{dataset_cfg.base_path}{dataset_cfg.labels_subdir}"

        # Verify that the directories exist
        assert os.path.exists(self.images_path), f"Images path not found: {self.images_path}"
        assert os.path.exists(self.labels_path), f"Labels path not found: {self.labels_path}"
        
        # Set target shape and list files
        self.target_shape = dataset_cfg.target_shape
        self.num_classes = dataset_cfg.num_classes
        self.image_files = sorted(os.listdir(self.images_path))
        self.label_files = sorted(os.listdir(self.labels_path))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.image_files[idx])
        label_path = os.path.join(self.labels_path, self.label_files[idx])

        image = nib.load(image_path).get_fdata() # Shape: [W, H, D]
        label = nib.load(label_path).get_fdata() # Shape: [W, H, D]
        assert label.min() >= 0 and label.max() < self.num_classes, "Invalid label values!"
        
        image = resize_nd_image(image, self.target_shape, is_mask=False)
        label = resize_nd_image(label, self.target_shape, is_mask=True)

        if not is_image_normalized(image):
            image = normalize(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # Add channel dimension 
        label = torch.tensor(label, dtype=torch.long) 

        return image, label

class BrainTumourDataset(MedicalDecathlonDataset):
    """
    Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd,T2w)
    """
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image = image[:, :, :, :, 1]
        # image = image.unsqueeze(0)
        return image, label

class ProstateDataset(MedicalDecathlonDataset):
    """
    Modality: Multimodal MR (T2, ADC)
    """
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image = image[:, :, :, :, 0]
        # image = image.unsqueeze(0)
        return image, label