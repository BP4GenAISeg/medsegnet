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
import numpy as np
import torchio as tio

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
    def __init__(self, arch_cfg: DictConfig, phase: str):
        self.arch_cfg = arch_cfg
        self.phase = phase
        self._init_paths()
        self.target_shape = self.arch_cfg.dataset.target_shape
        self.num_classes = self.arch_cfg.dataset.num_classes
        self.transform = self._get_transforms()

    def _init_paths(self):
        self.images_path = f"{self.arch_cfg.dataset.base_path}{self.arch_cfg.dataset.images_subdir}"
        self.masks_path = f"{self.arch_cfg.dataset.base_path}{self.arch_cfg.dataset.labels_subdir}"

        assert os.path.exists(self.images_path), f"Images path not found: {self.images_path}"
        assert os.path.exists(self.masks_path), f"Labels path not found: {self.masks_path}"
        
        self.image_files = sorted(os.listdir(self.images_path))
        self.label_files = sorted(os.listdir(self.masks_path))
        
        assert len(self.image_files) == len(self.label_files), "Mismatch between image and label files!"

    def _get_transforms(self):
        # Common preprocessing for all phases
        target_spacing = (1.0, 1.0, 1.0)  # Adjust if needed
        common_transforms = [
            tio.Resample(target_spacing),
            tio.CropOrPad(self.target_shape, padding_mode='constant'),
            tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5)),
        ]

        if self.phase == 'train':
            common_transforms += [
                # Spatial transforms (applied first)
                tio.RandomFlip(axes=(0, 1, 2), p=0.3),  # 30% chance
                # tio.RandomMotion(degrees=5, translation=5, p=0.3),
                # tio.RandomGhosting(intensity=0.3, p=0.2),
                # tio.RandomSpike(num_spikes=2, intensity=0.15, p=0.15),
                # tio.RandomAffine(
                #     scales=0.05, 
                #     degrees=5, 
                #     translation=3, 
                #     p=0.4  # Explicitly set for affine
                # ),
                # tio.RandomElasticDeformation(
                #     num_control_points=5,
                #     max_displacement=8,
                #     locked_borders=1,
                #     p=0.3
                # ),

                # # Intensity transforms
                tio.RandomNoise(std=0.01, p=0.2),
                # tio.RandomBiasField(p=0.25),
                # tio.RandomBlur(std=(0.25, 1.0), p=0.25),
                # tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.25),
            ]

        return tio.Compose(common_transforms)

    def __len__(self):
        return len(self.image_files)
    
    def load_img_and_gts(self, idx):
        image_path = os.path.join(self.images_path, self.image_files[idx])
        image = nib.as_closest_canonical(nib.load(image_path)).get_fdata()  # [W, H, D]

        mask_path = os.path.join(self.masks_path, self.label_files[idx])
        mask = nib.as_closest_canonical(nib.load(mask_path)).get_fdata()    # [W, H, D]
        return image, mask 

    
    def __getitem__(self, idx):
        image, mask = self.load_img_and_gts(idx)

        #FIXME with multiscale, resizing is only in training, no longer in test if ms is enabled.

        # Add channel dim and create subject
        image = image[np.newaxis]
        mask = mask[np.newaxis]

        #

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            mask=tio.LabelMap(tensor=mask)
        )

        # Apply transforms 
        subject = self.transform(subject)

        # Post-processing (existing code)
        # image_tensor = .permute(0, 3, 2, 1)
        # mask_tensor = subject.mask.data.permute(0, 3, 2, 1)

        return subject.image.data, subject.mask.data.squeeze(0).long()

class BrainTumourDataset(MedicalDecathlonDataset):
    """
    Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd, T2w)
    """
    def load_img_and_gts(self, idx, mod_idx=1):
        image, mask = super().load_img_and_gts(idx)                    # (W, H, D, Modalities)
        image = image[:, :, :, mod_idx]                               # (W, H, D)
        # image = torch.from_numpy(image).permute(3, 2, 1, 0)             # (Modalities, W, H, D)

        return image, mask
    ''
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
    
    


# OLD - dont delete request from simon :)
    # def __getitem__old(self, idx):
    #     image, mask = self.load_img_and_gts(idx)

    #     image = resize_nd_image(image, (32, 64, 32), is_mask=False)
    #     image = normalize_image(image)
        
    #     image = image.float().permute(2, 1, 0)                         # (D, H, W)
    #     image = image.unsqueeze(0)                                     # (C=1, D, H, W), add channel dimension (greyscale) now ready for model
        
    #     assert mask.min() >= 0 and mask.max() < self.num_classes, "Invalid mask values!"
    #     mask = resize_nd_image(mask, self.target_shape, is_mask=True)


    #     mask = torch.from_numpy(mask).permute(2, 1, 0).long()          # (D, H, W)
    #     return image, mask