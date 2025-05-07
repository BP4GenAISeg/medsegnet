import os
from typing import Optional
import numpy as np
from omegaconf import DictConfig
from data.augmenttion_utils import AugmentationUtils
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
from utils.assertions import ensure, ensure_pexists

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
    def __init__(
        self,
        arch_cfg: DictConfig,
        phase: str,
        image_files: Optional[list] = None,
        mask_files: Optional[list] = None,
        images_path: Optional[str] = None,
        masks_path: Optional[str] = None,
    ):
        self.arch_cfg = arch_cfg
        self.phase = phase
        base = arch_cfg.dataset.base_path
        self.images_path = (
            images_path
            if images_path is not None
            else os.path.join(base, arch_cfg.dataset.images_subdir)
        )
        self.masks_path = (
            masks_path
            if masks_path is not None
            else os.path.join(base, arch_cfg.dataset.labels_subdir)
        )
        ensure_pexists(self.images_path, FileNotFoundError)
        ensure_pexists(self.masks_path, FileNotFoundError)

        if image_files is None or mask_files is None:
            self.image_files = sorted(os.listdir(self.images_path))
            self.mask_files = sorted(os.listdir(self.masks_path))
        else:
            self.image_files = image_files
            self.mask_files = mask_files

        self.target_shape = self.arch_cfg.dataset.target_shape
        self.num_classes = self.arch_cfg.dataset.num_classes

        if self.phase == "train":
            self.transform = AugmentationUtils.get_train_transforms(self.target_shape)
        elif self.phase == "val":
            self.transform = AugmentationUtils.get_validation_transforms(
                self.target_shape
            )
        elif self.phase == "test":
            self.transform = AugmentationUtils.get_test_transforms()

    def __len__(self):
        return len(self.image_files)

    def load_img_and_gts(self, idx):
        image_path = os.path.join(self.images_path, self.image_files[idx])
        image = nib.as_closest_canonical(nib.load(image_path)).get_fdata()  # [W, H, D]

        mask_path = os.path.join(self.masks_path, self.mask_files[idx])
        mask = nib.as_closest_canonical(nib.load(mask_path)).get_fdata()  # [W, H, D]
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.load_img_and_gts(idx)

        # Add channel dim and create subject
        image = image[np.newaxis]
        mask = mask[np.newaxis]

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image), mask=tio.LabelMap(tensor=mask)
        )

        # Apply transforms
        subject = self.transform(subject)

        # FIXME: We need to check if image is shaped as (D, H, W)
        # mayube the library torchio is reshparing it WE NEED TO CHECK THIS!

        # Post-processing (existing code)
        # image_tensor = .permute(0, 3, 2, 1)
        # mask_tensor = subject.mask.data.permute(0, 3, 2, 1)

        return subject.image.data, subject.mask.data.squeeze(0).long()


class BrainTumourDataset(MedicalDecathlonDataset):
    """
    Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd, T2w)
    """

    def load_img_and_gts(self, idx, mod_idx=1):
        image, mask = super().load_img_and_gts(idx)  # (W, H, D, Modalities)
        image = image[:, :, :, mod_idx]  # (W, H, D)
        # image = torch.from_numpy(image).permute(3, 2, 1, 0)             # (Modalities, W, H, D)

        return image, mask

    ""

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image, label


class ProstateDataset(MedicalDecathlonDataset):
    """
    Modality: Multimodal MR (T2, ADC)
    """

    def load_img_and_gts(self, idx, mod_idx=0):
        image, mask = super().load_img_and_gts(idx)  # (W, H, D, Modalities)
        image = image[:, :, :, mod_idx]  # (W, H, D)
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
