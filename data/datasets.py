import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from preprocessing.normalization import normalize
import scipy.ndimage as ndimage
import numpy as np
import torch.nn.functional as F
import torch


from preprocessing.dimensions import resize_nd_image


class BaseMedicalDecathlonDataset(Dataset):
    def __init__(self, cfg, target_shape):
        self.cfg = cfg
        dataset_path = cfg.training.dataset_path
        self.images_path = os.path.join(dataset_path, cfg.training.images_subdir)
        self.labels_path = os.path.join(dataset_path, cfg.training.labels_subdir)
        self.target_shape = target_shape
        
        self.image_files = sorted(os.listdir(self.images_path))
        self.label_files = sorted(os.listdir(self.labels_path))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.image_files[idx])
        label_path = os.path.join(self.labels_path, self.label_files[idx])

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        assert label.min() >= 0 and label.max() < self.cfg.training.num_classes, "Invalid label values!"

        image = resize_nd_image(image, self.target_shape, is_mask=False)
        label = resize_nd_image(label, self.target_shape, is_mask=True)

        #one hot encoding of label perhaps?
        # label = F.one_hot(torch.tensor(label), num_classes=self.cfg.training.num_classes).permute(0, 4, 1, 2, 3)
        
        image = normalize(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: [C=1, D, H, W]
        label = torch.tensor(label, dtype=torch.long)  # Shape: [D, H, W]

        return image, label

# Dataset Loader
class HepaticVesselDataset(BaseMedicalDecathlonDataset):
    def __init__(self, cfg, target_shape):
        super().__init__(cfg, target_shape)
