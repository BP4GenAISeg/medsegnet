from omegaconf import OmegaConf
import os
import nibabel as nib
import numpy as np

cfg = OmegaConf.load("conf/config.yaml")
task_name = cfg.dataset.task_name
dataset_path = f"{cfg.dataset.base_path}{task_name}/"
labels_path = f"{dataset_path}{cfg.dataset.labels_subdir}"

label_files = [f for f in os.listdir(labels_path)
               if f.endswith('.nii')]

unique_labels = set()
for label_file in label_files:
    label = nib.load(f"{labels_path}/{label_file}")
    # print(label.get_fdata())
    print(np.array(label.get_fdata()).shape)
    unique_labels.update(np.unique(label.get_fdata()))
    break
print(unique_labels)

