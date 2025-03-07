# A python script for iterating through all .nii files in a dataset folder path
# Then get the labels for each .nii files add to a set, and in the end return the
# set containing all the classes.

import os
import nibabel as nib
import argparse
import argcomplete
from tqdm import tqdm

# import torch
# label = dataset[0][1]  # Get the label map from dataset
# print(torch.unique(label))  # Show unique label values
