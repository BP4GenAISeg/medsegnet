# A python script for iterating through all .nii files in a dataset folder path
# Then get the labels for each .nii files add to a set, and in the end return the
# set containing all the classes.

import os
import nibabel as nib
import argparse
import argcomplete
import concurrent.futures
from tqdm import tqdm

from utils.nifti_utils import load_nifti_files


def get_classes(dataset_path):
    """
    Get all classes in a dataset.
    It works by iterating through all .nii files in a dataset folder path
    Then get the labels for each .nii files add to a set, and in the end return the
    set containing all the classes.
    """
    images = load_nifti_files(dataset_path)
    assert images, "No NIfTI files found in the dataset directory."

    classes = set()
    for img in images:
        classes.update(set(img.flatten()))
    return classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get all classes in a dataset."
    )
    parser.add_argument("-dp", "--dataset", type=str, help="Path to the dataset directory", required=True)
    args = parser.parse_args()
    argcomplete.autocomplete(parser)


    dataset_path = args.dataset
    classes = get_classes(dataset_path)
    print(f"Classes: {classes}")