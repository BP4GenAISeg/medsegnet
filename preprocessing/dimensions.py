import os
import numpy as np
import nibabel as nib
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torch
import re
import concurrent.futures
from tqdm import tqdm
import json
import argparse
import argcomplete

# Our imports
from utils.nifti_utils import load_nifti_files
from utils.task import extract_task_name



def resize_nd_image(img, target_shape, is_mask=False):
    """
    Resizes an n-dimensional image or mask by either cropping (centered) or padding (symmetrical)
    to match the target shape.
    
    Args:
        img (np.ndarray): Input image array (any number of dimensions)
        target_shape (tuple): Target shape to resize to (must match number of dimensions)
        is_mask (bool): Whether the input is a mask (affects padding strategy)
        
    Returns:
        np.ndarray: Resized image/mask with exactly the target shape
    """
    if len(target_shape) != np.array(img).ndim:
        raise ValueError(f"Target shape {target_shape} must have same dimensions as input image ({img.ndim})")

    # Initialize lists to store crop/pad parameters
    slices = []         # Crop slices for each dimension
    pad_widths = []     # Padding amounts for each dimension

    # Calculate crop/pad for each dimension
    for dim, (current_size, target_size) in enumerate(zip(img.shape, target_shape)):
        # Calculate difference between current and target size
        size_diff = target_size - current_size

        # Handle cropping (current size > target size)
        if size_diff < 0:
            crop_start = (current_size - target_size) // 2
            crop_end = crop_start + target_size
            slices.append(slice(crop_start, crop_end))
            pad_widths.append((0, 0))  # No padding needed
            
        # Handle padding (current size < target size)
        elif size_diff > 0:
            slices.append(slice(None))  # Take entire dimension
            pad_before = size_diff // 2
            pad_after = size_diff - pad_before
            pad_widths.append((pad_before, pad_after))
            
        # No action needed
        else:
            slices.append(slice(None))
            pad_widths.append((0, 0))

    # Apply cropping first
    cropped = img[tuple(slices)]

    # Then apply padding with appropriate strategy
    if is_mask:
        # For masks, pad with 0s (background)
        padded = np.pad(cropped, pad_widths, mode='constant', constant_values=0)
    else:
        # For images, pad with edge values (avoids black borders)
        # padded = np.pad(cropped, pad_widths, mode='edge')
        padded = np.pad(cropped, pad_widths, mode='constant', constant_values=0)

    return padded

def save_precomputed_dimensions(dim_dict, filename="./data/precomputed_dimensions.json"):
    existing_data = load_precomputed_dimensions(filename)
    existing_data.update(dim_dict)
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

def load_precomputed_dimensions(filename="./data/precomputed_dimensions.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {} 

def nearest_power_of_two(value):
    return int(2 ** np.round(np.log2(value))) if value > 0 else 0

def get_save_dir(task):
    return os.path.join("preprocessing", "images", task if task else "other")

def plot_dimension_boxplot(dims, n_dim, save_dir):
    plt.figure(figsize=(8, 6))
    plt.boxplot(dims, tick_labels=[f"dim {i}" for i in range(1, n_dim + 1)])
    plt.ylabel("Image Dimension Size (Pixels)")
    plt.title(f"Distribution of Image Shapes ({n_dim}D) Across NIfTI Files")
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dimensions_boxplot.png")
    plt.savefig(save_path)

def precompute_dimensions(folder_path):
    assert os.path.exists(folder_path), f"Folder {folder_path} does not exist"
    
    task = extract_task_name(folder_path)
    save_dir = get_save_dir(task)

    precomputed_dims = load_precomputed_dimensions()
    plot_path = os.path.join(save_dir, "dimensions_boxplot.png")

    if all([precomputed_dims, task in precomputed_dims, os.path.exists(plot_path)]):
        print(f"Precomputed dimensions found for {folder_path}.")
        return precomputed_dims

    images = load_nifti_files(folder_path)
    assert images, "No NIfTI files found in the dataset directory."

    n_dim = len(images[0].shape)

    # Assert length of dimensions are the same
    assert all(len(img.shape) == n_dim for img in images) 

    # Lists to store dimensions
    dims = []

    # Process each file
    for img in tqdm(images, total=len(images), desc="Processing Dimensions"):
        for i in range(n_dim):
            if len(dims) <= i:
                dims.append([])
            dims[i].append(img.shape[i])

    percentiles = [np.percentile(dim, 90) for dim in dims]

    for i in range(len(dims)):
        print(f"90th Percentile for Dimension {i + 1}: {percentiles[i]}")

    plot_dimension_boxplot(dims, n_dim, save_dir)

    precomputed_dims = list(map(nearest_power_of_two, percentiles))
    precomputed_dims_dict = {task: precomputed_dims}
    save_precomputed_dimensions(precomputed_dims_dict)
    print(f"Precomputed Dimensions: {precomputed_dims}")
    return precomputed_dims

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute dimensions for dataset images."
    )
    parser.add_argument("-dp", "--dataset", type=str, help="Path to the dataset directory", required=True)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    precompute_dimensions(args.dataset)