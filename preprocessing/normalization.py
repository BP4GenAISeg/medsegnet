

import numpy as np
import argparse
import argcomplete
import torch
#our imports
from preprocessing.dimensions import nearest_power_of_two
from utils.table import print_norm_image_stats



def is_image_normalized(image: np.ndarray, tol=1e-6) -> bool:
    """
    Checks if the image is normalized.
    which covers both [0,1].
    
    Args:
        image (np.ndarray): The input image array.
        tol (float): Tolerance for floating point comparisons.
    
    Returns:
        bool: True if the image is normalized, False otherwise.
    """
    min_val, max_val = image.min(), image.max()
    return min_val >= 0 - tol and max_val <= 1 + tol

def normalize_image_old(input_image):
    if not isinstance(input_image, torch.Tensor):
        input_image = torch.tensor(input_image, dtype=torch.float32)

    min_val, max_val = torch.quantile(input_image, torch.tensor([0.05, 0.95], device=input_image.device))

    max_val = 2 ** torch.ceil(torch.log2(torch.max(torch.abs(min_val), max_val) + 1)) - 1
    min_val = -max_val * (min_val < 0)

    input_image = (input_image - min_val) / (max_val - min_val)

    input_image = torch.clamp(input_image, 0, 1)

    return input_image


def normalize_image(image):

    pmin, pmax = np.percentile(image, [5, 95])
    # if pmin == pmax:  # Handle edge case
    
    max_val = max(
        nearest_power_of_two(abs(pmin)), 
        nearest_power_of_two(abs(pmax))
    )
    
    image = (image + max_val) / (2 * max_val + 1e-6) 
 
    #Clip all voxel values to the range [0, 1]
    image = np.clip(image, 0, 1)

    return torch.tensor(image, dtype=torch.float32) 
