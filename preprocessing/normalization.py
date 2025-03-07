

import numpy as np
import argparse
import argcomplete

#our imports
from preprocessing.dimensions import nearest_power_of_two
from utils.nifti_utils import load_nifti
from utils.table import print_norm_image_stats



def is_image_normalized(image: np.ndarray, tol=1e-6) -> bool:
    """
    Checks if the image is normalized.
    An image is considered normalized if all its values are within [-1, 1],
    which covers both [0,1] and [-1,1] cases.
    
    Args:
        image (np.ndarray): The input image array.
        tol (float): Tolerance for floating point comparisons.
    
    Returns:
        bool: True if the image is normalized, False otherwise.
    """
    min_val, max_val = image.min(), image.max()
    return (min_val >= -1 - tol) and (max_val <= 1 + tol)

def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to the range [0, 1].
    Args:
        image (np.ndarray): Input image array.
    Returns:
        np.ndarray: Normalized image array.
    """
    pmin, pmax = np.percentile(image, [5, 95])
    epsilon = 1e-8  
    image = (image - pmin) / (pmax - pmin + epsilon)
    return np.clip(image, 0, 1)


# def normalize(image):
#     pmin, pmax = np.percentile(image, [5, 95])
#     # if pmin == pmax:  # Handle edge case
    
#     max_val = max(
#         nearest_power_of_two(abs(pmin)), 
#         nearest_power_of_two(abs(pmax))
#     )
    
#     image = (image + max_val) / (2 * max_val)

#     #Clip all voxel values to the range [0, 1]
#     return np.clip(image, 0, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize pixel values of a NIfTI image."
    )
    parser.add_argument("-i", "--image", type=str, help="Path to the NIfTI image", required=True)
    args = parser.parse_args()
    argcomplete.autocomplete(parser)

    image_path = args.image

    image = load_nifti(image_path)
    normalized_image = normalize(image)
    print_norm_image_stats(normalized_image)