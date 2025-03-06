

import numpy as np
import argparse
import argcomplete

#our imports
from preprocessing.dimensions import nearest_power_of_two
from utils.nifti_utils import load_nifti
from utils.table import print_norm_image_stats

def normalize(image):
    pmin, pmax = np.percentile(image, [10, 90])
    # if pmin == pmax:  # Handle edge case
    
    max_val = max(
        nearest_power_of_two(abs(pmin)), 
        nearest_power_of_two(abs(pmax))
    )
    
    image = (image + max_val) / (2 * max_val)

    #Clip all voxel values to the range [0, 1]
    return np.clip(image, 0, 1)

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