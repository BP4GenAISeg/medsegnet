import os
from tqdm import tqdm
import nibabel as nib


def load_nifti(file_path):
    """
    Load a NIfTI file and return as a NumPy array.

    Args:
        file_path (str): The path to the NIfTI file.

    Returns:
        np.ndarray: The loaded image, if successful. None otherwise.
    """
    try:
        return nib.load(file_path).get_fdata()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

