import os
import concurrent.futures
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

def load_nifti_files(dataset_path, file_extension=".nii", desc="Loading NIfTI Files"):
    """
    Load NIfTI files concurrently from a specified directory.

    Args:
        dataset_path (str): The path to the directory containing NIfTI files.
        file_extension (str): The file extension to filter files by.
        desc (str): The description to display in the progress bar using tqdm.

    Returns:
        list: A list of loaded images.
    """
    nii_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(file_extension)]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(tqdm(executor.map(load_nifti, nii_files), total=len(nii_files), desc=desc))

    if any(img is None for img in images):
        return None
    return images

