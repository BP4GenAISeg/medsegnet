import unittest
import numpy as np
import torch
from data.datasets import BaseMedicalDecathlonDataset
from omegaconf import OmegaConf
from preprocessing.dimensions import resize_nd_image
import matplotlib.pyplot as plt
import os

from utils.nifti_utils import load_nifti

import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_3d_slices(original_img, resized_img, output_path, 
                         cmap="gray", 
                        titles=("Original Slice", "Resized Slice"),
                        figsize=(10, 5)):
    """
    Visualizes and saves comparison slices from 3D medical images.
    
    Args:
        original_img (np.ndarray): 3D array of original image
        resized_img (np.ndarray): 3D array of processed image
        output_path (str): Full path to save output image
        slice_axis (int): Axis along which to take slices (0=x, 1=y, 2=z)
        cmap (str): Matplotlib colormap to use
        titles (tuple): Titles for (original, resized) subplots
        figsize (tuple): Figure size in inches
    """
    if original_img.ndim != 3 or resized_img.ndim != 3:
        raise ValueError("Both images must be 3D arrays")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    original_slice = original_img[:, :, original_img.shape[2] // 2]
    ax1.imshow(original_slice, cmap=cmap)
    ax2.set_title(titles[0])
    
    resized_slice = resized_img[:, :, resized_img.shape[2] // 2]
    ax2.imshow(resized_slice, cmap=cmap)
    ax2.set_title(titles[1])

    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

class TestBaseMedicalDecathlonDataset(unittest.TestCase):

    def setUp(self):
        self.img = load_nifti("datasets/Task02_Heart/imagesTr/la_003.nii")
        assert self.img is not None, "Image not loaded"

    def test_resize_nd_image_crop(self):
        target_shape = (128, 128, 128)

        resized_img = resize_nd_image(self.img, target_shape)

        self.assertEqual(resized_img.shape, target_shape)
        np.testing.assert_array_equal(resized_img, self.img[0+96:320-96, 0+96:320-96, 0+1:130-1])

        visualize_3d_slices(
            original_img=self.img,
            resized_img=resized_img,
            output_path="tests/unet3d/images/resized_slice_crop.png",
            titles=("Original Slice (320, 320, 130)", "Resized Slice (128, 128, 128)")
        )


    def test_resize_nd_image_pad(self):
        target_shape = (400, 400, 400)
        resized_img = resize_nd_image(self.img, target_shape)

        self.assertEqual(resized_img.shape, target_shape)
        np.testing.assert_array_equal(resized_img[0+40:400-40, 0+40:400-40, 0+135:400-135], self.img)

        visualize_3d_slices(
            original_img=self.img,
            resized_img=resized_img,
            output_path="tests/unet3d/images/resized_slice_pad.png",
            titles=("Original Slice (320, 320, 130)", "Resized Slice (400, 400, 400)")
        )

    def test_resize_nd_image_no_change(self):
        target_shape = (320, 320, 130)
        resized_img = resize_nd_image(self.img, target_shape)
        
        self.assertEqual(resized_img.shape, target_shape)
        np.testing.assert_array_equal(resized_img, self.img)

    def test_resize_nd_image_mask_mode(self):
        target_shape = (320, 320, 130)
        resized_img = resize_nd_image(self.img, target_shape, is_mask=True)

        self.assertEqual(resized_img.shape, target_shape)
        np.testing.assert_array_equal(resized_img, self.img)


if __name__ == "__main__":
    unittest.main()