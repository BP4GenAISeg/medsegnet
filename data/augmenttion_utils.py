
import torch
import torchio as tio
import numpy as np

def _nearest_pow2_shape(shape):
    return tuple(2 ** int(round(np.log2(s))) for s in shape)


class AugmentationUtils:

    @staticmethod
    def _get_preprocessing_list(
        target_shape,
        target_spacing=(1.0, 1.0, 1.0),
        rescale_percentiles=(0.5, 99.5)
    ):
        return [
            tio.Resample(target_spacing),
            tio.CropOrPad(target_shape, padding_mode='constant'),
            tio.RescaleIntensity((0, 1), percentiles=rescale_percentiles),
        ]

    @staticmethod
    def get_train_transforms(
        target_shape,
        target_spacing=(1.0, 1.0, 1.0),
        rescale_percentiles=(0.5, 99.5)
    ):
        preprocessing_list = AugmentationUtils._get_preprocessing_list(
            target_shape=target_shape,
            target_spacing=target_spacing,
            rescale_percentiles=rescale_percentiles
        )

        augmentations = [
            tio.RandomFlip(axes=(0, 1, 2), p=0.3),
            # tio.RandomNoise(std=0.01, p=0.2), this one is a bit weird xD but check again, visualize it!! in inference.ipynb
        ]

        all_transforms_list = preprocessing_list + augmentations

        return tio.Compose(all_transforms_list)

        
    @staticmethod
    def get_validation_transforms(
        target_shape,
        rescale_percentiles=(0.5, 99.5)
    ):
        transforms = [
            tio.RescaleIntensity((0, 1), percentiles=rescale_percentiles),
            tio.CropOrPad(target_shape, padding_mode='constant'),
        ]
        return tio.Compose(transforms) 
    
    @staticmethod
    def get_test_transforms(
        target_shape,
        rescale_percentiles=(0.5, 99.5)
    ):
        # For now, they're the same.
        return AugmentationUtils.get_validation_transforms(target_shape, rescale_percentiles)
