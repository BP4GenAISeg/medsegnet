
import torchio as tio

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
            tio.RandomNoise(std=0.01, p=0.2),
        ]

        all_transforms_list = preprocessing_list + augmentations

        return tio.Compose(all_transforms_list)

    @staticmethod
    def get_validation_transforms(
        rescale_percentiles=(0.5, 99.5)
    ):
        validation_transforms = [
            tio.RescaleIntensity((0, 1), percentiles=rescale_percentiles),
        ]
        return tio.Compose(validation_transforms)

    @staticmethod
    def get_test_transforms(
        rescale_percentiles=(0.5, 99.5)
    ):
        # For now, they're the same.
        return AugmentationUtils.get_validation_transforms(rescale_percentiles)