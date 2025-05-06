import os
import shutil
import random
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import argparse


def downsample_nifti(input_path, output_path, factor, interpolation):
    img = nib.load(input_path)
    data = img.get_fdata()
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    scale_factor = 1 / factor
    if interpolation == "trilinear":
        downsampled_tensor = F.interpolate(
            data_tensor,
            scale_factor=scale_factor,
            mode="trilinear",
            align_corners=False,
        )
    elif interpolation == "nearest":
        downsampled_tensor = F.interpolate(
            data_tensor, scale_factor=scale_factor, mode="nearest"
        )
    else:
        raise ValueError("Unsupported interpolation type")
    downsampled_data = downsampled_tensor.squeeze().numpy()
    downsampled_img = nib.Nifti1Image(downsampled_data, img.affine)
    nib.save(downsampled_img, output_path)


def create_scaled_directories(base_dir, scales):
    for scale in scales:
        scale_dir = os.path.join(base_dir, f"scale{scale}")
        os.makedirs(scale_dir, exist_ok=True)


def split_and_organize_data(input_dir, output_dir, split_ratio, scales):
    images_tr_dir = os.path.join(input_dir, "imagesTr")
    labels_tr_dir = os.path.join(input_dir, "labelsTr")

    images_ts_dir = os.path.join(output_dir, "imagesTs")
    labels_ts_dir = os.path.join(output_dir, "labelsTs")

    os.makedirs(images_ts_dir, exist_ok=True)
    os.makedirs(labels_ts_dir, exist_ok=True)

    create_scaled_directories(images_ts_dir, scales)
    create_scaled_directories(labels_ts_dir, scales)

    create_scaled_directories(os.path.join(output_dir, "imagesTr"), scales)
    create_scaled_directories(os.path.join(output_dir, "labelsTr"), scales)

    image_files = sorted(os.listdir(images_tr_dir))
    label_files = sorted(os.listdir(labels_tr_dir))

    combined = list(zip(image_files, label_files))
    random.shuffle(combined)

    split_index = int(len(combined) * split_ratio)
    train_set = combined[:split_index]
    test_set = combined[split_index:]

    for image_file, label_file in train_set:
        for scale in scales:
            factor = 2**scale
            downsample_nifti(
                os.path.join(images_tr_dir, image_file),
                os.path.join(output_dir, "imagesTr", f"scale{scale}", image_file),
                factor,
                interpolation="trilinear",
            )
            downsample_nifti(
                os.path.join(labels_tr_dir, label_file),
                os.path.join(output_dir, "labelsTr", f"scale{scale}", label_file),
                factor,
                interpolation="nearest",
            )

    for image_file, label_file in test_set:
        for scale in scales:
            factor = 2**scale
            downsample_nifti(
                os.path.join(images_tr_dir, image_file),
                os.path.join(images_ts_dir, f"scale{scale}", image_file),
                factor,
                interpolation="trilinear",
            )
            downsample_nifti(
                os.path.join(labels_tr_dir, label_file),
                os.path.join(labels_ts_dir, f"scale{scale}", label_file),
                factor,
                interpolation="nearest",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split and organize dataset into training and testing sets with scales."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing imagesTr and labelsTr.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Ratio of training to testing split (default: 0.8).",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="List of scales to generate (default: [0, 1, 2, 3]).",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.rstrip(os.sep)
    parent, name = os.path.split(input_dir)
    scaled_name = f"{name}_Scaled"
    output_dir = os.path.join(parent, scaled_name)

    split_and_organize_data(
        input_dir=args.input_dir,
        output_dir=f"{output_dir}",
        split_ratio=args.split_ratio,
        scales=args.scales,
    )
