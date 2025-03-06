#!/bin/sh
# =============================================================================
# Script Name:      decompress_nii.sh
# Description:      Extracts `.nii.gz` files inside `Task0*` directories located
#                   within a specified root directory, used for medical imaging
#                   datasets (Medical Decathlon Challenge).
#
# Authors:          Simon and Hjalte
# Created on:       26-02-2025
# Last Updated:     26-02-2025
# Version:          1.0
#
# License:          [Specify license, e.g., MIT, Apache 2.0, GPL-3.0]
# =============================================================================
#
# Usage:
#   ./decompress_nii.sh <root_directory>
# Example:
#   ./decompress_nii.sh /path/to/datasets
#
# =============================================================================


if [ -z "$1" ]; then
  echo "Usage: $0 <root_directory>"
  exit 1
fi

ROOT_DIR="$1"

for dir in "$ROOT_DIR"/Task**; do
  if [ -d "$dir" ]; then  # Ensure it's a directory
    # Find and decompress all .nii.gz files recursively in subdirectories
    find "$dir" -type f -name "*.nii.gz" -exec gunzip {} \;
    find "$dir" -type f -name "._*.nii.gz" -exec rm {} \;
  fi
done