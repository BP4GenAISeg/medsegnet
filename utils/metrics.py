from hamcrest import none
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# metrics.py:
# Dice coefficient
# IoU (Intersection over Union)
# Precision, Recall, F1-score
# Accuracy

def dice_coefficient(outputs, masks, num_classes, smooth=1e-6): # TODO change to cfg.num_classes
    """
    Compute the Dice coefficient for multi-class segmentation.
    
    Args:
        outputs (torch.Tensor): Model predictions (B, C, D, H, W) with logits.
        masks (torch.Tensor): Ground truth masks (B, D, H, W) with integer labels.
        num_classes (int): Number of classes (default: 3).
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6).
    
    Returns:
        float: Average Dice coefficient across all classes.
    """
    preds = torch.argmax(outputs, dim=1)  # Shape: (B, D, H, W)
    dice_total = 0.0
    
    for c in range(num_classes):
        pred_c = (preds == c).float()  # Binary mask for class c in predictions
        mask_c = (masks == c).float()  # Binary mask for class c in ground truth
        intersection = (pred_c * mask_c).sum()  # Sum of overlapping pixels
        sum_pred = pred_c.sum()  # Total predicted pixels for class c
        sum_mask = mask_c.sum()  # Total ground truth pixels for class c
        dice = (2 * intersection + smooth) / (sum_pred + sum_mask + smooth)
        dice_total += dice
    
    return dice_total / num_classes  # Average over all classes


# TODO move it to a better place