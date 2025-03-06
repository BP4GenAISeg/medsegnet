import numpy as np
import torch

def dice_score(y_pred, y_true, num_classes, epsilon=1e-6):
    """
    Computes Dice Score for each class and returns the average.
    - y_pred: raw logits (before softmax) with shape [B, C, D, H, W]
    - y_true: ground truth labels with shape [B, D, H, W]
    """
    dice_scores = []
    for class_idx in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (y_pred == class_idx).float()
        true_mask = (y_true == class_idx).float()
        
        # Compute intersection and union
        intersection = (pred_mask * true_mask).sum()
        total = pred_mask.sum() + true_mask.sum()
        
        dice = (2. * intersection + epsilon) / (total + epsilon)
        dice_scores.append(dice.item())
    
    return np.mean(dice_scores)  # Average Dice across classes
