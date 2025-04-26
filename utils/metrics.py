from typing import Optional
import numpy as np
from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F

# metrics.py:
# Dice coefficient
# IoU (Intersection over Union)
# Precision, Recall, F1-score
# Accuracy


def compute_dice_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 1e-6,  # Remove unused `ignore_index`
) -> torch.Tensor:
    """Compute Dice score for a single class."""
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()
    intersection = (pred_c * mask_c).sum()
    sum_pred = pred_c.sum()
    sum_mask = mask_c.sum()
    return (2 * intersection + smooth) / (sum_pred + sum_mask + smooth) #FIXME consider if smooth is stupid, since sum_mask should always have a value right and if not then we want 0, but in that case we get 1 => so we properly need a if-statement of sort that saftety check this.


def _compute_dice_scores(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float,
    ignore_index: Optional[int],
) -> list[torch.Tensor]:
    """Helper to compute Dice scores for all classes (excluding `ignore_index`)."""
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue  # Skip ignored class
        scores.append(compute_dice_score(preds, masks, c, smooth))
    return scores


def dice_coefficient(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> float:
    """Return average Dice coefficient across non-ignored classes."""
    dice_scores = _compute_dice_scores(preds, masks, num_classes, smooth, ignore_index)
    if not dice_scores:
        return 0.0
    return torch.mean(torch.stack(dice_scores)).item()


def dice_coefficient_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    """Return list of Dice scores for each non-ignored class."""
    return _compute_dice_scores(preds, masks, num_classes, smooth, ignore_index)