from typing import Optional, Tuple
import numpy as np
from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F

# metrics.py:
# Precision, Recall, F1-score
# Accuracy


# ----- IoU (Intersection over Union) -----
def compute_iou_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    cls: int,
    smooth: float = 1e-6,
) -> float:
    """Compute IoU score for a single class."""
    pred_c = (preds == cls).float()
    mask_c = (masks == cls).float()
    intersection = (pred_c * mask_c).sum()
    union = pred_c.sum() + mask_c.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def iou_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[float]:
    """Return list of IoU scores for each non-ignored class."""
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        score = compute_iou_score(preds, masks, c, smooth)
        scores.append(score)
    return scores


def iou_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> float:
    """Return average IoU score across non-ignored classes."""
    iou_scores = iou_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not iou_scores:
        return 0.0
    return sum(iou_scores) / len(iou_scores)


# ----- End IoU (Intersection over Union) -----


# ----- Dice coefficient -----
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
    return (2 * intersection + smooth) / (
        sum_pred + sum_mask + smooth
    )  # FIXME consider if smooth is stupid, since sum_mask should always have a value right and if not then we want 0, but in that case we get 1 => so we properly need a if-statement of sort that saftety check this.


def _compute_dice_scores(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float,
    ignore_index: Optional[int],
) -> list[float]:
    """Helper to compute Dice scores for all classes (excluding `ignore_index`)."""
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue  # Skip ignored class
        score = compute_dice_score(preds, masks, c, smooth).item()
        scores.append(score)
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
    return sum(dice_scores) / len(dice_scores)


def dice_coefficient_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[float]:
    """Return list of Dice scores for each non-ignored class."""
    return _compute_dice_scores(preds, masks, num_classes, smooth, ignore_index)


# ----- End Dice coefficient -----


# ----- Precision, Recall, F1 -----
def compute_precision_recall_f1(
    preds: torch.Tensor,
    masks: torch.Tensor,
    cls: int,
    smooth: float = 1e-6,
) -> Tuple[float, float, float]:
    pred_c = (preds == cls).float()
    mask_c = (masks == cls).float()

    tp = (pred_c * mask_c).sum()
    fp = (pred_c * (1 - mask_c)).sum()
    fn = ((1 - pred_c) * mask_c).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    return precision.item(), recall.item(), f1.item()


def f1_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[float]:
    f1_scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        _, _, f1 = compute_precision_recall_f1(preds, masks, c, smooth)
        f1_scores.append(f1)
    return f1_scores


def f1_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> float:
    f1_scores = f1_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not f1_scores:
        return 0.0
    return sum(f1_scores) / len(f1_scores)


# ----- End Precision, Recall, F1 ----
