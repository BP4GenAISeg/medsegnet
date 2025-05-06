from typing import Optional
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
    c: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute IoU score for a single class."""
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()
    intersection = (pred_c * mask_c).sum()
    union = pred_c.sum() + mask_c.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def iou_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
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
) -> torch.Tensor:
    """Return average IoU score across non-ignored classes."""
    iou_scores = iou_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not iou_scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(iou_scores).mean()


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
    return (2 * intersection + smooth) / (sum_pred + sum_mask + smooth)


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
        score = compute_dice_score(preds, masks, c, smooth)
        scores.append(score)
    return scores


def dice_coefficient(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Return average Dice coefficient across non-ignored classes."""
    dice_scores = _compute_dice_scores(preds, masks, num_classes, smooth, ignore_index)
    if not dice_scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(dice_scores).mean()

#TODO: We are using this below in code base, but actually is a copy of _compute_dice_scores
def dice_coefficient_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    """Return list of Dice scores for each non-ignored class."""
    return _compute_dice_scores(preds, masks, num_classes, smooth, ignore_index)


# ----- End Dice coefficient -----


# ----- Precision -----
def precision_score_class(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()
    tp = (pred_c * mask_c).sum()
    fp = (pred_c * (1 - mask_c)).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision


def precision_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        score = precision_score_class(preds, masks, c, smooth)
        scores.append(score)
    return scores


def precision_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    scores = precision_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(scores).mean()


# ----- End Precision -----


# ----- Recall -----
def recall_score_class(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()
    tp = (pred_c * mask_c).sum()
    fn = ((1 - pred_c) * mask_c).sum()
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall


def recall_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        score = recall_score_class(preds, masks, c, smooth)
        scores.append(score)
    return scores


def recall_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    scores = recall_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(scores).mean()


# ----- End Recall -----


# ----- F1 -----
def f1_score_class(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    precision = precision_score_class(preds, masks, c, smooth)
    recall = recall_score_class(preds, masks, c, smooth)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)
    return f1


def f1_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        score = f1_score_class(preds, masks, c, smooth)
        scores.append(score)
    return scores


def f1_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    scores = f1_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(scores).mean()


# ----- End F1 -----
