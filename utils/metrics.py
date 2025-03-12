import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# # Dice score / Dice coefficient
# def dice_coefficient(outputs, masks):
#     """
#     outputs: Model predictions (B, C, D, H, W)
#     masks: Ground truth masks (B, D, H, W)
#     """
#     preds = torch.argmax(outputs, dim=1)  # Assumes class logits
#     intersection = (preds * masks).sum().float()
#     union = preds.sum() + masks.sum() + 1e-8
#     return (2.0 * intersection + 1e-8) / union  # Add epsilon to avoid division by zero

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

# new tryt
class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-class segmentation.
    The loss is a linear combination of the cross-entropy loss and the Dice loss.
    """
    def __init__(self, alpha=0.5, ignore_index=0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.eps = eps

    def dice_loss(self, pred, target,eps=1e-6):
        pred = torch.softmax(pred, dim=1)  #(B, C, D, H, W)


        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3)
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        return 1 - ((2. * intersection + eps) / (union + eps)).mean()

    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + (1 - self.alpha) * self.dice_loss(pred, target)



##### Multi-class segmentation #####
class FocalDiceLoss(nn.Module):
    """
    Credits: https://github.com/usagisukisuki/Adaptive_t-vMF_Dice_loss/blob/main/SegLoss/focal_diceloss.py
    """
    def __init__(self, n_classes, beta=2.0): # TODO change to cfg.num_classes
        super(FocalDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.beta = 1.0 / beta

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _fdice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

        loss = 1 - loss**self.beta

        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(self.n_classes):
            dice = self._fdice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
            
        return loss / self.n_classes




class CombinedLossdasdas(nn.Module):
    def __init__(self, alpha=0.5, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def dice_loss(self, pred, target, smooth=1):
        # pred: (B, C, D, H, W); target: (B, D, H, W) with integer class labels
        pred = torch.softmax(pred, dim=1)
        # target: torch.Size([2, 32, 64, 32])  
        
        # Convert integer target to one-hot encoding: (B, D, H, W, C) -> (B, C, D, H, W)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Create valid mask to ignore pixels with ignore_index
        valid_mask = (target != self.ignore_index).unsqueeze(1)  # (B, 1, D, H, W)
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss