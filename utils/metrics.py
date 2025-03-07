import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, ignore_index=0, epsilon=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # Convert logits to probabilities

        #No, it doesn't pick just index 1. When you apply softmax along dim=1, it computes a probability distribution across all C classes for each voxel. 
        #This means that for every spatial position (W, H, D) in every batch, you'll have C probabilities that sum to 1, one for each class.
        probs = F.softmax(logits, dim=1)
       
        # print(targets)
        num_classes = logits.shape[1]
       
        # print(targets.shape) # B, W, H, D
        # print(logits.shape)  # B, C, W, H, D

        # Suppose num_classes is defined from logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # shape: (B, W, H, D, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # shape: (B, num_classes, W, H, D)

        # Optionally ignore specific indices
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        # Calculate dice loss over spatial dimensions
        dims = (2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
        dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, ignore_index=0):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = SoftDiceLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice
