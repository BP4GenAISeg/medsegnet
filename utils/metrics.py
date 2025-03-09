import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# new tryt
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
    
    def dice_loss(self, pred, target, smooth=1):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3)
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        return 1 - ((2. * intersection + smooth) / (union + smooth)).mean()

    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + (1 - self.alpha) * self.dice_loss(pred, target)
