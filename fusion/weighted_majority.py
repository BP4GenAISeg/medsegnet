import torch
import torch.nn.functional as F
from typing import Sequence
from .base_fusion import BaseFusion

class WeightedMajorityFusion(BaseFusion):
    def __init__(self, weights):
        super().__init__(weights)
        if self.weights is None:
            raise ValueError("Weights must be provided for WeightedMajorityFusion.")

    def fuse(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        B, C = outputs[0].shape[:2]
        target_shape = outputs[0].shape[2:]  # Use shape of the highest resolution output
        device = outputs[0].device
        dtype = outputs[0].dtype

        combined_scores = torch.zeros((B, C) + target_shape, dtype=dtype, device=device)

        for output, weight in zip(outputs, self.weights):
            if output.shape[2:] != target_shape:
                output = F.interpolate(output, size=target_shape, mode='trilinear', align_corners=False)

            preds = torch.argmax(output, dim=1)  # (B, D, H, W)
            preds_onehot = F.one_hot(preds, num_classes=C).permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
            combined_scores += weight * preds_onehot

        final_pred = torch.argmax(combined_scores, dim=1)  # (B, D, H, W)
        return final_pred