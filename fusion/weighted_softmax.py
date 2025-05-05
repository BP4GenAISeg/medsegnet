import torch
import torch.nn.functional as F
from typing import Sequence
from .base_fusion import BaseFusion

class WeightedSoftmaxFusion(BaseFusion):
    def __init__(self, weights):
        super().__init__(weights)
        if self.weights is None:
            raise ValueError("Weights must be provided for WeightedSoftmaxFusion.")

    def fuse(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        combined_prob = None
        target_shape = outputs[0].shape[2:]  # Use shape of the highest resolution output

        for output, weight in zip(outputs, self.weights):
            if output.shape[2:] != target_shape:
                output = F.interpolate(output, size=target_shape, mode='trilinear', align_corners=False)

            prob = torch.softmax(output, dim=1)
            if combined_prob is None:
                combined_prob = weight * prob
            else:
                combined_prob += weight * prob

        if combined_prob is None:
            raise ValueError("Combined probability is None. Check outputs.")

        final_pred = torch.argmax(combined_prob, dim=1)
        return final_pred