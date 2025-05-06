import torch
from typing import Sequence
from .base_fusion import BaseFusion

class OnlyFinalFusion(BaseFusion):
    def fuse(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        final_output = outputs[0]
        if final_output.ndim != 5:
            raise ValueError(f"Expected 5D tensor for the final output, got {final_output.ndim}D tensor.")

        probability_logits = torch.softmax(final_output, dim=1)  # probs: (B, C, D, H, W)
        return probability_logits