from typing import Any, Dict, List
import torch
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from utils.utils import resize_masks_to


class MultiscaleTrainer(BaseTrainer):
    def _compute_loss(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        segs, cons_pairs = outputs
        return self._compute_losses(segs, cons_pairs, masks)

    def _compute_consistency_loss(self, cons_pairs):
        cons_loss = torch.tensor(0.0, device=self.device)
        for w, (ms_feats, enc_feats) in zip(self.weights[1:], cons_pairs):
            cons_loss += w * F.mse_loss(ms_feats, enc_feats.detach())
        return cons_loss

    def _compute_segmentation_loss(self, segs, targets):
        seg_loss = torch.tensor(0.0, device=self.device)

        for w, pred in zip(self.weights, segs):
            gt = (
                F.interpolate(
                    targets.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest"
                )
                .squeeze(1)
                .long()
            )
            seg_loss += w * self.criterion(pred, gt)
        return seg_loss

    def _compute_losses(self, segs, cons_pairs, targets):
        seg_loss = self._compute_segmentation_loss(segs, targets)
        cons_loss = self._compute_consistency_loss(cons_pairs)
        return seg_loss + cons_loss

    def _compute_metrics(self, outputs, masks) -> Dict[str, Any]:
        """
        Compute and return both aggregated and per-scale metrics,
        including per-class per-scale metrics as nested lists.

        Returns keys:
          - <metric>: aggregated across scales (scalar or list per class)
          - <metric>_per_scale: raw values per scale
        """
        segs, _ = outputs

        # Collect metrics for each scale
        per_output_metrics: List[Dict[str, Any]] = []
        for seg in segs:
            gt = resize_masks_to(seg, masks)
            # super()._compute_metrics expects logits, not predictions
            metric = super()._compute_metrics(seg, gt)
            per_output_metrics.append(metric)

        # Group metrics by key across scales
        agg_metrics: Dict[str, List[torch.Tensor]] = {
            key: [] for key in per_output_metrics[0].keys()
        }
        for metric in per_output_metrics:
            for key, value in metric.items():
                agg_metrics[key].append(torch.tensor(value, device=self.device))

        # Compute aggregated and per-scale metrics
        final_metrics: Dict[str, Any] = {}
        for key, tensors in agg_metrics.items():
            # Stack tensors: shape = (num_scales, ...)
            stacked = torch.stack(tensors, dim=0)

            # 1) Aggregated across scales
            if stacked.ndim == 2:
                # Class-wise metrics: mean per class
                final_metrics[key] = stacked.mean(dim=0).tolist()
            else:
                # Scalar metrics: overall mean
                final_metrics[key] = stacked.mean().item()

            # 2) Raw per-scale metrics:
            #    - Scalars: list of floats [val0, val1, ...]
            #    - Class-wise: list of lists [[scale0_c0, scale0_c1, ...], ...]
            final_metrics[f"{key}_per_scale"] = stacked.tolist()

        return final_metrics
