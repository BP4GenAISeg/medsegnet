from typing import Any, Dict
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
        segs, _ = outputs

        # Collect individual metrics
        per_output_metrics = []

        for seg in segs:
            gt = resize_masks_to(seg, masks)
            # We call super()._compute_metrics which expects *logits* not predictions!
            metric = super()._compute_metrics(seg, gt)
            per_output_metrics.append(metric)

        # Now aggregate across outputs
        agg_metrics = {}

        # Initialize empty lists for each metric
        for key in per_output_metrics[0].keys():
            agg_metrics[key] = []

        # Fill lists
        for metric in per_output_metrics:
            for key, value in metric.items():
                if isinstance(value, list):
                    agg_metrics[key].append(torch.tensor(value, device=self.device))
                else:
                    agg_metrics[key].append(torch.tensor(value, device=self.device))

        # Now average across outputs
        final_metrics = {}
        for key, values in agg_metrics.items():
            stacked = torch.stack(values, dim=0)
            if stacked.ndim == 2:  # class-wise metrics like cls_dice, cls_iou
                mean = stacked.mean(dim=0).tolist()  # mean per class
            else:  # scalar metrics like avg_dice, avg_iou
                mean = stacked.mean().item()
            final_metrics[key] = mean

        return final_metrics