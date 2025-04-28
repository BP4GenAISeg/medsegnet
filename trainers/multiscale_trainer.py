import torch
import torch.nn.functional as F


class MultiscaleTrainer(BaseTrainer):
    def compute_loss(self, outputs, targets):
        segs, cons_pairs = outputs
        return self._compute_losses(segs, cons_pairs, targets)

    def _compute_losses(self, segs, cons_pairs, targets):
        weights = [1.0 / len(segs)] * len(segs)
        seg_loss = torch.tensor(0.0, device=self.device)
        cons_loss = torch.tensor(0.0, device=self.device)
        for w, pred in zip(weights, segs):
            gt = (
                F.interpolate(
                    targets.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest"
                )
                .squeeze(1)
                .long()
            )
            seg_loss += w * self.criterion(pred, gt)
        for w, (ms_feats, enc_feats) in zip(weights, cons_pairs):
            cons_loss += w * F.mse_loss(ms_feats, enc_feats.detach())
        return seg_loss + cons_loss

    def inference(self, images):
        strategy = self.cfg.inference.inference_input_strategy
        if strategy == "single":
            return self.model(images, strategy="single")
        elif strategy == "multi":
            return self.model(images, strategy="multi")
        elif strategy == "random":
            return self.model(images, strategy="random")
        else:
            raise ValueError(f"Unknown inference strategy: {strategy}")
