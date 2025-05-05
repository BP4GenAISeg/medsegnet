from fusion.fuser import OutputFuser
from models.Backbone import BackboneUNet3D, ConvBlock
import torch
from torch import nn
import torch.nn.functional as F


class DSUNet3D(BackboneUNet3D):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.weights = [1 / self.depth] * self.depth

        self.ds_levels = cfg.architecture.get("ds_levels", self.depth - 1)
        assert (
            0 < self.ds_levels < self.depth
        ), "ds_levels must be between 1 and depth-1"

        self.ds_heads = nn.ModuleList()
        for d in range(1, self.depth):
            self.ds_heads.append(
                nn.Conv3d(self.n_filters * (2**d), self.num_classes, kernel_size=1)
            )

    def forward(self, x):
        outputs = super().forward(x)
        self.dec_feats_copy.pop()

        # ===== Deep Supervision =====
        ds_outputs = []
        for d in range(1, self.ds_levels + 1):
            dec_feat = self.dec_feats_copy.pop()
            out_ds = self.ds_heads[d - 1](dec_feat)

            up_factor = 2**d
            out_ds = F.interpolate(
                out_ds, scale_factor=up_factor, mode="trilinear", align_corners=True
            )
            ds_outputs.append(out_ds)

        return (outputs, *ds_outputs)
