from models.Backbone import BackboneUNet3D, ConvBlock
import torch
from torch import nn
import torch.nn.functional as F


class MSUNet3D(BackboneUNet3D):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.msb_blocks = nn.ModuleList()
        self.ms_heads = nn.ModuleList()
        for k in range(1, self.depth):
            self.msb_blocks.append(
                ConvBlock(
                    self.in_channels,
                    self.n_filters * (2**k),
                    batch_norm=self.batch_norm,
                )
            )
            self.ms_heads.append(
                nn.Conv3d(self.n_filters * (2**k), self.num_classes, kernel_size=1)
            )

    def forward(self, x):
        full_seg = super().forward(x)

        # ===== Multiscale inputs (during training) =====
        ms_outputs = []
        D, H, W = x.shape[2:]
        msb_feats = []  # msb1, msb2, msb3
        for d in range(1, self.depth):
            # Downsampling
            target_size = (D // (2**d), H // (2**d), W // (2**d))
            x_ms = F.interpolate(
                x, size=target_size, mode="trilinear", align_corners=True
            )

            # Build encoder features for MS path
            ms_feats = []
            msb = self.msb_blocks[d - 1]
            out_ms = msb(x_ms)
            msb_feats.append(out_ms)
            ms_feats.append(out_ms)
            out_ms = self.pools[d](out_ms)
            out_ms = self.enc_dropouts[d](out_ms)

            for enc, pool, dropout in zip(
                list(self.encoders)[d + 1 :],
                list(self.pools)[d + 1 :],
                list(self.enc_dropouts)[d + 1 :],
            ):
                out_ms = enc(out_ms)
                ms_feats.append(out_ms)
                out_ms = dropout(pool(out_ms))

            # Bottleneck
            out_ms = self.bn(out_ms)

            num_ups = self.depth - d

            # Decoder up to match MS scale
            for up_conv, dec, drop in zip(
                list(self.up_convs)[:num_ups],
                list(self.decoders)[:num_ups],
                list(self.dec_dropouts)[:num_ups],
            ):
                out_ms = up_conv(out_ms)
                skip = ms_feats.pop()
                out_ms = torch.cat([out_ms, skip], dim=1)
                out_ms = dec(out_ms)
                out_ms = drop(out_ms)

            ms_seg = self.ms_heads[d - 1](out_ms)
            ms_outputs.append(ms_seg)

        segmentations = (full_seg, *ms_outputs)
        consistency_pairs = tuple(zip(msb_feats, self.enc_feats_copy[1:]))

        return segmentations, consistency_pairs


class AlternativeMSUNet3D(MSUNet3D):
    def forward(self, x):
        D, H, W = x.shape[2:]
        outputs = super().forward(x)
        ms_outputs = [
            ms_head(dec_feat)
            for ms_head, dec_feat in zip(self.ms_heads, self.dec_feats_copy[1:])
        ]
        msb_feats = []
        for d in range(1, self.depth):
            target_size = (D // (2**d), H // (2**d), W // (2**d))
            downsampled_x = F.interpolate(
                x, size=target_size, mode="trilinear", align_corners=True
            )
            msb = self.msb_blocks[d - 1]
            out_ms = msb(downsampled_x)
            msb_feats.append(out_ms)
        segmentations = (outputs, *ms_outputs)
        consistency_pairs = tuple(zip(msb_feats, self.enc_feats_copy[1:]))
        return segmentations, consistency_pairs
