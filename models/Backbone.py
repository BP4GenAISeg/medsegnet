import logging
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class BackboneUNet3D(nn.Module):
    def __init__(self, cfg):
        super(BackboneUNet3D, self).__init__()
        self.depth = cfg.architecture.depth
        self.target_shape = cfg.dataset.target_shape
        self.in_channels = cfg.architecture.in_channels
        self.n_filters = cfg.architecture.n_filters
        self.batch_norm = cfg.architecture.batch_norm
        self.dropout = cfg.architecture.dropout
        self.num_classes = cfg.dataset.num_classes
        self.logger = logging.getLogger(__name__)

        # Build the encoder pathway dynamically
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        for d in range(self.depth):
            in_ch = self.in_channels if d == 0 else self.n_filters * (2 ** (d - 1))
            out_ch = self.n_filters * (2**d)
            self.encoders.append(ConvBlock(in_ch, out_ch, batch_norm=self.batch_norm))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.enc_dropouts.append(nn.Dropout3d(self.dropout))

        # Bottleneck layer (center block, bottom of the U-Net).
        self.bn = ConvBlock(
            self.n_filters * (2 ** (self.depth - 1)),
            self.n_filters * (2**self.depth),
            batch_norm=self.batch_norm,
        )

        # Build the decoder pathway dynamically
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dec_dropouts = nn.ModuleList()

        # 0..3, 3..0
        for d in range(self.depth - 1, -1, -1):
            in_ch = self.n_filters * (2 ** (d + 1))
            out_ch = self.n_filters * (2**d)
            self.up_convs.append(
                nn.ConvTranspose3d(
                    in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            self.decoders.append(ConvBlock(in_ch, out_ch, batch_norm=self.batch_norm))
            self.dec_dropouts.append(nn.Dropout3d(self.dropout))

        # Final layer convolution to map to the number of classes.
        self.final_conv = nn.Conv3d(self.n_filters, self.num_classes, kernel_size=1)

    def forward(self, x):
        min_size = 2**self.depth
        assert all(
            dim >= min_size for dim in x.shape[2:]
        ), f"Input spatial dimensions must be at least {min_size}, but got {x.shape[2:]}"

        enc_feats = []
        out = x

        # Encoder pathway
        for enc, pool, drop in zip(self.encoders, self.pools, self.enc_dropouts):
            out = enc(out)
            enc_feats.append(out)
            out = drop(pool(out))

        self.enc_feats_copy = list(enc_feats)

        # Center
        center_out = self.bn(out)

        # Decoder pathway
        dec_feats = []
        out = center_out
        for up_conv, dec, drop in zip(self.up_convs, self.decoders, self.dec_dropouts):
            out = up_conv(out)
            skip = enc_feats.pop()
            out = torch.cat([out, skip], dim=1)
            out = dec(out)
            out = drop(out)
            dec_feats.append(out)

        self.dec_feats_copy = list(dec_feats)

        final = self.final_conv(out)
        return final
