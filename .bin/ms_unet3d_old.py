from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchio as tio

import logging

import random


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


class MSUNet3D(nn.Module):
    """
    A dynamic 3D U-Net that can adapt its depth based on the input parameter.
    It supports deep supervision outputs from multiple decoder levels.

    Additional SOTA considerations:
      - Ensure that your input volumes are large enough for the number of downsamplings.
      - Experiment with residual connections or attention mechanisms.
      - Consider using instance normalization instead of batch normalization in some cases.
      - Use appropriate loss functions (e.g. Dice loss, focal loss) and data augmentation.
    """

    def __init__(
        self,
        cfg,
    ):
        super(MSUNet3D, self).__init__()
        self.depth = cfg.architecture.depth
        self.target_shape = cfg.dataset.target_shape
        self.in_channels = cfg.architecture.in_channels
        self.n_filters = cfg.architecture.n_filters
        self.num_classes = cfg.dataset.num_classes
        self.batch_norm = cfg.architecture.batch_norm
        self.dropout = cfg.architecture.dropout
        self.logger = logging.getLogger(__name__)
        # Build the encoder pathway dynamically
        self.encoders: nn.ModuleList = nn.ModuleList()
        self.pools: nn.ModuleList = nn.ModuleList()
        self.enc_dropouts: nn.ModuleList = nn.ModuleList()
        self.logger.debug("----Full resolution----")

        for d in range(self.depth):
            in_ch = self.in_channels if d == 0 else self.n_filters * (2 ** (d - 1))
            out_ch = self.n_filters * (2**d)
            self.encoders.append(ConvBlock(in_ch, out_ch, batch_norm=self.batch_norm))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.enc_dropouts.append(nn.Dropout3d(self.dropout))
            self.logger.debug(f"in: {in_ch}\t\tout: {out_ch}")

        # Bottleneck layer (center block, bottom of the U-Net).
        bn_in_channels = self.n_filters * (2 ** (self.depth - 1))
        bn_out_channels = self.n_filters * (2**self.depth)
        self.bn = ConvBlock(bn_in_channels, bn_out_channels, batch_norm=self.batch_norm)
        self.logger.debug(f"b_in: {bn_in_channels}\tb_out: {bn_out_channels}")

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
            self.logger.debug(f"in: {in_ch}\t\tout: {out_ch}")

        # Final layer convolution to map to the number of classes.
        self.final_conv = nn.Conv3d(self.n_filters, self.num_classes, kernel_size=1)

        self.logger.debug(f"f_in: {self.n_filters}\t\tf_out: {self.num_classes}")

        self.logger.debug("-----MS resolution-----")
        # ===== Multiscale Input Blocks =====
        # one MSB per downsampling level (excluding full resolution)
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
            self.logger.debug(
                f"in: {self.in_channels}\t\tout: {self.n_filters * (2 ** k)}"
            )
            # 1x1 head to produce segmentation at that scale,
            # otherwise, we'd just have n_filters * (2 ** k) channels/feature maps for output
            # when we want num_classes
            self.ms_heads.append(
                nn.Conv3d(self.n_filters * (2**k), self.num_classes, kernel_size=1)
            )
        self.logger.debug("-----------------------")

    def forward(self, x):
        min_size = 2**self.depth
        assert all(
            dim >= min_size for dim in x.shape[2:]
        ), f"Input spatial dimensions must be at least {min_size}, but got {x.shape[2:]}"

        encoder_feats = []
        out = x

        # Encoder pathway
        for enc, pool, dropout in zip(self.encoders, self.pools, self.enc_dropouts):
            out = enc(out)
            encoder_feats.append(out)
            out = dropout(pool(out))

        # Copying as we will pop() encoder_feats in decoder,
        # but still need the encoder features for consistency loss.
        full_enc_feats = list(encoder_feats)

        # Center
        center_out = self.bn(out)

        # Decoder pathway
        decoder_feats = []
        out = center_out

        for up_conv, decoder, dropout in zip(
            self.up_convs, self.decoders, self.dec_dropouts
        ):
            out = up_conv(out)
            skip = encoder_feats.pop()  # pop the last feature appended
            out = torch.cat([out, skip], dim=1)
            out = decoder(out)
            out = dropout(out)
            decoder_feats.append(out)

        final_out = self.final_conv(out)

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
            msb_feats.append(out_ms)  # This line i added
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

            # bottleneck
            out_ms = self.bn(out_ms)

            num_ups = self.depth - d

            # Decoder up to match MS scale
            for up_conv, dec, drop in zip(
                list(self.up_convs)[
                    :num_ups
                ],  # or remove list but # type: ignore[reportArgumentType]
                list(self.decoders)[:num_ups],
                list(self.dec_dropouts)[:num_ups],
            ):
                out_ms = up_conv(out_ms)
                skip = ms_feats.pop()  # pop the last feature appended
                out_ms = torch.cat([out_ms, skip], dim=1)
                out_ms = dec(out_ms)
                out_ms = drop(out_ms)

            ms_seg = self.ms_heads[d - 1](out_ms)
            ms_outputs.append(ms_seg)

        segmentations = (final_out, *ms_outputs)
        consistency_pairs = tuple(zip(msb_feats, full_enc_feats[1:]))

        return segmentations, consistency_pairs
