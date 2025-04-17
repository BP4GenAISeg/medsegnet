# =============================================================================
#  Multiscale Brain MRI Segmentation with Deep Generative Models
# =============================================================================
#  Copyright (c) 2025 Simon & Hjalte
#  All rights reserved.
#
#  This code is part of the Bachelor Project at University of Copenhagen.
#  It may not be used, modified, or distributed without explicit permission
#  from the authors.
#
#  Authors: Simon & Hjalte
# =============================================================================

# common imports
import numpy as np
from omegaconf import DictConfig
import torch.nn.functional as F
import torch
import torch.nn as nn

from models import register_model
from models.base import ModelBase
from models.config_utils import get_common_args

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class UNet3D(ModelBase):
    """
    A 3D U-Net implementation for volumetric data segmentation with deep supervision.
    Note: This model operates on 3D volumes and can be memory-intensive. 
    For a typical input of size (batch, channels, D, H, W), expect GPU memory usage 
    to scale with the number of filters (n_filters) and volume size (D*H*W). 
    E.g., with n_filters=16 and 128x128x128 volumes, memory usage may exceed your GPU's capacity.
    """
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 n_filters: int,
                 dropout: float,
                 batch_norm:bool,
                 ds: bool,
                 inference_fusion_mode: str 
    ):
        super(UNet3D, self).__init__()
        # (1, 1, 32, 64, 32)
        assert in_channels > 0, "in_channels must be positive"
        assert num_classes > 0, "num_classes must be positive"
        assert n_filters > 0, "n_filters must be positive"
        assert 0 <= dropout <= 1, "dropout must be between 0 and 1"
        
        self.ds = ds
        self.inference_fusion_mode = inference_fusion_mode

        self.encoder1 = ConvBlock(in_channels, n_filters, batch_norm=batch_norm)
        self.pool1    = nn.MaxPool3d(2, stride=2)
        self.drop1    = nn.Dropout3d(dropout)
        # (1, 8, 16, 32, 16)

        self.ms2      = ConvBlock(in_channels, n_filters, batch_norm=batch_norm)
        self.encoder2 = ConvBlock(n_filters, n_filters * 2, batch_norm=batch_norm)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.drop2 = nn.Dropout3d(dropout)
        # (1, 16, 8, 16, 8)

        self.ms3      = ConvBlock(in_channels, n_filters * 2, batch_norm=batch_norm)
        self.encoder3 = ConvBlock(n_filters * 2, n_filters * 4, batch_norm=batch_norm)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.drop3 = nn.Dropout3d(dropout)
        # (1, 32, 4, 8, 4)

        self.ms4      = ConvBlock(in_channels, n_filters * 4, batch_norm=batch_norm)
        self.encoder4 = ConvBlock(n_filters * 4, n_filters * 8, batch_norm=batch_norm)
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.drop4 = nn.Dropout3d(dropout)
        # (1, 64, 2, 4, 2)

        """
        Bottleneck layer (center block, bottom of the U-Net).
        """
        self.center = ConvBlock(n_filters * 8, n_filters * 16, batch_norm=batch_norm)
        # (1, 128, 2, 4, 2)

        self.up4 = nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = ConvBlock(n_filters * 16, n_filters * 8, batch_norm=batch_norm)
        self.drop5 = nn.Dropout3d(dropout)
        # (1, 64, 4, 8, 4)

        self.up3 = nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = ConvBlock(n_filters * 8, n_filters * 4, batch_norm=batch_norm)
        self.drop6 = nn.Dropout3d(dropout)
        # (1, 32, 8, 16, 8)

        self.up2 = nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = ConvBlock(n_filters * 4, n_filters * 2, batch_norm=batch_norm)
        self.drop7 = nn.Dropout3d(dropout)
        # (1, 16, 16, 32, 16)

        self.up1 = nn.ConvTranspose3d(n_filters * 2, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = ConvBlock(n_filters + n_filters, n_filters, batch_norm=batch_norm)
        # (1, 8, 32, 64, 32)

        """
        Final layer 
        """
        self.final_conv = nn.Conv3d(n_filters, num_classes, kernel_size=1, stride=1)
        # (1, num_classes, 32, 64, 32)
    
        # Deep supervision branches
        self.ds4 = nn.Conv3d(n_filters * 8, num_classes, kernel_size=1)  # Supervision at 1/8 resolution
        self.ds3 = nn.Conv3d(n_filters * 4, num_classes, kernel_size=1)  # Supervision at 1/4 resolution
        self.ds2 = nn.Conv3d(n_filters * 2, num_classes, kernel_size=1)  # Supervision at 1/2 resolution

    def forward(self, x): 
        enc1 = self.encoder1(x)
        pool1 = self.drop1(self.pool1(enc1))

        enc2 = self.encoder2(pool1)
        pool2 = self.drop2(self.pool2(enc2))

        enc3 = self.encoder3(pool2)
        pool3 = self.drop3(self.pool3(enc3))

        enc4 = self.encoder4(pool3)
        pool4 = self.drop4(self.pool4(enc4))

        bottleneck = self.center(pool4)

        up4 = self.up4(bottleneck) #d4?
        up4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.drop5(self.decoder4(up4))

        up3 = self.up3(dec4)
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.drop6(self.decoder3(up3))

        up2 = self.up2(dec3)
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.drop7(self.decoder2(up2))

        up1 = self.up1(dec2)
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.decoder1(up1)

        # Final conv to get channels=num_classes
        final = self.final_conv(dec1)

        if self.ds and (self.training or self.inference_fusion_mode != 'only_final'):
            # Compute deep supervision outputs only in training mode
            ds4 = self.ds4(dec4)
            ds4 = F.interpolate(ds4, scale_factor=8, mode='trilinear', align_corners=True)

            ds3 = self.ds3(dec3)
            ds3 = F.interpolate(ds3, scale_factor=4, mode='trilinear', align_corners=True)

            ds2 = self.ds2(dec2)
            ds2 = F.interpolate(ds2, scale_factor=2, mode='trilinear', align_corners=True)
            return (final, ds2, ds3, ds4)
        else:
            return (final,)

    @classmethod
    def from_config(cls, config: DictConfig) -> 'UNet3D':
        return cls(**get_common_args(config))
