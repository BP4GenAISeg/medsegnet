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
import torch.nn.functional as F
import torch
import torch.nn as nn

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

class UNet3D(nn.Module):
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
                 n_filters: int = 8,
                 dropout: float = 0.5,
                 batch_norm:bool = True,
                 ds: bool = True,
    ):
        super(UNet3D, self).__init__()
        assert in_channels > 0, "in_channels must be positive"
        assert num_classes > 0, "num_classes must be positive"
        assert n_filters > 0, "n_filters must be positive"
        assert 0 <= dropout <= 1, "dropout must be between 0 and 1"
        self.ds = ds
        
        self.encoder1 = ConvBlock(in_channels, n_filters, batch_norm=batch_norm)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.drop1 = nn.Dropout3d(dropout)

        self.encoder2 = ConvBlock(n_filters, n_filters * 2, batch_norm=batch_norm)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.drop2 = nn.Dropout3d(dropout)

        self.encoder3 = ConvBlock(n_filters * 2, n_filters * 4, batch_norm=batch_norm)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.drop3 = nn.Dropout3d(dropout)

        self.encoder4 = ConvBlock(n_filters * 4, n_filters * 8, batch_norm=batch_norm)
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.drop4 = nn.Dropout3d(dropout)

        self.center = ConvBlock(n_filters * 8, n_filters * 16, batch_norm=batch_norm)

        self.up4 = nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = ConvBlock(n_filters * 8 + n_filters * 8, n_filters * 8, batch_norm=batch_norm)
        self.drop5 = nn.Dropout3d(dropout)

        self.up3 = nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = ConvBlock(n_filters * 4 + n_filters * 4, n_filters * 4, batch_norm=batch_norm)
        self.drop6 = nn.Dropout3d(dropout)

        self.up2 = nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = ConvBlock(n_filters * 2 + n_filters * 2, n_filters * 2, batch_norm=batch_norm)
        self.drop7 = nn.Dropout3d(dropout)

        self.up1 = nn.ConvTranspose3d(n_filters * 2, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = ConvBlock(n_filters + n_filters, n_filters, batch_norm=batch_norm)

        self.final_conv = nn.Conv3d(n_filters, num_classes, kernel_size=1, stride=1)
    
        # Deep supervision branches
        self.ds2 = nn.Conv3d(n_filters * 2, num_classes, kernel_size=1)  # Supervision at 1/2 resolution
        self.ds3 = nn.Conv3d(n_filters * 4, num_classes, kernel_size=1)  # Supervision at 1/4 resolution
        self.ds4 = nn.Conv3d(n_filters * 8, num_classes, kernel_size=1)  # Supervision at 1/8 resolution

    def forward(self, x): # TODO variable changes e.g. see below:
        c1 = self.encoder1(x) # e1
        p1 = self.drop1(self.pool1(c1)) # act_e1

        c2 = self.encoder2(p1)
        p2 = self.drop2(self.pool2(c2))

        c3 = self.encoder3(p2)
        p3 = self.drop3(self.pool3(c3))

        c4 = self.encoder4(p3)
        p4 = self.drop4(self.pool4(c4))

        c5 = self.center(p4)

        u6 = self.up4(c5) #d4?
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.drop5(self.decoder4(u6))

        u7 = self.up3(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.drop6(self.decoder3(u7))

        u8 = self.up2(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.drop7(self.decoder2(u8))

        u9 = self.up1(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.decoder1(u9)

        final = self.final_conv(c9)


        if self.ds and (self.training or self.use_all_outputs_during_inference):
            # Compute deep supervision outputs only in training mode
            ds2 = self.ds2(c8)
            ds2 = F.interpolate(ds2, scale_factor=2, mode='trilinear', align_corners=True)

            ds3 = self.ds3(c7)
            ds3 = F.interpolate(ds3, scale_factor=4, mode='trilinear', align_corners=True)

            ds4 = self.ds4(c6)
            ds4 = F.interpolate(ds4, scale_factor=8, mode='trilinear', align_corners=True)
            return final, ds2, ds3, ds4
        else:
            return final