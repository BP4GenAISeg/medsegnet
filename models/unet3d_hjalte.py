
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
                 inference_fusion_mode: str ,
                 init_target_shape: tuple
    ):
        super(UNet3D, self).__init__()
        # (1, 1, 32, 64, 32)
        assert in_channels > 0, "in_channels must be positive"
        assert num_classes > 0, "num_classes must be positive"
        assert n_filters > 0, "n_filters must be positive"
        assert 0 <= dropout <= 1, "dropout must be between 0 and 1"
        
        self.ds = ds
        self.inference_fusion_mode = inference_fusion_mode
        self.init_target_shape = init_target_shape

        self.encoder1 = ConvBlock(in_channels, n_filters, batch_norm=batch_norm)
        self.pool1    = nn.MaxPool3d(2, stride=2)
        self.drop1    = nn.Dropout3d(dropout)
        # (1, 8, 16, 32, 16)

        # self.ms2      = ConvBlock(in_channels, n_filters, batch_norm=batch_norm)
        self.msb2       = ConvBlock(in_channels,  n_filters * 2, batch_norm=batch_norm)
        self.encoder2   = ConvBlock(n_filters,    n_filters * 2, batch_norm=batch_norm)
        self.pool2      = nn.MaxPool3d(2, stride=2)
        self.drop2      = nn.Dropout3d(dropout)
        # (1, 16, 8, 16, 8)

        # self.msb3       = ConvBlock(in_channels,   n_filters * 2, batch_norm=batch_norm)
        self.msb3       = ConvBlock(in_channels,   n_filters * 4, batch_norm=batch_norm)
        self.encoder3   = ConvBlock(n_filters * 2, n_filters * 4, batch_norm=batch_norm)
        self.pool3      = nn.MaxPool3d(2, stride=2)
        self.drop3      = nn.Dropout3d(dropout)
        # (1, 32, 4, 8, 4)

        self.msb4       = ConvBlock(in_channels,   n_filters * 8, batch_norm=batch_norm)
        self.encoder4   = ConvBlock(n_filters * 4, n_filters * 8, batch_norm=batch_norm)
        self.pool4      = nn.MaxPool3d(2, stride=2)
        self.drop4      = nn.Dropout3d(dropout)
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
    
    def _forward_training(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='trilinear', align_corners=True)
        x3 = F.interpolate(x2, scale_factor=0.5, mode='trilinear', align_corners=True)
        x4 = F.interpolate(x3, scale_factor=0.5, mode='trilinear', align_corners=True)
        
        # full-resolution path
        enc1 = self.encoder1(x1)

        p1 = self.drop1(self.pool1(x1))
        enc2 = self.encoder2(p1)
        
        p2 = self.drop2(self.pool2(enc2))
        enc3 = self.encoder3(p2)
        
        p3 = self.drop3(self.pool3(enc3))
        enc4 = self.encoder4(p3)
        
        p4 = self.drop4(self.pool4(enc4))
        bottleneck = self.center(p4)

        ms4_u4 = self.up4(bottleneck) 
        ms4_u4 = torch.cat([ms4_u4, enc4], dim=1)
        ms4_dec4 = self.drop5(self.decoder4(ms4_u4))

        ms3_u3 = self.up3(ms4_dec4)
        ms3_u3 = torch.cat([ms3_u3, enc3], dim=1)
        ms3_dec3 = self.drop6(self.decoder3(ms3_u3))

        ms2_u2 = self.up2(ms3_dec3)
        ms2_u2 = torch.cat([ms2_u2, enc2], dim=1)
        ms2_dec2 = self.drop7(self.decoder2(ms2_u2))

        u1 = self.up1(ms2_dec2)
        u1 = torch.cat([u1, enc1], dim=1)
        dec1 = self.decoder1(u1)

        final = self.final_conv(dec1)

        # multiscale inputs
        ms2 = self.msb2(x2)
        ms3 = self.msb3(x3)
        ms4 = self.msb4(x4)


        # Multiscale path at ms2
        ms2_ps2 = self.drop2(self.pool2(ms2))
        ms2_enc3 = self.encoder3(ms2_ps2)

        ms2_p3 = self.drop3(self.pool3(ms2_enc3))
        ms2_enc4 = self.encoder4(ms2_p3)

        ms2_p4 = self.drop4(self.pool4(ms2_enc4))
        ms2_bottleneck = self.center(ms2_p4)

        ms4_u4 = self.up4(ms2_bottleneck) 
        ms4_u4 = torch.cat([ms4_u4, enc4], dim=1)
        ms4_dec4 = self.drop5(self.decoder4(ms4_u4))

        ms3_u3 = self.up3(ms4_dec4)
        ms3_u3 = torch.cat([ms3_u3, enc3], dim=1)
        ms3_dec3 = self.drop6(self.decoder3(ms3_u3))

        ms2_u2 = self.up2(ms3_dec3)
        ms2_u2 = torch.cat([ms2_u2, enc2], dim=1)
        ms2_dec2 = self.drop7(self.decoder2(ms2_u2))

        ms2_output = self.ds2(ms2_dec2)

        # Multiscale path at ms3
        ms3_p3 = self.drop3(self.pool3(ms3))
        ms3_enc4 = self.encoder4(ms3_p3)

        ms3_p4 = self.drop4(self.pool4(ms3_enc4))
        ms3_bottleneck = self.center(ms3_p4)

        ms4_u4 = self.up4(ms3_bottleneck) 
        ms4_u4 = torch.cat([ms4_u4, enc4], dim=1)
        ms4_dec4 = self.drop5(self.decoder4(ms4_u4))

        ms3_u3 = self.up3(ms4_dec4)
        ms3_u3 = torch.cat([ms3_u3, enc3], dim=1)
        ms3_dec3 = self.drop6(self.decoder3(ms3_u3))

        ms3_output = self.ds3(ms3_dec3)

        # Multiscale path 3 (starting at ms4)
        ms4_p4 = self.drop4(self.pool4(ms4))
        ms4_bottleneck = self.center(ms4_p4)

        ms4_u4 = self.up4(ms4_bottleneck) 
        ms4_u4 = torch.cat([ms4_u4, enc4], dim=1)
        ms4_dec4 = self.drop5(self.decoder4(ms4_u4))

        ms4_output = self.ds2(ms4_dec4)


        segmentation_outputs = (final, ms2_output, ms3_output, ms4_output)
        consistency_pairs  = (enc2, enc3, enc4), (ms2, ms3, ms4) 
        return segmentation_outputs, consistency_pairs


    def _forward_inference(self, x):
        target_shape = self.init_target_shape
        input_shape = x.shape[2:]

        print(input_shape, " -vs- ", target_shape)
        
        def div_shape(shape, factor):
            return tuple(s // factor for s in shape)

        # shape_to_module = {
            # target_shape: self.encoder1,                       # full-res
            # div_shape(target_shape, 2): self.msb2,             # half-res
            # div_shape(target_shape, 4): self.msb3,             # quarter-res
            # div_shape(target_shape, 8): self.msb4              # eighth-res
        # }
        shape_to_module = {
            target_shape: self.encoder1,                       # full-res
            div_shape(target_shape, 2): 2,             # half-res
            div_shape(target_shape, 4): 3,             # quarter-res
            div_shape(target_shape, 8): 4              # eighth-res
        }
        
        # Round to nearest powers of two for shape
        rounded_shape = tuple(2 ** round(np.log2(s)) for s in input_shape)
        
        if rounded_shape not in shape_to_module:
            raise ValueError(f"Unsupported input shape {rounded_shape}. Expected one of: {list(shape_to_module.keys())}")
        
        segmentation_outputs, consistency_pairs = self._forward_training(x)
        final, ms2, ms3, ms4 = segmentation_outputs
        num_to_ms_output = {
            2: ms2,
            3: ms3,
            4: ms4,
        }
        ms_output = num_to_ms_output[shape_to_module[input_shape]]
        return ms_output

        


        
    def forward(self, x):
       return self._forward_training(x) if self.training else self._forward_inference(x)


    @classmethod
    def from_config(cls, config: DictConfig) -> 'UNet3D':
        return cls(**get_common_args(config))



