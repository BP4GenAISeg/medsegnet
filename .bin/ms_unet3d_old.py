import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models import register_model
from models.base import ModelBase
from models.config_utils import get_common_args


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

@register_model("simon-msunet3d")
class TestUNet3D(ModelBase):
    #FIXME delete MS
    def __init__(self, in_channels, num_classes, n_filters, dropout, batch_norm, ds, ms, inference_fusion_mode):
        super().__init__()
        self.ds = ds
        self.inference_fusion_mode = inference_fusion_mode

        # assert all(spatial) > 2, "all spatial dimensions must be > 2"
        
        # ======= Encoder =======
        # (2, 1, 32, 64, 32)
        self.encoder1 = ConvBlock(in_channels, n_filters, batch_norm=batch_norm)
        # (1, 8, 32, 64, 32)
        self.pool1    = nn.MaxPool3d(2, stride=2)
        # (1, 8, 16, 32, 16)
        self.drop1    = nn.Dropout3d(dropout)
        # (1, 8, 16, 32, 16)

        self.encoder2 = ConvBlock(n_filters, n_filters * 2, batch_norm=batch_norm)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.drop2 = nn.Dropout3d(dropout)
        # (1, 16, 8, 16, 8)

        self.encoder3 = ConvBlock(n_filters * 2, n_filters * 4, batch_norm=batch_norm)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.drop3 = nn.Dropout3d(dropout)
        # (1, 32, 4, 8, 4)

        self.encoder4 = ConvBlock(n_filters * 4, n_filters * 8, batch_norm=batch_norm)
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.drop4 = nn.Dropout3d(dropout)
        # (1, 64, 2, 4, 2)

        # ======= Bottleneck =======
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16, batch_norm=batch_norm)

        # ======= Decoder =======
        self.up4 = nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=3, stride=2)
        self.decoder4 = ConvBlock(n_filters * 16, n_filters * 8, batch_norm=batch_norm)
        self.drop5 = nn.Dropout3d(dropout)
        # (1, 64, 4, 8, 4)

        self.up3 = nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=3, stride=2)
        self.decoder3 = ConvBlock(n_filters * 8, n_filters * 4, batch_norm=batch_norm)
        self.drop6 = nn.Dropout3d(dropout)
        # (1, 32, 8, 16, 8)

        self.up2 = nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=3, stride=2)
        self.decoder2 = ConvBlock(n_filters * 4, n_filters * 2, batch_norm=batch_norm)
        self.drop7 = nn.Dropout3d(dropout)
        # (1, 16, 16, 32, 16)

        self.up1 = nn.ConvTranspose3d(n_filters * 2, n_filters, kernel_size=3, stride=2)
        self.decoder1 = ConvBlock(n_filters + n_filters, n_filters, batch_norm=batch_norm)
        # (1, 8, 32, 64, 32)

        self.final_conv = nn.Conv3d(n_filters, num_classes, kernel_size=1)

        # ======= Multiscale Input Blocks (MSBs) =======
        self.msb2 = ConvBlock(in_channels, n_filters * 2, batch_norm=batch_norm)
        self.msb3 = ConvBlock(in_channels, n_filters * 4, batch_norm=batch_norm)
        self.msb4 = ConvBlock(in_channels, n_filters * 8, batch_norm=batch_norm)

        # ======= Deep Supervision Heads =======
        self.ds2 = nn.Conv3d(n_filters * 2, num_classes, kernel_size=1)
        self.ds3 = nn.Conv3d(n_filters * 4, num_classes, kernel_size=1)
        self.ds4 = nn.Conv3d(n_filters * 8, num_classes, kernel_size=1)

    def _forward_training(self, x):
        x2 = F.interpolate(x, size=(16, 32, 16), mode="trilinear", align_corners=True)
        x3 = F.interpolate(x, size=(8, 16, 8), mode="trilinear", align_corners=True)
        x4 = F.interpolate(x, size=(4, 8, 4), mode="trilinear", align_corners=True)

        # print("All shapes")
        # print("x.shape:", x.shape)
        # print("x2.shape:", x2.shape)
        # print("x3.shape:", x3.shape)
        # print("x4.shape:", x4.shape)

        # Full resolution 
        enc1 = self.encoder1(x)
        pool1 = self.drop1(self.pool1(enc1))

        enc2 = self.encoder2(pool1)
        pool2 = self.drop2(self.pool2(enc2))
        
        enc3 = self.encoder3(pool2)
        pool3 = self.drop3(self.pool3(enc3))
        
        enc4 = self.encoder4(pool3)
        pool4 = self.drop4(self.pool4(enc4))

        bot = self.bottleneck(pool4)
        
        up4 = self.up4(bot)
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

        final = self.final_conv(dec1)

        # ===== MSB2 =====
        enc2_ms = self.msb2(x2)
        pool2_ms = self.drop2(self.pool2(enc2_ms))

        enc3_ms = self.encoder3(pool2_ms)
        pool3_ms = self.drop3(self.pool3(enc3_ms))

        enc4_ms = self.encoder4(pool3_ms)
        pool4_ms = self.drop4(self.pool4(enc4_ms))
        
        bot2 = self.bottleneck(pool4_ms)

        up4_ms = self.up4(bot2)
        up4_ms = torch.cat([up4_ms, enc4_ms], dim=1)
        dec4_ms = self.drop5(self.decoder4(up4_ms))

        up3_ms = self.up3(dec4_ms)
        up3_ms = torch.cat([up3_ms, enc3_ms], dim=1)
        dec3_ms = self.drop6(self.decoder3(up3_ms))

        up2_ms = self.up2(dec3_ms)
        up2_ms = torch.cat([up2_ms, enc2_ms], dim=1)
        dec2_ms = self.drop7(self.decoder2(up2_ms))
        out_ms2 = self.ds2(dec2_ms)

        # ===== MSB3 =====
        enc3_ms2 = self.msb3(x3)
        pool3_ms2 = self.drop3(self.pool3(enc3_ms2))
        enc4_ms2 = self.encoder4(pool3_ms2)
        pool4_ms2 = self.drop4(self.pool4(enc4_ms2))
        bot3 = self.bottleneck(pool4_ms2)

        up4_ms3 = self.up4(bot3)
        up4_ms3 = torch.cat([up4_ms3, enc4_ms2], dim=1)
        dec4_ms3 = self.drop5(self.decoder4(up4_ms3))

        up3_ms3 = self.up3(dec4_ms3)
        up3_ms3 = torch.cat([up3_ms3, enc3_ms2], dim=1)
        dec3_ms3 = self.drop6(self.decoder3(up3_ms3))
        out_ms3 = self.ds3(dec3_ms3)
        
        # ===== MSB4 =====
        enc4_ms3 = self.msb4(x4)
        pool4_ms3 = self.drop4(self.pool4(enc4_ms3))
        bot4 = self.bottleneck(pool4_ms3)

        up4_ms4 = self.up4(bot4)
        up4_ms4 = torch.cat([up4_ms4, enc4_ms3], dim=1)
        dec4_ms4 = self.drop5(self.decoder4(up4_ms4))
        out_ms4 = self.ds4(dec4_ms4)

        segmentations = (final, out_ms2, out_ms3, out_ms4)
        consistency = (enc2, enc3, enc4), (enc2_ms, enc3_ms2, enc4_ms3)
        return segmentations, consistency

    def _forward_inference(self, x):
        _, _, D, H, W = x.shape
        input_shape = (D, H, W)
        print("[INFERENCE]: The input shape is", input_shape) 
        #FIXME should be target shape 
        
        base_shape = (32, 64, 32)

        def div_shape(shape, factor):
            return tuple(s // factor for s in shape)

        shape_to_entry = {
            base_shape: 'enc1',
            div_shape(base_shape, 2): 'msb2',
            div_shape(base_shape, 4): 'msb3',
            div_shape(base_shape, 8): 'msb4'
        }

        rounded_shape = tuple(2 ** round(np.log2(s)) for s in input_shape)
        if rounded_shape not in shape_to_entry:
            raise ValueError(f"Unsupported input shape {input_shape} (rounded: {rounded_shape}). "
                             f"Expected one of: {list(shape_to_entry.keys())}")

        entry = shape_to_entry[rounded_shape]
        
        if entry == 'enc1':
            enc1 = self.encoder1(x)
            pool1 = self.drop1(self.pool1(enc1))
            enc2 = self.encoder2(pool1)
            pool2 = self.drop2(self.pool2(enc2))
            enc3 = self.encoder3(pool2)
            pool3 = self.drop3(self.pool3(enc3))
            enc4 = self.encoder4(pool3)
            pool4 = self.drop4(self.pool4(enc4))
            bot = self.bottleneck(pool4)

            up4 = self.up4(bot)
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
            return self.final_conv(dec1)

        elif entry == 'msb2':
            enc2 = self.msb2(x)
            pool2 = self.drop2(self.pool2(enc2))
            enc3 = self.encoder3(pool2)
            pool3 = self.drop3(self.pool3(enc3))
            enc4 = self.encoder4(pool3)
            pool4 = self.drop4(self.pool4(enc4))
            bot = self.bottleneck(pool4)

            up4 = self.up4(bot)
            up4 = torch.cat([up4, enc4], dim=1)
            dec4 = self.drop5(self.decoder4(up4))
            up3 = self.up3(dec4)
            up3 = torch.cat([up3, enc3], dim=1)
            dec3 = self.drop6(self.decoder3(up3))
            up2 = self.up2(dec3)
            up2 = torch.cat([up2, enc2], dim=1)
            dec2 = self.drop7(self.decoder2(up2))
            return self.ds2(dec2)

        elif entry == 'msb3':
            enc3 = self.msb3(x)
            pool3 = self.drop3(self.pool3(enc3))
            enc4 = self.encoder4(pool3)
            pool4 = self.drop4(self.pool4(enc4))
            bot = self.bottleneck(pool4)

            up4 = self.up4(bot)
            up4 = torch.cat([up4, enc4], dim=1)
            dec4 = self.drop5(self.decoder4(up4))
            up3 = self.up3(dec4)
            up3 = torch.cat([up3, enc3], dim=1)
            dec3 = self.drop6(self.decoder3(up3))
            return self.ds3(dec3)

        elif entry == 'msb4':
            enc4 = self.msb4(x)
            pool4 = self.drop4(self.pool4(enc4))
            bot = self.bottleneck(pool4)
            up4 = self.up4(bot)
            up4 = torch.cat([up4, enc4], dim=1)
            dec4 = self.drop5(self.decoder4(up4))
            return self.ds4(dec4)

    def forward(self, x):
        return self._forward_training(x) if self.training else self._forward_inference(x)

    @classmethod
    def from_config(cls, config: DictConfig) -> 'TestUNet3D':
        return cls(**get_common_args(config))
