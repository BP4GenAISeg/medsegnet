from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.base import ModelBase
from models.config_utils import get_common_args
from utils.inference import compute_weights_depth


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

@register_model("simonmsdyn-unet3d")
class MultiScaleDynamicUNet3D(ModelBase):
    """
    A dynamic 3D U-Net that can adapt its depth based on the input parameter.
    It supports deep supervision outputs from multiple decoder levels.
    
    Additional SOTA considerations:
      - Ensure that your input volumes are large enough for the number of downsamplings.
      - Experiment with residual connections or attention mechanisms.
      - Consider using instance normalization instead of batch normalization in some cases.
      - Use appropriate loss functions (e.g. Dice loss, focal loss) and data augmentation.
    """
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 n_filters: int,
                 dropout: float,
                 batch_norm: bool,
                 ds: bool,
                 ms: bool,
                 inference_fusion_mode: str,
                 deep_supervision_levels: int,
                 depth: int,
    ):
        super(MultiScaleDynamicUNet3D, self).__init__()
        self.depth = depth
        
        # assert 0 < deep_supervision_levels < depth, "deep_supervision_levels must be between 1 and depth-1" 
        # self.ds_levels = min(deep_supervision_levels, depth) 
        # print("depth of the model: ", depth)
        # print("deep supervision levels: ", self.ds_levels)

        # Build the encoder pathway dynamically
        self.encoders: nn.ModuleList = nn.ModuleList()
        self.pools: nn.ModuleList = nn.ModuleList()
        self.enc_dropouts: nn.ModuleList = nn.ModuleList()
        for d in range(depth):
            in_ch = in_channels if d == 0 else n_filters * (2 ** (d - 1))
            out_ch = n_filters * (2 ** d)
            self.encoders.append(ConvBlock(in_ch, out_ch, batch_norm=batch_norm))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.enc_dropouts.append(nn.Dropout3d(dropout))
            
        
        # Bottleneck layer (center block, bottom of the U-Net).
        bn_in_channels  = n_filters * (2 ** (depth - 1))
        bn_out_channels = n_filters * (2 ** depth)
        self.bn         = ConvBlock(bn_in_channels, bn_out_channels, batch_norm=batch_norm)
        
        # Build the decoder pathway dynamically
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dec_dropouts = nn.ModuleList()

        # 0..3, 3..0
        for d in range(depth - 1, -1, -1): 
            in_ch = n_filters * (2 ** (d + 1))
            out_ch = n_filters * (2 ** d)
            self.up_convs.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            self.decoders.append(
                ConvBlock(in_ch, out_ch, batch_norm=batch_norm)
            )
            self.dec_dropouts.append(nn.Dropout3d(dropout))

        # Final layer convolution to map to the number of classes.
        self.final_conv = nn.Conv3d(n_filters, num_classes, kernel_size=1)
        
        
        # ===== Multiscale Input Blocks =====
        # one MSB per downsampling level (excluding full resolution)
        self.msb_blocks = nn.ModuleList()
        self.ms_heads = nn.ModuleList()
        for k in range(1, depth):
            self.msb_blocks.append(
                ConvBlock(in_channels, n_filters * (2 ** k), batch_norm=batch_norm)
            )
            # 1x1 head to produce segmentation at that scale,
            # otherwise, we'd just have n_filters * (2 ** k) channels/feature maps for output
            # when we want num_classes
            self.ms_heads.append(
                nn.Conv3d(n_filters * (2 ** k), num_classes, kernel_size=1)
            )
               
        # # Deep supervision branches:
        # # We attach deep supervision branches to the first few decoder blocks.
        # # For example, if depth==4 and deep_supervision_levels==3, then we attach supervision
        # # to the first three decoder outputs (which will be upsampled appropriately).
        
        # self.ds_convs = nn.ModuleList()
        # for d in range(self.ds_levels, 0, -1):
        #     out_channels = n_filters * (2 ** d)
        #     self.ds_convs.append(nn.Conv3d(out_channels, num_classes, kernel_size=1))

        # if self.ds:
        #     # ds_levels - num of ds outputs, and +1 for the final output
        #     # init_probs = compute_weights_depth(self.ds_levels + 1)  
        #     # init_logits = torch.log(torch.tensor(init_probs, dtype=torch.float))
        #     # self.ds_weights_params = nn.Parameter(init_logits)
            
        #     # self.ds_weights_params = nn.Parameter(torch.zeros(self.ds_levels + 1))
        #     init_weights = torch.linspace(1, 0.1, self.ds_levels+1)
        #     self.ds_weights_params = nn.Parameter(torch.log(init_weights))  # Softmax will amplify deeper layers

    def forward(self, x):
        min_size = 2 ** self.depth
        assert all(dim >= min_size for dim in x.shape[2:]), \
            f"Input spatial dimensions must be at least {min_size}, but got {x.shape[2:]}"
    
        encoder_feats = []
        out = x
        
        # Encoder pathway
        for enc, pool, dropout in zip(self.encoders, self.pools, self.enc_dropouts):
            out = enc(out)
            encoder_feats.append(out)
            out = dropout(pool(out))
        
        # Center
        center_out = self.bn(out)
        
        # Decoder pathway
        decoder_feats = []
        out = center_out
        
        for up_conv, decoder, dropout in zip(self.up_convs, self.decoders, self.dec_dropouts):
            # Use the corresponding encoder feature for the skip connection (in reverse order)
            #numbers 0, 1, 2 represent indexes.
            #enc0,                              dec3
            #   enc1,                       dec2
            #       enc2,               dec1
            #           enc3,       dec0
            #               center,
            # FIXME: fix in the normal unet 3d use pop instead
            # skip = encoder_feats[-(i+1)]
            out  = up_conv(out)
            skip = encoder_feats.pop() # pop the last feature appended
            out  = torch.cat([out, skip], dim=1)
            out  = decoder(out)
            out  = dropout(out)
            decoder_feats.append(out)

        final_out = self.final_conv(out)
     
        # ===== Multiscale inputs (during training) =====
        ms_outputs = []
        D, H, W = x.shape[2:]
        for d in range(1, self.depth):
            # Downsampling
            target_size = (D // (2 ** d), H // (2 ** d), W // (2 ** d))
            x_ms = F.interpolate(x, size=target_size, mode='trilinear', align_corners=True)
            
            # Build encoder features for MS path
            ms_feats = []
            msb      = self.msb_blocks[d - 1]
            out_ms   = msb(x_ms)
            ms_feats.append(out_ms)
            out_ms = self.pools[d](out_ms)
            out_ms = self.enc_dropouts[d](out_ms)
            
            for enc, pool, dropout in zip(
                list(self.encoders)[d+1:],
                list(self.pools)[d+1:],
                list(self.enc_dropouts)[d+1:]
            ):
                out_ms = enc(out_ms)
                ms_feats.append(out_ms)
                out_ms = dropout(pool(out_ms))
        
            # bottleneck
            out_ms = self.bn(out_ms)

            num_ups = self.depth - d
            # decoder up to match MS scale
            for (up_conv, dec, drop) in zip(
                list(self.up_convs)[:num_ups],
                list(self.decoders)[:num_ups],
                list(self.dec_dropouts)[:num_ups]
            ):
                out_ms = up_conv(out_ms)
                skip   = ms_feats.pop() # pop the last feature appended
                out_ms = torch.cat([out_ms, skip], dim=1)
                out_ms = dec(out_ms)
                out_ms = drop(out_ms)
            
            ms_seg = self.ms_heads[d - 1](out_ms)
            ms_outputs.append(ms_seg)
        print(ms_outputs)
        return (final_out, *ms_outputs)
        # if self.ds and (self.training or self.inference_fusion_mode != 'only_final'):
        #     # Compute deep supervision outputs from selected decoder features.
        #     # In our design, decoder_features[0] is the deepest decoder output,
        #     # and we attach supervision branches to the first few blocks.
        #     ds_outputs = []
        #     for i, (d, ds_conv) in enumerate(zip(range(self.ds_levels, 0, -1), self.ds_convs)):
        #         ds_out = ds_conv(decoder_feats[i])

        #         # Calculate upsampling factor. For example, if depth==4:
        #         #  - For i==0 (deepest decoder output): factor = 2^(4-1)=8
        #         #  - For i==1: factor = 2^(4-2)=4
        #         #  - For i==2: factor = 2^(4-3)=2
        #         up_factor = 2 ** (d)
        #         ds_out = F.interpolate(ds_out, scale_factor=up_factor, mode='trilinear', align_corners=True)
        #         ds_outputs.append(ds_out)
        #     return (final, *ds_outputs)
        # else:
        #     return (final,)

    def get_ds_weights(self):
        assert self.ds, "Deep supervision is not enabled in this model."
        return F.softmax(self.ds_weights_params, dim=0)

    @classmethod
    def from_config(cls, config: DictConfig) -> 'MultiScaleDynamicUNet3D':
        base_args = get_common_args(config)
        return cls(
            **base_args,
            deep_supervision_levels=config.model.deep_supervision.levels,
            depth=config.model.depth,
        )

