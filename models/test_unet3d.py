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
            nn.ReLU(inplace=True)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class UNet3DDynamic(nn.Module):
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
                 n_filters: int = 8,
                 dropout: float = 0.5,
                 batch_norm: bool = True,
                 depth: int = 4,
                 deep_supervision_levels: int = 3
    ):
        super(UNet3DDynamic, self).__init__()
        self.depth = depth
        self.deep_supervision_levels = deep_supervision_levels
        
        # Build the encoder pathway dynamically
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        for d in range(depth):
            in_ch = in_channels if d == 0 else n_filters * (2 ** (d - 1))
            out_ch = n_filters * (2 ** d)
            self.encoders.append(ConvBlock(in_ch, out_ch, batch_norm=batch_norm))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.enc_dropouts.append(nn.Dropout3d(dropout))
        
        # Center block (the bottom of the U)
        self.center = ConvBlock(n_filters * (2 ** (depth - 1)), n_filters * (2 ** depth), batch_norm=batch_norm)
        
        # Build the decoder pathway dynamically
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dec_dropouts = nn.ModuleList()
        for d in range(depth - 1, -1, -1):
            in_ch = n_filters * (2 ** (d + 1))
            out_ch = n_filters * (2 ** d)
            self.up_convs.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            # The decoder block takes the concatenation of the upsampled output and the encoder output.
            self.decoders.append(
                ConvBlock(out_ch + n_filters * (2 ** d), out_ch, batch_norm=batch_norm)
            )
            self.dec_dropouts.append(nn.Dropout3d(dropout))
        
        # Final 1x1 convolution to map to the number of classes.
        self.final_conv = nn.Conv3d(n_filters, num_classes, kernel_size=1)
        
        # Deep supervision branches:
        # We attach deep supervision branches to the first few decoder blocks.
        # For example, if depth==4 and deep_supervision_levels==3, then we attach supervision
        # to the first three decoder outputs (which will be upsampled appropriately).
        self.ds_convs = nn.ModuleList()
        ds_levels = min(deep_supervision_levels, depth)  # ensure we do not exceed available decoder blocks
        for i in range(ds_levels):
            # The output channel of the corresponding decoder block is n_filters * (2 ** (depth - i - 1))
            out_channels = n_filters * (2 ** (depth - i - 1))
            self.ds_convs.append(nn.Conv3d(out_channels, num_classes, kernel_size=1))
    
    def forward(self, x):
        # (Optionally add a check that the input size is large enough for the desired depth)
        encoder_features = []
        out = x
        
        # Encoder pathway
        for enc, pool, dropout in zip(self.encoders, self.pools, self.enc_dropouts):
            out = enc(out)
            encoder_features.append(out)
            out = dropout(pool(out))
        
        # Center
        center_out = self.center(out)
        
        # Decoder pathway
        decoder_features = []
        out = center_out
        for i, (up_conv, decoder, dropout) in enumerate(zip(self.up_convs, self.decoders, self.dec_dropouts)):
            out = up_conv(out)
            # Use the corresponding encoder feature for the skip connection (in reverse order)
            skip = encoder_features[-(i+1)]
            out = torch.cat([out, skip], dim=1)
            out = decoder(out)
            out = dropout(out)
            decoder_features.append(out)
        
        final = self.final_conv(out)
        
        if self.training:
            # Compute deep supervision outputs from selected decoder features.
            # In our design, decoder_features[0] is the deepest decoder output,
            # and we attach supervision branches to the first few blocks.
            ds_outputs = []
            for i, ds_conv in enumerate(self.ds_convs):
                ds_out = ds_conv(decoder_features[i])
                # Calculate upsampling factor. For example, if depth==4:
                #  - For i==0 (deepest decoder output): factor = 2^(4-1)=8
                #  - For i==1: factor = 2^(4-2)=4
                #  - For i==2: factor = 2^(4-3)=2
                up_factor = 2 ** (self.depth - (i + 1))
                ds_out = F.interpolate(ds_out, scale_factor=up_factor, mode='trilinear', align_corners=True)
                ds_outputs.append(ds_out)
            return final, *ds_outputs
        else:
            return final
