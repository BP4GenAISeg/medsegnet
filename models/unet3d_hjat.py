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

@register_model("unet3d_hjat")
class UNet3D(ModelBase):
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
                 inference_fusion_mode: str,
                 deep_supervision_levels: int,
                 depth: int,
    ):
        super(UNet3D, self).__init__()
        self.depth = depth
        self.ds = ds
        self.inference_fusion_mode = inference_fusion_mode
        
        assert 0 < deep_supervision_levels < depth, "deep_supervision_levels must be between 1 and depth-1" 
        self.ds_levels = min(deep_supervision_levels, depth) 
        print("depth of the model: ", depth)
        print("deep supervision levels: ", self.ds_levels)

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

        # Build Auxilary MS Blocks
        self.ms_blocks = nn.ModuleList()
        for d in range(depth - 1):
            ms_out_channels = n_filters * (2**d)
            self.ms_blocks.append(
                ConvBlock(in_channels, ms_out_channels, batch_norm=batch_norm)
            )
        
        """
        Bottleneck layer (center block, bottom of the U-Net).
        """
        self.center = ConvBlock(n_filters * (2 ** (depth - 1)), n_filters * (2 ** depth), batch_norm=batch_norm)
        
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

        """
        Final layer convolution to map to the number of classes.
        """
        self.final_conv = nn.Conv3d(n_filters, num_classes, kernel_size=1)
        
        # Deep supervision branches:
        self.ds_convs = nn.ModuleList()
        for d in range(self.ds_levels, 0, -1):
            out_channels = n_filters * (2 ** d)
            self.ds_convs.append(nn.Conv3d(out_channels, num_classes, kernel_size=1))

        if self.ds:
            init_weights = torch.linspace(1, 0.1, self.ds_levels+1)
            self.ds_weights_params = nn.Parameter(torch.log(init_weights))  # Softmax will amplify deeper layers

    def forward(self, x, phase):

        min_size = 2 ** self.depth
        assert all(dim >= min_size for dim in x.shape[2:]), \
            f"Input spatial dimensions must be at least {min_size}, but got {x.shape[2:]}"
    

        # --- Calculate Downsampled Inputs and Run MS Blocks (Training Only) ---
        ms_features = [] # List to store m2, m3, ...
        if phase == 'train':
            with torch.no_grad(): # No gradients needed for input downsampling - maybe check? FIXME
                x_down = x
                downsampled_inputs = []
                for _ in range(self.depth - 1):
                    x_down = F.interpolate(x_down, scale_factor=0.5, mode='trilinear', align_corners=False, recompute_scale_factor=False)
                    downsampled_inputs.append(x_down)

            # Run MS blocks
            for i, ms_block in enumerate(self.ms_blocks):
                mX = ms_block(downsampled_inputs[i])
                ms_features.append(mX)


        encoder_features_for_skip = []
        intermediate_features_for_loss = []
        out = x
        
        # Encoder pathway
        for d in range(self.depth):
            enc_out = self.encoders[d](out)
            encoder_features_for_skip.append(enc_out) # Store pre-pool features for skip

            pooled_out = self.pools[d](enc_out)
            out = self.enc_dropouts[d](pooled_out)

            # Store post-pool features (pX) for consistency loss, if needed
            if phase == 'train' and d < self.depth - 1: # Store p2, p3, ..., p<depth>
                intermediate_features_for_loss.append(out.detach() if self.training else out)
                # Use .detach() on pX if consistency loss should ONLY train ms_blocks
            # ------------------------------------
        
        # Center
        center_out = self.center(out)
        
        # Decoder pathway
        decoder_outputs_for_ds = [] # Store decoder outputs for DS
        out = center_out
        for i in range(self.depth): # Iterate 'depth' times for decoder
            up_conv = self.up_convs[i]
            decoder = self.decoders[i]
            dropout = self.dec_dropouts[i]

            out = up_conv(out)
            skip = encoder_features_for_skip[-(i + 1)] # Get corresponding skip connection
            out = torch.cat([out, skip], dim=1)
            out = decoder(out)
            out = dropout(out)
            decoder_outputs_for_ds.append(out) # Store decoder feature maps
        
        final = self.final_conv(out)
        
        if self.ds and (self.training or self.inference_fusion_mode != 'only_final'):
            # Compute deep supervision outputs from selected decoder features.
            ds_outputs = []
            for i, (d, ds_conv) in enumerate(zip(range(self.ds_levels, 0, -1), self.ds_convs)):
                ds_out = ds_conv(decoder_features[i])

                up_factor = 2 ** (d)
                ds_out = F.interpolate(ds_out, scale_factor=up_factor, mode='trilinear', align_corners=True)
                ds_outputs.append(ds_out)
            return (final, *ds_outputs)
        else:
            return (final,)

    def get_ds_weights(self):
        assert self.ds, "Deep supervision is not enabled in this model."
        return F.softmax(self.ds_weights_params, dim=0)

    @classmethod
    def from_config(cls, config: DictConfig) -> 'UNet3D':
        base_args = get_common_args(config)
        return cls(
            **base_args,
            deep_supervision_levels=config.model.deep_supervision.levels,
            depth=config.model.depth,
        )

