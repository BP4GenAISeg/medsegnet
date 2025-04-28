from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.base import ModelBase
from utils.config_utils import get_common_args
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
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


@register_model("ds-unet3d")
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

    def __init__(
        self,
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
        super(UNet3D, self).__init__()
        self.depth = depth
        self.ds = ds
        self.inference_fusion_mode = inference_fusion_mode

        assert (
            0 < deep_supervision_levels < depth
        ), "deep_supervision_levels must be between 1 and depth-1"
        self.ds_levels = min(deep_supervision_levels, depth)
        print("depth of the model: ", depth)
        print("deep supervision levels: ", self.ds_levels)

        # Build the encoder pathway dynamically
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        for d in range(depth):
            in_ch = in_channels if d == 0 else n_filters * (2 ** (d - 1))
            out_ch = n_filters * (2**d)
            self.encoders.append(ConvBlock(in_ch, out_ch, batch_norm=batch_norm))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.enc_dropouts.append(nn.Dropout3d(dropout))

        """
        Bottleneck layer (center block, bottom of the U-Net).
        """
        self.center = ConvBlock(
            n_filters * (2 ** (depth - 1)),
            n_filters * (2**depth),
            batch_norm=batch_norm,
        )

        # Build the decoder pathway dynamically
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dec_dropouts = nn.ModuleList()

        # 0..3, 3..0
        for d in range(depth - 1, -1, -1):
            in_ch = n_filters * (2 ** (d + 1))
            out_ch = n_filters * (2**d)
            self.up_convs.append(
                nn.ConvTranspose3d(
                    in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            self.decoders.append(ConvBlock(in_ch, out_ch, batch_norm=batch_norm))
            self.dec_dropouts.append(nn.Dropout3d(dropout))

        """
        Final layer convolution to map to the number of classes.
        """
        self.final_conv = nn.Conv3d(n_filters, num_classes, kernel_size=1)

        # Deep supervision branches:
        # We attach deep supervision branches to the first few decoder blocks.
        # For example, if depth==4 and deep_supervision_levels==3, then we attach supervision
        # to the first three decoder outputs (which will be upsampled appropriately).

        self.ds_convs = nn.ModuleList()
        for d in range(self.ds_levels, 0, -1):
            out_channels = n_filters * (2**d)
            self.ds_convs.append(nn.Conv3d(out_channels, num_classes, kernel_size=1))

        if self.ds:
            # ds_levels - num of ds outputs, and +1 for the final output
            # init_probs = compute_weights_depth(self.ds_levels + 1)
            # init_logits = torch.log(torch.tensor(init_probs, dtype=torch.float))
            # self.ds_weights_params = nn.Parameter(init_logits)

            # self.ds_weights_params = nn.Parameter(torch.zeros(self.ds_levels + 1))
            init_weights = torch.linspace(1, 0.1, self.ds_levels + 1)
            self.ds_weights_params = nn.Parameter(
                torch.log(init_weights)
            )  # Softmax will amplify deeper layers

    def forward(self, x, phase):

        # x1 = x       # full resolution
        # x2 = downsample(input, factor=2)

        min_size = 2**self.depth
        assert all(
            dim >= min_size for dim in x.shape[2:]
        ), f"Input spatial dimensions must be at least {min_size}, but got {x.shape[2:]}"

        encoder_features = []
        out = x

        # Encoder pathway
        for enc, pool, drop in zip(self.encoders, self.pools, self.enc_dropouts):
            out = enc(out)
            encoder_features.append(out)
            out = drop(pool(out))

        # Center
        center_out = self.center(out)

        # Decoder pathway
        decoder_features = []
        out = center_out
        for i, (up_conv, dec, drop) in enumerate(
            zip(self.up_convs, self.decoders, self.dec_dropouts)
        ):
            out = up_conv(out)
            # Use the corresponding encoder feature for the skip connection (in reverse order)
            # numbers 0, 1, 2 represent indexes.
            # enc0,                              dec3
            #   enc1,                       dec2
            #       enc2,               dec1
            #           enc3,       dec0
            #               center,
            skip = encoder_features[-(i + 1)]
            out = torch.cat([out, skip], dim=1)
            out = dec(out)
            out = drop(out)
            decoder_features.append(out)

        final = self.final_conv(out)

        if self.ds and (self.training or self.inference_fusion_mode != "only_final"):
            # Compute deep supervision outputs from selected decoder features.
            # In our design, decoder_features[0] is the deepest decoder output,
            # and we attach supervision branches to the first few blocks.
            ds_outputs = []
            for i, (d, ds_conv) in enumerate(
                zip(range(self.ds_levels, 0, -1), self.ds_convs)
            ):
                ds_out = ds_conv(decoder_features[i])

                # Calculate upsampling factor. For example, if depth==4:
                #  - For i==0 (deepest decoder output): factor = 2^(4-1)=8
                #  - For i==1: factor = 2^(4-2)=4
                #  - For i==2: factor = 2^(4-3)=2
                up_factor = 2 ** (d)
                ds_out = F.interpolate(
                    ds_out, scale_factor=up_factor, mode="trilinear", align_corners=True
                )
                ds_outputs.append(ds_out)
            return (final, *ds_outputs)
        else:
            return (final,)

    def get_ds_weights(self):
        assert self.ds, "Deep supervision is not enabled in this model."
        return F.softmax(self.ds_weights_params, dim=0)

    @classmethod
    def from_config(cls, config: DictConfig) -> "UNet3D":
        base_args = get_common_args(config)
        return cls(
            **base_args,
            deep_supervision_levels=config.model.deep_supervision.levels,
            depth=config.model.depth,
        )
