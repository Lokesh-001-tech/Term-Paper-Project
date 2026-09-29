import torch
import torch.nn as nn

from feature_extractor import conv_block


class ImageDecoder(nn.Module):
    """
    Reconstructs the image from the attention-refined deep features,
    using skip connections from the feature extractor to recover detail.

    It outputs a RESIDUAL (a correction), not the final image. The model
    adds this correction to the input, so it starts as "do nothing" and
    only has to learn what to change.
    """

    def __init__(self, base_channels=48):
        super(ImageDecoder, self).__init__()

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = conv_block(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, deep_features, f2, f1):

        x = self.up2(deep_features)
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)

        return self.out(x)