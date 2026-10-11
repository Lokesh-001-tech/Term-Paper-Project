import torch
import torch.nn as nn
from feature_extractor import conv_block
class ImageDecoder(nn.Module):
    """
    Multi-scale decoder.
    Uses skip connections from the encoder and predicts
    a residual correction to the original image.
    """
    def __init__(self, base_channels=48):
        super(ImageDecoder, self).__init__()
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )
        self.dec2 = conv_block(
            base_channels * 4,
            base_channels * 2
        )
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )
        self.dec1 = conv_block(
            base_channels * 2,
            base_channels
        )
        self.out = nn.Conv2d(
            base_channels,
            3,
            kernel_size=3,
            padding=1
        )
    def forward(self, deep_features, f2, f1):
        x = self.up2(deep_features)
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)
        correction = self.out(x)
        return correction
if __name__ == "__main__":
    model = ImageDecoder()
    deep_features = torch.randn(
        1,
        48 * 4,
        64,
        64
    )
    f2 = torch.randn(
        1,
        48 * 2,
        128,
        128
    )
    f1 = torch.randn(
        1,
        48,
        256,
        256
    )
    output = model(
        deep_features,
        f2,
        f1
    )
    print("Deep features:", deep_features.shape)
    print("f2:", f2.shape)
    print("f1:", f1.shape)
    print("Output:", output.shape)
