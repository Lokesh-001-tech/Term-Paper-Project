
import torch
import torch.nn as nn

from feature_extractor import conv_block


class DetailRefinementBlock(nn.Module):
    """
    Refines high-frequency details such as edges and textures.

    The residual connection helps preserve useful information while
    allowing the network to learn additional detail restoration.
    """

    def __init__(self, channels):
        super(DetailRefinementBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.body(x)


class ImageDecoder(nn.Module):
    """
    Reconstructs the restored image from multi-scale features.

    Skip connections from f2 and f1 preserve spatial information,
    while DetailRefinementBlock modules help reconstruct edges,
    textures and fine details.

    The decoder outputs a residual correction rather than the
    final image.
    """

    def __init__(self, base_channels=48):
        super(ImageDecoder, self).__init__()

        # ---------------------------------------------------------
        # Deep feature: 1/4 resolution -> 1/2 resolution
        # ---------------------------------------------------------

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )

        # Concatenation:
        # up2 output = base_channels * 2
        # f2         = base_channels * 2
        # total      = base_channels * 4

        self.dec2 = conv_block(
            base_channels * 4,
            base_channels * 2
        )

        # Detail refinement at 1/2 resolution
        self.detail2 = DetailRefinementBlock(
            base_channels * 2
        )

        # ---------------------------------------------------------
        # 1/2 resolution -> full resolution
        # ---------------------------------------------------------

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )

        # Concatenation:
        # up1 output = base_channels
        # f1         = base_channels
        # total      = base_channels * 2

        self.dec1 = conv_block(
            base_channels * 2,
            base_channels
        )

        # Detail refinement at full resolution
        self.detail1 = DetailRefinementBlock(
            base_channels
        )

        # ---------------------------------------------------------
        # Final residual correction
        # ---------------------------------------------------------

        self.out = nn.Conv2d(
            base_channels,
            3,
            kernel_size=3,
            padding=1
        )

    def forward(self, deep_features, f2, f1):

        # ---------------------------------------------------------
        # Stage 1: reconstruct medium-scale details
        # ---------------------------------------------------------

        x = self.up2(deep_features)

        # Skip connection from encoder
        x = torch.cat([x, f2], dim=1)

        x = self.dec2(x)

        # Refine textures and edges
        x = self.detail2(x)

        # ---------------------------------------------------------
        # Stage 2: reconstruct fine details
        # ---------------------------------------------------------

        x = self.up1(x)

        # Skip connection from encoder
        x = torch.cat([x, f1], dim=1)

        x = self.dec1(x)

        # Refine fine edges and textures
        x = self.detail1(x)

        # ---------------------------------------------------------
        # Predict residual correction
        # ---------------------------------------------------------

        correction = self.out(x)

        return correction


if __name__ == "__main__":

    model = ImageDecoder()

    deep_features = torch.randn(1, 48 * 4, 64, 64)
    f2 = torch.randn(1, 48 * 2, 128, 128)
    f1 = torch.randn(1, 48, 256, 256)

    output = model(
        deep_features,
        f2,
        f1
    )

    print("Deep features:", deep_features.shape)
    print("f2:", f2.shape)
    print("f1:", f1.shape)
    print("Output:", output.shape)
