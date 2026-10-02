import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """
    Two 3x3 convolutions for feature extraction.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(inplace=True),

        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(inplace=True)
    )


class DetailFeatureBlock(nn.Module):
    """
    Residual feature block designed to preserve and refine
    fine spatial details such as edges and textures.
    """

    def __init__(self, channels):
        super(DetailFeatureBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x + residual


class CNNFeatureExtractor(nn.Module):
    """
    Extracts features at 3 scales.

    f1:
        Full resolution.
        Preserves fine edges and spatial details.

    f2:
        Half resolution.
        Captures textures and medium-scale structures.

    f3:
        Quarter resolution.
        Captures larger structures, colour and illumination context.
    """

    def __init__(self, base_channels=48):
        super(CNNFeatureExtractor, self).__init__()

        # ---------------------------------------------------------
        # Stage 1 - FULL RESOLUTION
        # ---------------------------------------------------------

        self.stage1 = conv_block(
            3,
            base_channels
        )

        # Additional full-resolution detail refinement
        self.detail1 = DetailFeatureBlock(
            base_channels
        )

        # ---------------------------------------------------------
        # Stage 2 - HALF RESOLUTION
        # ---------------------------------------------------------

        self.stage2 = conv_block(
            base_channels,
            base_channels * 2
        )

        self.detail2 = DetailFeatureBlock(
            base_channels * 2
        )

        # ---------------------------------------------------------
        # Stage 3 - QUARTER RESOLUTION
        # ---------------------------------------------------------

        self.stage3 = conv_block(
            base_channels * 2,
            base_channels * 4
        )

        self.detail3 = DetailFeatureBlock(
            base_channels * 4
        )

        # Downsampling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        # ---------------------------------------------------------
        # Full-resolution features
        # ---------------------------------------------------------

        f1 = self.stage1(x)

        # Refine fine details
        f1 = self.detail1(f1)

        # ---------------------------------------------------------
        # Half-resolution features
        # ---------------------------------------------------------

        x2 = self.pool(f1)

        f2 = self.stage2(x2)

        # Refine texture information
        f2 = self.detail2(f2)

        # ---------------------------------------------------------
        # Quarter-resolution features
        # ---------------------------------------------------------

        x3 = self.pool(f2)

        f3 = self.stage3(x3)

        # Refine deeper features
        f3 = self.detail3(f3)

        return f1, f2, f3


if __name__ == "__main__":

    model = CNNFeatureExtractor()

    image = torch.randn(
        1,
        3,
        256,
        256
    )

    f1, f2, f3 = model(image)

    print("Input:", image.shape)
    print("f1:", f1.shape)
    print("f2:", f2.shape)
    print("f3:", f3.shape)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )