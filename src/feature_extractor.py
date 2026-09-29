import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Two 3x3 convolutions with ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class CNNFeatureExtractor(nn.Module):
    """
    Extracts features at 3 scales:
        f1: full resolution   (fine detail, edges)
        f2: 1/2 resolution    (textures)
        f3: 1/4 resolution    (global colour / lighting context)
    """

    def __init__(self, base_channels=48):
        super(CNNFeatureExtractor, self).__init__()

        self.stage1 = conv_block(3, base_channels)
        self.stage2 = conv_block(base_channels, base_channels * 2)
        self.stage3 = conv_block(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(self.pool(f1))
        f3 = self.stage3(self.pool(f2))

        return f1, f2, f3


if __name__ == "__main__":

    model = CNNFeatureExtractor()
    image = torch.randn(1, 3, 256, 256)

    f1, f2, f3 = model(image)

    print("f1:", f1.shape)
    print("f2:", f2.shape)
    print("f3:", f3.shape)