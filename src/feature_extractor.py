import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):

    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        # Low-level feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # Deep feature extraction
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # Extract low-level features
        low_level_features = self.conv1(x)

        # Extract deeper features
        deep_features = self.conv2(low_level_features)

        return low_level_features, deep_features


if __name__ == "__main__":

    model = CNNFeatureExtractor()

    image = torch.randn(1, 3, 256, 256)

    low_level, deep_features = model(image)

    print("Input shape:", image.shape)
    print("Low-level feature shape:", low_level.shape)
    print("Deep feature shape:", deep_features.shape)