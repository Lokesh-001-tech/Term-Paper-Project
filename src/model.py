import torch
import torch.nn as nn

from feature_extractor import CNNFeatureExtractor
from attention import ChannelSpatialAttention
from decoder import ImageDecoder


class UnderwaterRestorationModel(nn.Module):

    def __init__(self):
        super(UnderwaterRestorationModel, self).__init__()

        # CNN feature extraction
        self.feature_extractor = CNNFeatureExtractor()

        # Channel + Spatial Attention
        self.attention = ChannelSpatialAttention(128)

        # Image reconstruction
        self.decoder = ImageDecoder()

    def forward(self, x):

        # ---------------------------------------
        # 1. Extract low-level and deep features
        # ---------------------------------------
        low_level_features, deep_features = self.feature_extractor(x)

        # ---------------------------------------
        # 2. Apply channel + spatial attention
        # ---------------------------------------
        refined_features = self.attention(deep_features)

        # ---------------------------------------
        # 3. Reconstruct image using skip connection
        # ---------------------------------------
        output = self.decoder(
            refined_features,
            low_level_features
        )

        return output


if __name__ == "__main__":

    model = UnderwaterRestorationModel()

    # Test image
    image = torch.randn(
        1,
        3,
        256,
        256
    )

    output = model(image)

    print("Input shape:", image.shape)
    print("Output shape:", output.shape)