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

        # 1. Extract features
        features = self.feature_extractor(x)

        # 2. Apply attention
        refined_features = self.attention(features)

        # 3. Reconstruct enhanced image
        output = self.decoder(refined_features)

        return output


if __name__ == "__main__":

    model = UnderwaterRestorationModel()

    # Test image
    image = torch.randn(1, 3, 256, 256)

    output = model(image)

    print("Input shape:", image.shape)
    print("Output shape:", output.shape) 