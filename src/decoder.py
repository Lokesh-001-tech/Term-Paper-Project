import torch
import torch.nn as nn


class ImageDecoder(nn.Module):

    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.decoder = nn.Sequential(

            # 128 attention features + 64 skip features
            nn.Conv2d(
                192,
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
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, attention_features, skip_features):

        # Combine deep features with low-level features
        combined_features = torch.cat(
            [attention_features, skip_features],
            dim=1
        )

        # Reconstruct restored image
        output = self.decoder(combined_features)

        return output


if __name__ == "__main__":

    decoder = ImageDecoder()

    attention_features = torch.randn(
        1,
        128,
        256,
        256
    )

    skip_features = torch.randn(
        1,
        64,
        256,
        256
    )

    output = decoder(
        attention_features,
        skip_features
    )

    print("Attention feature shape:", attention_features.shape)
    print("Skip feature shape:", skip_features.shape)
    print("Output shape:", output.shape)