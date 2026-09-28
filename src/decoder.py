import torch
import torch.nn as nn


class ImageDecoder(nn.Module):

    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.decoder = nn.Sequential(

            nn.Conv2d(
                128,
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

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":

    decoder = ImageDecoder()

    # Simulated attention output
    features = torch.randn(1, 128, 256, 256)

    output = decoder(features)

    print("Input feature shape:", features.shape)
    print("Reconstructed image shape:", output.shape)