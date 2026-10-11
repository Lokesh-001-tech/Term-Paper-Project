import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import CNNFeatureExtractor
from attention import ResidualAttentionBlock
from decoder import ImageDecoder
class UnderwaterRestorationModel(nn.Module):
    def __init__(self, base_channels=48, num_attention_blocks=4):
        super(UnderwaterRestorationModel, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(base_channels)
        self.attention = nn.Sequential(*[
            ResidualAttentionBlock(base_channels * 4)
            for _ in range(num_attention_blocks)
        ])
        self.decoder = ImageDecoder(base_channels)
        nn.init.zeros_(self.decoder.out.weight)
        nn.init.zeros_(self.decoder.out.bias)
    def forward(self, x):
        _, _, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x
        f1, f2, f3 = self.feature_extractor(x_in)
        refined = self.attention(f3)
        correction = self.decoder(refined, f2, f1)
        output = x_in + correction
        return output[:, :, :h, :w]
if __name__ == "__main__":
    model = UnderwaterRestorationModel()
    image = torch.randn(1, 3, 256, 256)
    output = model(image)
    print("Input shape:", image.shape)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
