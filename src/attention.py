import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, channels):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        attention = self.avg_pool(x)
        attention = self.fc(attention)

        return x * attention


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_features = torch.mean(x, dim=1, keepdim=True)

        max_features, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat(
            [avg_features, max_features],
            dim=1
        )

        attention = self.conv(combined)
        attention = self.sigmoid(attention)

        return x * attention


class ChannelSpatialAttention(nn.Module):

    def __init__(self, channels):
        super(ChannelSpatialAttention, self).__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x