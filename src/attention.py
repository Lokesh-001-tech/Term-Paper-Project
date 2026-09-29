import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Learns which feature channels matter (helps with colour cast)."""

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
    """Learns which image regions matter (helps with haze / dark areas)."""

    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_features = torch.mean(x, dim=1, keepdim=True)
        max_features, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg_features, max_features], dim=1)

        attention = self.sigmoid(self.conv(combined))
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


class ResidualAttentionBlock(nn.Module):
    """
    conv -> ReLU -> conv -> channel+spatial attention, plus a skip
    connection. Stacking several of these gives the network a much
    larger receptive field and stronger attention than a single block.
    """

    def __init__(self, channels):
        super(ResidualAttentionBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

        self.attention = ChannelSpatialAttention(channels)

    def forward(self, x):
        return x + self.attention(self.body(x))