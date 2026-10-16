"""AdapNet segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import conv2d


class ConvBlock(nn.Module):
    """Pre-activation convolution: batch norm, ReLU, then convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride: int = 1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=False)
        self.conv = conv2d(in_ch, out_ch, kernel_size, stride=stride, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.relu(self.bn(x)))


class ResNetBlock1(nn.Module):
    def __init__(self, channels: int, filters_1: int, filters_2: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, filters_1, 1)
        self.conv2 = ConvBlock(filters_1, filters_1, 3)
        self.conv3 = ConvBlock(filters_1, filters_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv3(self.conv2(self.conv1(x)))


class ResNetBlock2(nn.Module):
    def __init__(self, in_ch: int, filters_1: int, filters_2: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, filters_1, 1, stride=stride)
        self.conv2 = ConvBlock(filters_1, filters_1, 3)
        self.conv3 = ConvBlock(filters_1, filters_2, 1)
        self.proj = ConvBlock(in_ch, filters_2, 1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(self.conv1(x))) + self.proj(x)


class MultiscaleBlock1(nn.Module):
    def __init__(self, channels: int, filters_1: int, filters_2: int, filters_3: int, rate_p: int, rate_d: int):
        super().__init__()
        half = filters_3 // 2
        self.reduce = ConvBlock(channels, filters_1, 1)
        self.scale_p = ConvBlock(filters_1, half, 3)
        self.scale_p.conv = conv2d(filters_1, half, 3, dilation=rate_p, bias=True)
        self.scale_d = ConvBlock(filters_1, half, 3)
        self.scale_d.conv = conv2d(filters_1, half, 3, dilation=rate_d, bias=True)
        self.expand = ConvBlock(filters_3, filters_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.reduce(x)
        merged = torch.cat([self.scale_p(net), self.scale_d(net)], dim=1)
        return x + self.expand(merged)


class MultiscaleBlock2(nn.Module):
    def __init__(self, in_ch: int, filters_1: int, filters_2: int, filters_3: int, rate_p: int, rate_d: int):
        super().__init__()
        half = filters_3 // 2
        self.reduce = ConvBlock(in_ch, filters_1, 1)
        self.scale_p = ConvBlock(filters_1, half, 3)
        self.scale_p.conv = conv2d(filters_1, half, 3, dilation=rate_p, bias=True)
        self.scale_d = ConvBlock(filters_1, half, 3)
        self.scale_d.conv = conv2d(filters_1, half, 3, dilation=rate_d, bias=True)
        self.expand = ConvBlock(filters_3, filters_2, 1)
        self.proj = ConvBlock(in_ch, filters_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.reduce(x)
        merged = torch.cat([self.scale_p(net), self.scale_d(net)], dim=1)
        return self.expand(merged) + self.proj(x)


class AdapNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, 3, stride=1)
        self.conv2 = ConvBlock(64, 64, 7, stride=2)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.layer1 = nn.Sequential(
            ResNetBlock2(64, 64, 256, stride=1),
            ResNetBlock1(256, 64, 256),
            ResNetBlock1(256, 64, 256),
        )
        self.layer2 = nn.Sequential(
            ResNetBlock2(256, 128, 512, stride=2),
            ResNetBlock1(512, 128, 512),
            ResNetBlock1(512, 128, 512),
        )
        self.skip = ConvBlock(512, 12, 1)
        self.ms1 = MultiscaleBlock1(512, 128, 512, 64, rate_p=1, rate_d=2)
        self.layer3 = nn.Sequential(
            ResNetBlock2(512, 256, 1024, stride=2),
            ResNetBlock1(1024, 256, 1024),
            MultiscaleBlock1(1024, 256, 1024, 64, rate_p=1, rate_d=2),
            MultiscaleBlock1(1024, 256, 1024, 64, rate_p=1, rate_d=4),
            MultiscaleBlock1(1024, 256, 1024, 64, rate_p=1, rate_d=8),
            MultiscaleBlock1(1024, 256, 1024, 64, rate_p=1, rate_d=16),
        )
        self.layer4 = nn.Sequential(
            MultiscaleBlock2(1024, 512, 2048, 512, rate_p=2, rate_d=4),
            MultiscaleBlock1(2048, 512, 2048, 512, rate_p=2, rate_d=8),
            MultiscaleBlock1(2048, 512, 2048, 512, rate_p=2, rate_d=16),
        )
        self.head = ConvBlock(2048, 12, 1)
        self.logits = conv2d(12, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.layer2(self.layer1(x))
        skip = self.skip(x)
        x = self.layer4(self.layer3(self.ms1(x)))
        x = self.head(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x + skip
        x = F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False)
        return self.logits(x)
