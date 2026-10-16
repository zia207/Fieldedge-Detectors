"""RefineNet with a ResNet v2 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBNReLU, ConvTransposeBNReLU, conv2d
from models.resnet import ResNetV2

_DEPTHS = {"RefineNet-Res50": 50, "RefineNet-Res101": 101, "RefineNet-Res152": 152}


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = conv2d(channels, channels, 3, bias=True)
        self.conv2 = conv2d(channels, channels, 3, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.conv1(self.relu(x))
        net = self.conv2(self.relu(net))
        return net + x


class ChainedResidualPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = conv2d(channels, channels, 3, bias=True)
        self.conv2 = conv2d(channels, channels, 3, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.relu(x)
        pooled = self.conv1(self.pool(net))
        summed = pooled + net
        pooled = self.conv2(self.pool(pooled))
        return pooled + summed


class RefineBlock(nn.Module):
    """One RefineNet block. ``channels`` is 512 for the coarsest block and 256 otherwise."""

    def __init__(self, channels: int, top: bool, low_channels: int | None = None):
        super().__init__()
        self.top = top
        self.rcu1 = ResidualConvUnit(channels)
        self.rcu2 = ResidualConvUnit(channels)
        incoming = channels if low_channels is None else low_channels
        self.adapt_low = nn.Sequential(conv2d(incoming, channels, 3, bias=True), nn.ReLU(inplace=False))
        if not top:
            self.adapt_high = nn.Sequential(conv2d(channels, channels, 3, bias=True), nn.ReLU(inplace=False))
        self.pool = ChainedResidualPooling(channels)
        self.out = ResidualConvUnit(channels)

    def forward(self, high: torch.Tensor, low: torch.Tensor | None = None) -> torch.Tensor:
        if self.top:
            net = self.rcu2(self.rcu1(high))
            fused = self.adapt_low(net)
        else:
            high = self.rcu2(self.rcu1(high))
            low = F.interpolate(self.adapt_low(low), scale_factor=2, mode="bilinear", align_corners=False)
            if low.shape[-2:] != high.shape[-2:]:
                low = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
            fused = low + self.adapt_high(high)
        return self.out(self.pool(fused))


class RefineNet(nn.Module):
    def __init__(
        self,
        preset_model: str = "RefineNet-Res101",
        num_classes: int = 2,
        upscaling_method: str = "bilinear",
    ):
        super().__init__()
        if preset_model not in _DEPTHS:
            raise ValueError(f"Unsupported RefineNet model '{preset_model}'.")
        self.preset_model = preset_model
        self.upscaling_method = upscaling_method.lower()
        self.backbone = ResNetV2(_DEPTHS[preset_model])
        self.proj5 = nn.Sequential(conv2d(2048, 512, 1, bias=True), nn.ReLU(inplace=False))
        self.proj4 = nn.Sequential(conv2d(512, 256, 1, bias=True), nn.ReLU(inplace=False))
        self.proj3 = nn.Sequential(conv2d(256, 256, 1, bias=True), nn.ReLU(inplace=False))
        self.proj2 = nn.Sequential(conv2d(64, 256, 1, bias=True), nn.ReLU(inplace=False))
        self.block4 = RefineBlock(512, top=True)
        self.block3 = RefineBlock(256, top=False, low_channels=512)
        self.block2 = RefineBlock(256, top=False, low_channels=256)
        self.block1 = RefineBlock(256, top=False, low_channels=256)
        self.rcu1 = ResidualConvUnit(256)
        self.rcu2 = ResidualConvUnit(256)
        if self.upscaling_method == "conv":
            self.up1 = ConvTransposeBNReLU(256, 128)
            self.conv1 = ConvBNReLU(128, 128)
            self.up2 = ConvTransposeBNReLU(128, 64)
            self.conv2 = ConvBNReLU(64, 64)
            logit_ch = 64
        elif self.upscaling_method == "bilinear":
            logit_ch = 256
        else:
            raise ValueError(f"Unsupported upscaling method '{upscaling_method}'.")
        self.logits = conv2d(logit_ch, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        end_points = self.backbone(x)
        high5 = self.proj5(end_points["pool5"])
        high4 = self.proj4(end_points["pool4"])
        high3 = self.proj3(end_points["pool3"])
        high2 = self.proj2(end_points["pool2"])
        low = self.block4(high5, None)
        low = self.block3(high4, low)
        low = self.block2(high3, low)
        net = self.block1(high2, low)
        net = self.rcu2(self.rcu1(net))
        if self.upscaling_method == "conv":
            net = self.conv1(self.up1(net))
            net = self.conv2(self.up2(net))
        else:
            net = F.interpolate(net, scale_factor=4, mode="bilinear", align_corners=False)
        if net.shape[-2:] != x.shape[-2:]:
            net = F.interpolate(net, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.logits(net)
