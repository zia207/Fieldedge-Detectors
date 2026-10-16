"""MobileUNet with depthwise-separable convolutions.

The TensorFlow stem calls ``conv2d`` without a kernel size, so that line cannot
run. The stem here is a 3x3 convolution and ReLU, which is what that helper
applies once a kernel is supplied.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import ConvTransposeBNReLU, conv2d


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=True)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.dw_bn(self.depthwise(x)))
        return self.relu(self.pw_bn(self.pointwise(x)))


def _stack(in_ch: int, channels: list[int]) -> nn.Sequential:
    layers = []
    current = in_ch
    for out_ch in channels:
        layers.append(DepthwiseSeparableConv(current, out_ch))
        current = out_ch
    return nn.Sequential(*layers)


class MobileUNet(nn.Module):
    def __init__(self, preset_model: str = "MobileUNet", num_classes: int = 2):
        super().__init__()
        if preset_model == "MobileUNet":
            self.has_skip = False
        elif preset_model == "MobileUNet-Skip":
            self.has_skip = True
        else:
            raise ValueError(f"Unsupported MobileUNet model '{preset_model}'.")
        self.preset_model = preset_model

        self.stem = nn.Sequential(conv2d(3, 64, 3, bias=True), nn.ReLU(inplace=False))
        self.down1 = _stack(64, [64])
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.down2 = _stack(64, [128, 128])
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.down3 = _stack(128, [256, 256, 256])
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.down4 = _stack(256, [512, 512, 512])
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.down5 = _stack(512, [512, 512, 512])
        self.pool5 = nn.MaxPool2d(2, stride=2)

        self.up4 = ConvTransposeBNReLU(512, 512)
        self.dec4 = _stack(512, [512, 512, 512])
        self.up3 = ConvTransposeBNReLU(512, 512)
        self.dec3 = _stack(512, [512, 512, 256])
        self.up2 = ConvTransposeBNReLU(256, 256)
        self.dec2 = _stack(256, [256, 256, 128])
        self.up1 = ConvTransposeBNReLU(128, 128)
        self.dec1 = _stack(128, [128, 64])
        self.up0 = ConvTransposeBNReLU(64, 64)
        self.dec0 = _stack(64, [64, 64])
        self.logits = conv2d(64, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.down1(self.stem(x)))
        skip1 = x
        x = self.pool2(self.down2(x))
        skip2 = x
        x = self.pool3(self.down3(x))
        skip3 = x
        x = self.pool4(self.down4(x))
        skip4 = x
        x = self.pool5(self.down5(x))

        x = self.dec4(self.up4(x))
        if self.has_skip:
            x = x + skip4
        x = self.dec3(self.up3(x))
        if self.has_skip:
            x = x + skip3
        x = self.dec2(self.up2(x))
        if self.has_skip:
            x = x + skip2
        x = self.dec1(self.up1(x))
        if self.has_skip:
            x = x + skip1
        x = self.dec0(self.up0(x))
        return self.logits(x)
