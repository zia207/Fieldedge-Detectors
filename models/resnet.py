"""Pre-activation ResNet v2 backbone.

This follows ``cropboundarycode/models/resnet_v2.py``. The segmentation models
call that network with the default arguments, so there is no atrous output
stride and no classification head. Feature maps match the names those models
read:

- ``pool2``: stem (7x7 stride 2, then 3x3 max pool), stride 4, 64 channels
- ``pool3``: end of block1, stride 8, 256 channels
- ``pool4``: end of block2, stride 16, 512 channels
- ``pool5``: end of block4 after the final batch-norm and ReLU, stride 32, 2048 channels

Downsampling sits on the last unit of each block, as in the slim implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckV2(nn.Module):
    def __init__(self, in_ch: int, depth: int, depth_bottleneck: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.preact_bn = nn.BatchNorm2d(in_ch)
        self.preact_relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_ch, depth_bottleneck, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(depth_bottleneck)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            depth_bottleneck,
            depth_bottleneck,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(depth_bottleneck)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(depth_bottleneck, depth, 1, bias=True)
        self.project = in_ch != depth
        if self.project:
            self.shortcut = nn.Conv2d(in_ch, depth, 1, stride=stride, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preact = self.preact_relu(self.preact_bn(x))
        if self.project:
            shortcut = self.shortcut(preact)
        elif self.stride > 1:
            shortcut = x[:, :, :: self.stride, :: self.stride]
        else:
            shortcut = x
        residual = self.relu1(self.bn1(self.conv1(preact)))
        residual = self.relu2(self.bn2(self.conv2(residual)))
        residual = self.conv3(residual)
        return shortcut + residual


def _make_block(in_ch: int, base_depth: int, num_units: int, stride: int) -> nn.Sequential:
    depth = base_depth * 4
    units = []
    channels = in_ch
    for index in range(num_units):
        unit_stride = stride if index == num_units - 1 else 1
        units.append(BottleneckV2(channels, depth, base_depth, stride=unit_stride))
        channels = depth
    return nn.Sequential(*units)


class ResNetV2(nn.Module):
    """ResNet v2 feature extractor. ``depth`` is 50, 101, or 152."""

    LAYER_COUNTS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}

    def __init__(self, depth: int = 50):
        super().__init__()
        if depth not in self.LAYER_COUNTS:
            raise ValueError(f"Unsupported ResNet depth {depth}. Use 50, 101, or 152.")
        units = self.LAYER_COUNTS[depth]
        self.depth = depth
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = _make_block(64, 64, units[0], stride=2)
        self.block2 = _make_block(256, 128, units[1], stride=2)
        self.block3 = _make_block(512, 256, units[2], stride=2)
        self.block4 = _make_block(1024, 512, units[3], stride=1)
        self.post_bn = nn.BatchNorm2d(2048)
        self.post_relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.pool1(self.conv1(x))
        pool2 = x
        pool3 = self.block1(x)
        pool4 = self.block2(pool3)
        x = self.block3(pool4)
        x = self.post_relu(self.post_bn(self.block4(x)))
        return {"pool2": pool2, "pool3": pool3, "pool4": pool4, "pool5": x}
