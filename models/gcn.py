"""Global Convolutional Network with a ResNet v2 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvTranspose2dSame, conv2d
from models.resnet import ResNetV2

_DEPTHS = {"GCN-Res50": 50, "GCN-Res101": 101, "GCN-Res152": 152}


class GlobalConvBlock(nn.Module):
    def __init__(self, in_ch: int, n_filters: int = 21, size: int = 3):
        super().__init__()
        pad = size // 2
        self.left_a = nn.Conv2d(in_ch, n_filters, (size, 1), padding=(pad, 0), bias=True)
        self.left_b = nn.Conv2d(n_filters, n_filters, (1, size), padding=(0, pad), bias=True)
        self.right_a = nn.Conv2d(in_ch, n_filters, (1, size), padding=(0, pad), bias=True)
        self.right_b = nn.Conv2d(n_filters, n_filters, (size, 1), padding=(pad, 0), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left_b(self.left_a(x)) + self.right_b(self.right_a(x))


class BoundaryRefinement(nn.Module):
    def __init__(self, channels: int = 21):
        super().__init__()
        self.conv1 = conv2d(channels, channels, 3, bias=True)
        self.conv2 = conv2d(channels, channels, 3, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class _GCNStage(nn.Module):
    def __init__(self, in_ch: int, channels: int = 21):
        super().__init__()
        self.gcn = GlobalConvBlock(in_ch, channels, size=3)
        self.refine_in = BoundaryRefinement(channels)
        self.refine_sum = BoundaryRefinement(channels)
        self.up = ConvTranspose2dSame(channels, channels)

    def forward(self, features: torch.Tensor, higher: torch.Tensor | None) -> torch.Tensor:
        net = self.refine_in(self.gcn(features))
        if higher is not None:
            net = self.refine_sum(net + higher)
        return self.up(net)


class GCN(nn.Module):
    def __init__(self, preset_model: str = "GCN-Res101", num_classes: int = 2):
        super().__init__()
        if preset_model not in _DEPTHS:
            raise ValueError(f"Unsupported GCN model '{preset_model}'.")
        self.preset_model = preset_model
        self.backbone = ResNetV2(_DEPTHS[preset_model])
        # pool5, pool4, pool3, pool2 channel counts from ResNet v2.
        self.stage5 = _GCNStage(2048)
        self.stage4 = _GCNStage(512)
        self.stage3 = _GCNStage(256)
        self.stage2 = _GCNStage(64)
        self.refine = BoundaryRefinement(21)
        self.up = ConvTranspose2dSame(21, 21)
        self.refine_out = BoundaryRefinement(21)
        self.logits = conv2d(21, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        end_points = self.backbone(x)
        net = self.stage5(end_points["pool5"], None)
        net = self.stage4(end_points["pool4"], _match(net, end_points["pool4"]))
        net = self.stage3(end_points["pool3"], _match(net, end_points["pool3"]))
        net = self.stage2(end_points["pool2"], _match(net, end_points["pool2"]))
        net = self.up(self.refine(net))
        net = self.refine_out(net)
        if net.shape[-2:] != x.shape[-2:]:
            net = F.interpolate(net, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.logits(net)


def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x
