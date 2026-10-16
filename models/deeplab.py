"""DeepLabV3 and DeepLabV3+ with a ResNet v2 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import conv2d
from models.resnet import ResNetV2

_DEPTHS = {
    "DeepLabV3-Res50": 50,
    "DeepLabV3-Res101": 101,
    "DeepLabV3-Res152": 152,
    "DeepLabV3_plus-Res50": 50,
    "DeepLabV3_plus-Res101": 101,
    "DeepLabV3_plus-Res152": 152,
}


class ASPP(nn.Module):
    def __init__(self, in_ch: int, depth: int = 256, project: bool = True):
        super().__init__()
        self.image_conv = conv2d(in_ch, depth, 1, bias=True)
        self.conv1 = conv2d(in_ch, depth, 1, bias=True)
        self.conv6 = conv2d(in_ch, depth, 3, dilation=6, bias=True)
        self.conv12 = conv2d(in_ch, depth, 3, dilation=12, bias=True)
        self.conv18 = conv2d(in_ch, depth, 3, dilation=18, bias=True)
        self.project = conv2d(depth * 5, depth, 1, bias=True) if project else None
        self.out_ch = depth if project else depth * 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = self.image_conv(x.mean(dim=(2, 3), keepdim=True))
        image = F.interpolate(image, size=x.shape[-2:], mode="bilinear", align_corners=False)
        parts = [image, self.conv1(x), self.conv6(x), self.conv12(x), self.conv18(x)]
        net = torch.cat(parts, dim=1)
        if self.project is not None:
            net = self.project(net)
        return net


class DeepLabV3(nn.Module):
    def __init__(self, preset_model: str = "DeepLabV3-Res50", num_classes: int = 2):
        super().__init__()
        if preset_model not in ("DeepLabV3-Res50", "DeepLabV3-Res101", "DeepLabV3-Res152"):
            raise ValueError(f"Unsupported DeepLabV3 model '{preset_model}'.")
        self.preset_model = preset_model
        self.backbone = ResNetV2(_DEPTHS[preset_model])
        self.aspp = ASPP(512, depth=256, project=True)
        self.logits = conv2d(256, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)["pool4"]
        net = self.aspp(features)
        net = F.interpolate(net, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.logits(net)


class DeepLabV3Plus(nn.Module):
    def __init__(self, preset_model: str = "DeepLabV3_plus-Res50", num_classes: int = 2):
        super().__init__()
        if preset_model not in ("DeepLabV3_plus-Res50", "DeepLabV3_plus-Res101", "DeepLabV3_plus-Res152"):
            raise ValueError(f"Unsupported DeepLabV3+ model '{preset_model}'.")
        self.preset_model = preset_model
        self.backbone = ResNetV2(_DEPTHS[preset_model])
        self.aspp = ASPP(512, depth=256, project=False)
        self.aspp_proj = conv2d(self.aspp.out_ch, 256, 1, bias=True)
        self.low_proj = nn.Sequential(conv2d(64, 48, 1, bias=True), nn.ReLU(inplace=False))
        self.dec_conv1 = nn.Sequential(conv2d(304, 256, 3, bias=True), nn.ReLU(inplace=False))
        self.dec_conv2 = nn.Sequential(conv2d(256, 256, 3, bias=True), nn.ReLU(inplace=False))
        self.logits = conv2d(256, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        end_points = self.backbone(x)
        low = self.low_proj(end_points["pool2"])
        high = self.aspp_proj(self.aspp(end_points["pool4"]))
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        net = self.dec_conv2(self.dec_conv1(torch.cat([low, high], dim=1)))
        net = F.interpolate(net, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.logits(net)
