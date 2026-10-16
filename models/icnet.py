"""ICNet-style cascade with a shared ResNet v2 backbone.

The TensorFlow file does not build: it calls undefined names, adds tensors
whose channel counts differ, and concatenates maps of different sizes. This
module keeps the three-scale cascade, the pyramid sum, and the cascade feature
fusion, with matching channel counts, and returns full-resolution logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import PyramidSum, conv2d
from models.resnet import ResNetV2

_DEPTHS = {"ICNet-Res50": 50, "ICNet-Res101": 101, "ICNet-Res152": 152}


class CascadeFusion(nn.Module):
    def __init__(self, in_low: int, in_high: int, fused_ch: int, num_classes: int):
        super().__init__()
        self.cls = conv2d(in_low, num_classes, 1, bias=True)
        self.low_conv = conv2d(in_low, fused_ch, 3, dilation=2, bias=True)
        self.low_bn = nn.BatchNorm2d(fused_ch)
        self.high_conv = conv2d(in_high, fused_ch, 1, bias=True)
        self.high_bn = nn.BatchNorm2d(fused_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        low_up = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.cls(low_up)
        fused = self.relu(self.low_bn(self.low_conv(low_up)) + self.high_bn(self.high_conv(high)))
        return logits, fused


class ICNet(nn.Module):
    def __init__(self, preset_model: str = "ICNet-Res50", num_classes: int = 2, pooling_type: str = "MAX"):
        super().__init__()
        if preset_model not in _DEPTHS:
            raise ValueError(f"Unsupported ICNet model '{preset_model}'.")
        self.preset_model = preset_model
        self.backbone = ResNetV2(_DEPTHS[preset_model])
        self.pyramid = PyramidSum(256, pooling_type=pooling_type)
        self.fuse_mid = CascadeFusion(256, 256, 512, num_classes)
        self.fuse_high = CascadeFusion(512, 256, 512, num_classes)
        self.cls_quarter = conv2d(num_classes, num_classes, 1, bias=True)
        self.logits = conv2d(num_classes, num_classes, 1, bias=True)

    def _downsample(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = self.pyramid(self.backbone(self._downsample(x, 0.25))["pool3"])
        mid = self.backbone(self._downsample(x, 0.5))["pool3"]
        fine = self.backbone(x)["pool3"]
        _, fused_mid = self.fuse_mid(coarse, mid)
        out_fine, _ = self.fuse_high(fused_mid, fine)
        out = self.cls_quarter(F.interpolate(out_fine, scale_factor=2, mode="bilinear", align_corners=False))
        out = self.logits(F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False))
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
