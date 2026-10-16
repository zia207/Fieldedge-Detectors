"""Pyramid Scene Parsing Network with a ResNet v2 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBNReLU, ConvTransposeBNReLU, PyramidPooling, conv2d
from models.resnet import ResNetV2

_DEPTHS = {"PSPNet-Res50": 50, "PSPNet-Res101": 101, "PSPNet-Res152": 152}


class PSPNet(nn.Module):
    def __init__(
        self,
        preset_model: str = "PSPNet-Res50",
        num_classes: int = 2,
        pooling_type: str = "MAX",
        upscaling_method: str = "conv",
    ):
        super().__init__()
        if preset_model not in _DEPTHS:
            raise ValueError(f"Unsupported PSPNet model '{preset_model}'.")
        self.preset_model = preset_model
        self.upscaling_method = upscaling_method.lower()
        self.backbone = ResNetV2(_DEPTHS[preset_model])
        self.pyramid = PyramidPooling(256, out_ch=512, pooling_type=pooling_type)
        self.conv = nn.Sequential(
            conv2d(self.pyramid.out_ch, 512, 3, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        if self.upscaling_method == "conv":
            self.up1 = ConvTransposeBNReLU(512, 256)
            self.block1 = ConvBNReLU(256, 256)
            self.up2 = ConvTransposeBNReLU(256, 128)
            self.block2 = ConvBNReLU(128, 128)
            self.up3 = ConvTransposeBNReLU(128, 64)
            self.block3 = ConvBNReLU(64, 64)
            logit_ch = 64
        elif self.upscaling_method == "bilinear":
            logit_ch = 512
        else:
            raise ValueError(f"Unsupported upscaling method '{upscaling_method}'.")
        self.logits = conv2d(logit_ch, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)["pool3"]
        net = self.conv(self.pyramid(features))
        if self.upscaling_method == "conv":
            net = self.block1(self.up1(net))
            net = self.block2(self.up2(net))
            net = self.block3(self.up3(net))
        else:
            net = F.interpolate(net, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if net.shape[-2:] != x.shape[-2:]:
            net = F.interpolate(net, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.logits(net)
