"""SegNet-style encoder-decoder, with an optional additive skip path.

``Encoder-Decoder-Skip`` is the network defined in ``custom_model.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import ConvBNReLU, ConvTransposeBNReLU, conv2d


class EncoderDecoder(nn.Module):
    def __init__(self, preset_model: str = "Encoder-Decoder", num_classes: int = 2):
        super().__init__()
        if preset_model == "Encoder-Decoder":
            self.has_skip = False
        elif preset_model == "Encoder-Decoder-Skip":
            self.has_skip = True
        else:
            raise ValueError(f"Unsupported Encoder-Decoder model '{preset_model}'.")
        self.preset_model = preset_model

        self.down1 = nn.Sequential(ConvBNReLU(3, 64), ConvBNReLU(64, 64))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.down2 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.down3 = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256), ConvBNReLU(256, 256))
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.down4 = nn.Sequential(ConvBNReLU(256, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512))
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.down5 = nn.Sequential(ConvBNReLU(512, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512))
        self.pool5 = nn.MaxPool2d(2, stride=2)

        self.up4 = ConvTransposeBNReLU(512, 512)
        self.dec4 = nn.Sequential(ConvBNReLU(512, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512))
        self.up3 = ConvTransposeBNReLU(512, 512)
        self.dec3 = nn.Sequential(ConvBNReLU(512, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 256))
        self.up2 = ConvTransposeBNReLU(256, 256)
        self.dec2 = nn.Sequential(ConvBNReLU(256, 256), ConvBNReLU(256, 256), ConvBNReLU(256, 128))
        self.up1 = ConvTransposeBNReLU(128, 128)
        self.dec1 = nn.Sequential(ConvBNReLU(128, 128), ConvBNReLU(128, 64))
        self.up0 = ConvTransposeBNReLU(64, 64)
        self.dec0 = nn.Sequential(ConvBNReLU(64, 64), ConvBNReLU(64, 64))
        self.logits = conv2d(64, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.down1(x))
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
