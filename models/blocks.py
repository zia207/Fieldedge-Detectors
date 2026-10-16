"""Shared convolution blocks used by the segmentation models.

Tensor layout is NCHW. Spatial padding matches the TensorFlow SAME padding
used by the slim layers these blocks replace.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d(in_ch: int, out_ch: int, kernel_size, stride: int = 1, dilation: int = 1, bias: bool = True) -> nn.Conv2d:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    padding = tuple(dilation * (k - 1) // 2 for k in kernel_size)
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )


class ConvBNReLU(nn.Module):
    """Convolution, batch norm, ReLU, then optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride: int = 1, dropout_p: float = 0.0):
        super().__init__()
        self.conv = conv2d(in_ch, out_ch, kernel_size, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout_p) if dropout_p else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.relu(self.bn(self.conv(x))))


class PreactConv(nn.Module):
    """Batch norm, ReLU, convolution, then optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, dropout_p: float = 0.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=False)
        self.conv = conv2d(in_ch, out_ch, kernel_size, bias=True)
        self.drop = nn.Dropout(dropout_p) if dropout_p else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.conv(self.relu(self.bn(x))))


class ConvTransposeBNReLU(nn.Module):
    """Stride-2 transposed convolution, batch norm, and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.deconv(x)))


class ConvTranspose2dSame(nn.Module):
    """Stride-2 transposed convolution without normalization."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size, stride=2, padding=1, output_padding=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


def align_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Bilinear-resize ``x`` to the spatial size of ``ref`` when they differ."""
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class InterpBlock(nn.Module):
    """Pool to a fixed bin grid, project, and resize back to the input map."""

    def __init__(self, in_ch: int, out_ch: int, level: int, pooling_type: str = "MAX"):
        super().__init__()
        self.level = level
        self.pooling_type = pooling_type.upper()
        self.conv = conv2d(in_ch, out_ch, 1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = (self.level, self.level)
        if self.pooling_type == "AVG":
            pooled = F.adaptive_avg_pool2d(x, size)
        else:
            pooled = F.adaptive_max_pool2d(x, size)
        pooled = self.relu(self.bn(self.conv(pooled)))
        return F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)


class PyramidPooling(nn.Module):
    """PSPNet pyramid. Concatenates the input with four pooled scales."""

    def __init__(self, in_ch: int, out_ch: int = 512, pooling_type: str = "MAX"):
        super().__init__()
        self.blocks = nn.ModuleList(
            InterpBlock(in_ch, out_ch, level, pooling_type) for level in (1, 2, 3, 6)
        )
        self.out_ch = in_ch + out_ch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x] + [block(x) for block in self.blocks]
        return torch.cat(parts, dim=1)


class PyramidSum(nn.Module):
    """ICNet pyramid. Sums the input with four pooled scales of the same depth."""

    def __init__(self, in_ch: int, pooling_type: str = "MAX"):
        super().__init__()
        self.blocks = nn.ModuleList(
            InterpBlock(in_ch, in_ch, level, pooling_type) for level in (1, 2, 3, 6)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = out + block(x)
        return out
