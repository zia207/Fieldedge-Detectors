"""Full-Resolution Residual Networks (FRRN-A and FRRN-B)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import conv2d


class ResidualUnit(nn.Module):
    """Two 3x3 convolutions with batch norm. The source does not add a skip."""

    def __init__(self, in_ch: int, n_filters: int = 48):
        super().__init__()
        self.conv1 = conv2d(in_ch, n_filters, 3, bias=True)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = conv2d(n_filters, n_filters, 3, bias=True)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(x))


class FullResolutionResidualUnit(nn.Module):
    def __init__(self, pool_ch: int, res_ch: int, n_filters_3: int, n_filters_1: int, pool_scale: int):
        super().__init__()
        self.pool_scale = pool_scale
        self.conv1 = conv2d(pool_ch + res_ch, n_filters_3, 3, bias=True)
        self.bn1 = nn.BatchNorm2d(n_filters_3)
        self.conv2 = conv2d(n_filters_3, n_filters_3, 3, bias=True)
        self.bn2 = nn.BatchNorm2d(n_filters_3)
        self.proj = conv2d(n_filters_3, n_filters_1, 1, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.out_channels = n_filters_3

    def forward(self, pool_stream: torch.Tensor, res_stream: torch.Tensor):
        pooled_res = F.max_pool2d(res_stream, kernel_size=self.pool_scale, stride=self.pool_scale)
        if pooled_res.shape[-2:] != pool_stream.shape[-2:]:
            pooled_res = F.interpolate(pooled_res, size=pool_stream.shape[-2:], mode="nearest")
        net = self.relu(self.bn1(self.conv1(torch.cat([pool_stream, pooled_res], dim=1))))
        net = self.relu(self.bn2(self.conv2(net)))
        residual = F.interpolate(self.proj(net), scale_factor=self.pool_scale, mode="nearest")
        if residual.shape[-2:] != res_stream.shape[-2:]:
            residual = F.interpolate(residual, size=res_stream.shape[-2:], mode="nearest")
        return net, res_stream + residual


def _spec(preset_model: str):
    down = [(2, 96, 3), (4, 192, 4), (8, 384, 2), (16, 384, 2)]
    up = [(8, 192, 2), (4, 192, 2), (2, 96, 2)]
    if preset_model == "FRRN-B":
        # The source uses pool_scale 17 on the first up stage. That size does
        # not match the unpooled map, so the stage uses 16.
        down.append((32, 384, 2))
        up.insert(0, (16, 192, 2))
    elif preset_model != "FRRN-A":
        raise ValueError(f"Unsupported FRRN model '{preset_model}'.")
    return down, up


class FRRN(nn.Module):
    def __init__(self, preset_model: str = "FRRN-A", num_classes: int = 2):
        super().__init__()
        self.preset_model = preset_model
        down_spec, up_spec = _spec(preset_model)
        self.stem = nn.Sequential(
            conv2d(3, 48, 5, bias=True),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=False),
        )
        self.stem_units = nn.Sequential(ResidualUnit(48, 48), ResidualUnit(48, 48), ResidualUnit(48, 48))
        self.res_proj = conv2d(48, 32, 1, bias=True)

        self.down_pools = nn.ModuleList()
        self.down_units = nn.ModuleList()
        pool_ch = 48
        for scale, filters, count in down_spec:
            self.down_pools.append(nn.MaxPool2d(2, stride=2))
            stage = nn.ModuleList()
            for _ in range(count):
                unit = FullResolutionResidualUnit(pool_ch, 32, filters, 32, scale)
                stage.append(unit)
                pool_ch = unit.out_channels
            self.down_units.append(stage)

        self.up_units = nn.ModuleList()
        for scale, filters, count in up_spec:
            stage = nn.ModuleList()
            for _ in range(count):
                unit = FullResolutionResidualUnit(pool_ch, 32, filters, 32, scale)
                stage.append(unit)
                pool_ch = unit.out_channels
            self.up_units.append(stage)

        self.final_units = nn.Sequential(
            ResidualUnit(pool_ch + 32, 48),
            ResidualUnit(48, 48),
            ResidualUnit(48, 48),
        )
        self.logits = conv2d(48, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_units(self.stem(x))
        res_stream = self.res_proj(x)
        pool_stream = x
        for pool, stage in zip(self.down_pools, self.down_units):
            pool_stream = pool(pool_stream)
            for unit in stage:
                pool_stream, res_stream = unit(pool_stream, res_stream)
        for stage in self.up_units:
            pool_stream = F.interpolate(pool_stream, scale_factor=2, mode="bilinear", align_corners=False)
            for unit in stage:
                pool_stream, res_stream = unit(pool_stream, res_stream)
        if pool_stream.shape[-2:] != res_stream.shape[-2:]:
            pool_stream = F.interpolate(pool_stream, size=res_stream.shape[-2:], mode="bilinear", align_corners=False)
        net = torch.cat([pool_stream, res_stream], dim=1)
        return self.logits(self.final_units(net))
