"""FC-DenseNet (Tiramisu) segmentation models."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import PreactConv, conv2d


class DenseBlock(nn.Module):
    def __init__(self, in_ch: int, n_layers: int, growth_rate: int, dropout_p: float):
        super().__init__()
        layers = []
        channels = in_ch
        for _ in range(n_layers):
            layers.append(PreactConv(channels, growth_rate, kernel_size=3, dropout_p=dropout_p))
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_ch = channels
        self.new_ch = growth_rate * n_layers

    def forward(self, x: torch.Tensor):
        grown = []
        for layer in self.layers:
            features = layer(x)
            grown.append(features)
            x = torch.cat([x, features], dim=1)
        return x, torch.cat(grown, dim=1)


class TransitionDown(nn.Module):
    def __init__(self, in_ch: int, n_filters: int, dropout_p: float):
        super().__init__()
        self.conv = PreactConv(in_ch, n_filters, kernel_size=1, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class TransitionUp(nn.Module):
    def __init__(self, in_ch: int, n_filters: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = x[:, :, : skip.shape[-2], : skip.shape[-1]]
        return torch.cat([x, skip], dim=1)


_PRESETS = {
    "FC-DenseNet56": {"n_pool": 5, "growth_rate": 12, "n_layers_per_block": 4},
    "FC-DenseNet67": {"n_pool": 5, "growth_rate": 16, "n_layers_per_block": 5},
    "FC-DenseNet103": {
        "n_pool": 5,
        "growth_rate": 16,
        "n_layers_per_block": [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
    },
}


class FCDenseNet(nn.Module):
    def __init__(
        self,
        preset_model: str = "FC-DenseNet56",
        num_classes: int = 2,
        n_filters_first_conv: int = 48,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        if preset_model not in _PRESETS:
            raise ValueError(f"Unsupported FC-DenseNet model '{preset_model}'.")
        spec = _PRESETS[preset_model]
        n_pool = spec["n_pool"]
        growth_rate = spec["growth_rate"]
        layers = spec["n_layers_per_block"]
        if isinstance(layers, int):
            layers = [layers] * (2 * n_pool + 1)
        self.preset_model = preset_model
        self.first = conv2d(3, n_filters_first_conv, 3, bias=True)

        channels = n_filters_first_conv
        self.down_blocks = nn.ModuleList()
        self.down_transitions = nn.ModuleList()
        skip_channels = []
        for index in range(n_pool):
            block = DenseBlock(channels, layers[index], growth_rate, dropout_p)
            channels = block.out_ch
            skip_channels.append(channels)
            self.down_blocks.append(block)
            self.down_transitions.append(TransitionDown(channels, channels, dropout_p))

        self.bottleneck = DenseBlock(channels, layers[n_pool], growth_rate, dropout_p)
        up_channels = self.bottleneck.new_ch
        self.up_transitions = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for index, skip_ch in enumerate(reversed(skip_channels)):
            n_keep = growth_rate * layers[n_pool + index]
            self.up_transitions.append(TransitionUp(up_channels, n_keep))
            block = DenseBlock(n_keep + skip_ch, layers[n_pool + index + 1], growth_rate, dropout_p)
            self.up_blocks.append(block)
            up_channels = block.new_ch
        self.logits = conv2d(self.up_blocks[-1].out_ch, num_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        skips = []
        for block, transition in zip(self.down_blocks, self.down_transitions):
            x, _ = block(x)
            skips.append(x)
            x = transition(x)
        x, upsample = self.bottleneck(x)
        for skip, transition, block in zip(reversed(skips), self.up_transitions, self.up_blocks):
            x = transition(upsample, skip)
            x, upsample = block(x)
        return self.logits(x)
