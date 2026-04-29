from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ch(n: int, max_ch: int = 128) -> int:
    return min(n, max_ch)


class BabyConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers used by the scratch Baby U-Net."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BabyUpBlock(nn.Module):
    """Bilinear upsample, concatenate skip connection, then convolve."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = BabyConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(
                x,
                [
                    0,
                    skip.shape[-1] - x.shape[-1],
                    0,
                    skip.shape[-2] - x.shape[-2],
                ],
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BabyUNet(nn.Module):
    """Scratch compact U-Net matching the `baby_unet_best.pth` checkpoint."""

    def __init__(self, in_channels: int = 1, base_ch: int = 16, max_ch: int = 128):
        super().__init__()
        bc = base_ch
        self.max_ch = max_ch

        self.enc0 = BabyConvBlock(in_channels, _ch(bc, max_ch))
        self.enc1 = BabyConvBlock(_ch(bc, max_ch), _ch(bc * 2, max_ch))
        self.enc2 = BabyConvBlock(_ch(bc * 2, max_ch), _ch(bc * 4, max_ch))
        self.enc3 = BabyConvBlock(_ch(bc * 4, max_ch), _ch(bc * 8, max_ch))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = BabyConvBlock(_ch(bc * 8, max_ch), _ch(bc * 16, max_ch))

        self.up3 = BabyUpBlock(_ch(bc * 16, max_ch), _ch(bc * 8, max_ch), _ch(bc * 8, max_ch))
        self.up2 = BabyUpBlock(_ch(bc * 8, max_ch), _ch(bc * 4, max_ch), _ch(bc * 4, max_ch))
        self.up1 = BabyUpBlock(_ch(bc * 4, max_ch), _ch(bc * 2, max_ch), _ch(bc * 2, max_ch))
        self.up0 = BabyUpBlock(_ch(bc * 2, max_ch), _ch(bc, max_ch), _ch(bc, max_ch))
        self.out_conv = nn.Conv2d(_ch(bc, max_ch), 1, kernel_size=1)

        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

    def hook_target_layers(self) -> List[Tuple[str, nn.Module]]:
        return [
            ("enc0", self.enc0),
            ("enc1", self.enc1),
            ("enc2", self.enc2),
            ("enc3", self.enc3),
            ("bottleneck", self.bottleneck),
            ("up3", self.up3.conv),
            ("up2", self.up2.conv),
            ("up1", self.up1.conv),
            ("up0", self.up0.conv),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))

        d3 = self.up3(bn, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)
        return torch.sigmoid(self.out_conv(d0))
