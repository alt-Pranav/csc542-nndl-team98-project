from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


class ConvBlock(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU ops (decoder building block)."""

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


class UpBlock(nn.Module):
    """
    Decoder stage: bilinear upsample → concatenate encoder skip → ConvBlock.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=scale, mode="bilinear", align_corners=True
        )
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet18UNet(nn.Module):
    """
    U-Net with a ResNet18 encoder. Training can freeze the encoder for an
    initial block of epochs, then fine-tune with a lower encoder LR.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.encoder_layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.encoder_layer1 = backbone.layer1
        self.encoder_layer2 = backbone.layer2
        self.encoder_layer3 = backbone.layer3
        self.encoder_layer4 = backbone.layer4

        self._encoder_modules = [
            self.encoder_layer0,
            self.encoder_layer1,
            self.encoder_layer2,
            self.encoder_layer3,
            self.encoder_layer4,
        ]

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            ConvBlock(32, 16),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

    def hook_target_layers(self) -> List[Tuple[str, nn.Module]]:
        """(name, module) pairs for interpretability hooks — stable API for Baby U-Net parity."""
        return [
            ("enc_layer0", self.encoder_layer0),
            ("enc_layer1", self.encoder_layer1),
            ("enc_layer2", self.encoder_layer2),
            ("enc_layer3", self.encoder_layer3),
            ("enc_layer4", self.encoder_layer4),
            ("dec_up4", self.up4.conv),
            ("dec_up3", self.up3.conv),
            ("dec_up2", self.up2.conv),
            ("dec_up1", self.up1.conv),
            ("dec_up0", self.up0[1]),
        ]

    def freeze_encoder(self) -> None:
        for mod in self._encoder_modules:
            mod.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for mod in self._encoder_modules:
            mod.requires_grad_(True)

    def encoder_parameters(self):
        for mod in self._encoder_modules:
            yield from mod.parameters()

    def decoder_parameters(self):
        decoder_mods = [self.up4, self.up3, self.up2, self.up1, self.up0, self.out_conv]
        for mod in decoder_mods:
            yield from mod.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.encoder_layer0(x)
        e1 = self.encoder_layer1(e0)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3)

        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        d0 = self.up0(d1)

        return torch.sigmoid(self.out_conv(d0))
