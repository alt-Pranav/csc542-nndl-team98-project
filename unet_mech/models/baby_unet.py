import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = {
    "in_channels": 1,
    "base_ch": 16,
    "max_ch": 128
}

def _ch(n: int) -> int:
    return min(n, CFG["max_ch"])

class ConvBlock(nn.Module):
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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.pad(
                x,
                [0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]],
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class BabyUNet(nn.Module):
    def __init__(self, in_channels: int = CFG["in_channels"], base_ch: int = CFG["base_ch"], pretrained=False):
        super().__init__()
        bc = base_ch
        
        self.enc0 = ConvBlock(in_channels, _ch(bc))
        self.enc1 = ConvBlock(_ch(bc),     _ch(bc * 2))
        self.enc2 = ConvBlock(_ch(bc * 2), _ch(bc * 4))
        self.enc3 = ConvBlock(_ch(bc * 4), _ch(bc * 8))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(_ch(bc * 8), _ch(bc * 16))

        self.up3 = UpBlock(_ch(bc * 16), _ch(bc * 8), _ch(bc * 8))
        self.up2 = UpBlock(_ch(bc * 8),  _ch(bc * 4), _ch(bc * 4))
        self.up1 = UpBlock(_ch(bc * 4),  _ch(bc * 2), _ch(bc * 2))
        self.up0 = UpBlock(_ch(bc * 2),  _ch(bc),     _ch(bc))

        self.out_conv = nn.Conv2d(_ch(bc), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)              
        e1 = self.enc1(self.pool(e0))  
        e2 = self.enc2(self.pool(e1))  
        e3 = self.enc3(self.pool(e2))  
        
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)

        out = self.out_conv(d0)
        return out

    def hook_target_layers(self):
        return [
            ("enc0", self.enc0),
            ("enc1", self.enc1),
            ("enc2", self.enc2),
            ("enc3", self.enc3),
            ("enc_layer4", self.bottleneck),
            ("dec3", self.up3),
            ("dec2", self.up2),
            ("dec1", self.up1),
            ("dec0", self.up0),
        ]
