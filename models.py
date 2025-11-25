# ==========================================
# Minimal U-Net + Training/Validation Loop for KITTI Depth Inpainting
# works with KITTIDepthInpaintingDataset + loaders
# ==========================================
import math, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# U-Net building blocks
# -----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch)  # cheap downsample
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed (in case of odd dims)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, (0, diffX, 0, diffY))
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x

class DepthUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16
        self.enc1 = ConvBlock(in_ch, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5 + c4, c4)
        self.up2 = Up(c4 + c3, c3)
        self.up3 = Up(c3 + c2, c2)
        self.up4 = Up(c2 + c1, c1)

        self.head = nn.Conv2d(c1, 1, 1)

        # Kaiming init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)         # c1
        e2 = self.down1(e1)       # c2
        e3 = self.down2(e2)       # c3
        e4 = self.down3(e3)       # c4
        e5 = self.down4(e4)       # c5

        d1 = self.up1(e5, e4)     # c4
        d2 = self.up2(d1, e3)     # c3
        d3 = self.up3(d2, e2)     # c2
        d4 = self.up4(d3, e1)     # c1

        out = self.head(d4)       # (B,1,H,W)
        return out
    


# ==========================================
# Depth Autoencoder (same as DepthUNet but WITHOUT skip connections)
# For ablation: no encoder–decoder skip paths
# ==========================================

class UpNoSkip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Note: no concatenation with encoder features, so ConvBlock just sees in_ch
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.block(x)
        return x


class DepthAutoencoder(nn.Module):
    """
    Same encoder depth and channel widths as DepthUNet, but decoder has NO skip connections.
    This is a clean ablation to isolate the effect of skip connections.
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        # Encoder (identical to DepthUNet)
        self.enc1 = ConvBlock(in_ch, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        # Decoder WITHOUT skip connections
        # Channel flow: c5 -> c4 -> c3 -> c2 -> c1
        self.up1 = UpNoSkip(c5, c4)
        self.up2 = UpNoSkip(c4, c3)
        self.up3 = UpNoSkip(c3, c2)
        self.up4 = UpNoSkip(c2, c1)

        self.head = nn.Conv2d(c1, 1, 1)

        # Same Kaiming init as DepthUNet
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # Decoder path (no skips; only bottleneck features)
        d1 = self.up1(e5)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
        d4 = self.up4(d3)

        out = self.head(d4) # (B,1,H,W)
        return out
