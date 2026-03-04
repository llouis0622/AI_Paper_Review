import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = F.max_pool2d(c1, 2)
        c2 = self.down2(p1)
        p2 = F.max_pool2d(c2, 2)
        c3 = self.down3(p2)

        u2 = F.interpolate(c3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.up2(u2)

        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.up1(u1)

        return self.final(u1)
