"""
Dual-stream radar fusion model — separate encoders for horizontal and vertical
radar heatmaps, fused at the bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ResidualBlock, CBAM, SoftArgmax2D


class HoriEncoder(nn.Module):
    """Encoder for the horizontal radar heatmap (1 channel)."""
    def __init__(self, out_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32, 64,     stride=2), CBAM(64),
            ResidualBlock(64, out_ch, stride=2), CBAM(out_ch),
        )
    def forward(self, x): return self.net(x)


class VertEncoder(nn.Module):
    """Encoder for the vertical radar heatmap (1 channel)."""
    def __init__(self, out_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32, 64,     stride=2), CBAM(64),
            ResidualBlock(64, out_ch, stride=2), CBAM(out_ch),
        )
    def forward(self, x): return self.net(x)


class FusionModel(nn.Module):
    """
    Dual-stream fusion model: separate encoders for horizontal and vertical
    radar heatmaps, fused at the bottleneck, shared decoder.

    Input:  radar (B, 2, RADAR_H, RADAR_W)
    Output: heatmaps (B, 17, H', W'), coords (B, 17, 2)
    """

    def __init__(self, num_kp=17):
        super().__init__()
        self.hori_enc = HoriEncoder(out_ch=128)
        self.vert_enc = VertEncoder(out_ch=128)

        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 256, stride=2), CBAM(256)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        # Decoder — bilinear upsample + conv, no skip connections
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,  64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.head        = nn.Conv2d(64, num_kp, 1)
        self.soft_argmax = SoftArgmax2D()

    def forward(self, radar):
        hori_feat = self.hori_enc(radar[:, 0:1, :, :])
        vert_feat = self.vert_enc(radar[:, 1:2, :, :])

        # Align spatial sizes before concat
        vert_feat = F.interpolate(vert_feat, size=hori_feat.shape[2:],
                                  mode='bilinear', align_corners=False)

        fused = torch.cat([hori_feat, vert_feat], dim=1)
        fused = self.fusion(fused)
        fused = self.bottleneck(fused)

        d2 = self.dec2(fused)
        d1 = self.dec1(d2)

        heatmaps = self.head(d1)
        coords   = self.soft_argmax(heatmaps)
        return heatmaps, coords
