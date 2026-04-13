"""
ResNet-18 baseline adapted for radar-based pose estimation.
"""

import torch.nn as nn
from torchvision.models import resnet18

from models.blocks import SoftArgmax2D


class ResNet18PoseModel(nn.Module):
    """
    ResNet-18 adapted for radar-based pose estimation.
    First conv replaced to accept 2-channel radar input.

    Input:  radar (B, 2, RADAR_H, RADAR_W)
    Output: heatmaps (B, 17, H', W'), coords (B, 17, 2)
    """

    def __init__(self, num_kp=17, in_channels=2):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3   # → 256ch
        )
        # Bilinear upsample decoder — no skip connections
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64,  3, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )
        self.head        = nn.Conv2d(64, num_kp, 1)
        self.soft_argmax = SoftArgmax2D()

    def forward(self, x):
        feats    = self.encoder(x)
        feats    = self.decoder(feats)
        heatmaps = self.head(feats)
        coords   = self.soft_argmax(heatmaps)
        return heatmaps, coords
