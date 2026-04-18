"""
Custom CNN encoder-decoder for keypoint heatmap estimation from radar.
"""

import torch.nn as nn

from models.blocks import ResidualBlock, CBAM, SoftArgmax2D


class CustomCNN(nn.Module):
    """
    Custom CNN encoder-decoder for keypoint heatmap estimation from radar.

    Input:  radar Tensor (B, 2, RADAR_H, RADAR_W)
    Output: (heatmaps (B, 17, H', W'), coords (B, 17, 2))

    Skip connections are omitted to avoid size mismatches from the
    non-square radar input (256x128). The decoder uses bilinear upsampling
    followed by convolutions to recover spatial detail instead.
    """

    def __init__(self, num_kp=17, in_channels=2):
        super().__init__()
        self.num_kp = num_kp

        # ── Encoder ─────────────────────────────────────────────────
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.enc1 = nn.Sequential(ResidualBlock(32,  64,  stride=2), CBAM(64))
        self.enc2 = nn.Sequential(ResidualBlock(64,  128, stride=2), CBAM(128))
        self.enc3 = nn.Sequential(ResidualBlock(128, 256, stride=2), CBAM(256))

        # ── Decoder (bilinear upsample + conv, no skip additions) ───
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

        # ── Output head ─────────────────────────────────────────────
        self.head        = nn.Conv2d(64, num_kp, 1)
        self.soft_argmax = SoftArgmax2D()
        # Log-variance head for Gaussian NLL coordinate loss (B, 17, 2)
        self.log_var_head = nn.Conv2d(64, num_kp * 2, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(e3)
        d1 = self.dec1(d2)
        heatmaps = self.head(d1)
        coords   = self.soft_argmax(heatmaps)
        # Log-variance head: pool spatially then reshape to (B, 17, 2)
        log_var  = self.log_var_head(d1).flatten(2).mean(dim=2).view(-1, self.num_kp, 2)
        # return heatmaps, coords               # original (no log_var)
        return heatmaps, coords, log_var
