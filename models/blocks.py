"""
Reusable building blocks for the pose estimation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → Add → ReLU with optional shortcut."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x)).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention — highlights informative spatial regions."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))


class SoftArgmax2D(nn.Module):
    """
    Differentiable Soft-Argmax.
    Converts heatmaps (B, K, H, W) → coordinates (B, K, 2) in [0, 1].
    A small epsilon is added before softmax to prevent all-zero heatmaps
    from producing nan gradients.
    """
    def forward(self, heatmaps):
        B, K, H, W = heatmaps.shape
        # Clamp to prevent extreme values before softmax
        heatmaps = torch.clamp(heatmaps, -88.0, 88.0)
        flat     = heatmaps.view(B, K, -1)
        # Shift by max for numerical stability (log-sum-exp trick)
        flat     = flat - flat.max(dim=-1, keepdim=True)[0]
        probs    = F.softmax(flat, dim=-1).view(B, K, H, W)

        xs = torch.linspace(0, 1, W, device=heatmaps.device)
        ys = torch.linspace(0, 1, H, device=heatmaps.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')

        pred_x = (probs * grid_x).sum(dim=[2, 3])
        pred_y = (probs * grid_y).sum(dim=[2, 3])
        return torch.stack([pred_x, pred_y], dim=-1)
