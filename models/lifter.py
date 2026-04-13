"""
PoseLiftingMLP — lifts 2D keypoint coordinates to 3D using radar depth cues.
"""

import torch
import torch.nn as nn


class PoseLiftingMLP(nn.Module):
    """
    Simple MLP to lift 2D keypoint coordinates + radar depth cue to 3D.

    Input:  2D coords (B, 17, 2)  concatenated with radar depth feature
            (B, radar_feat_dim) broadcast to each keypoint.
    Output: 3D coords (B, 17, 3)  [x, y, z] normalised.

    In practice the radar depth feature is the mean activation of the
    radar heatmap per-channel, giving a coarse depth estimate.
    """

    def __init__(self, num_kp=17, radar_feat_dim=2):
        super().__init__()
        in_dim = num_kp * 2 + radar_feat_dim   # flattened 2D kps + radar cue

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, 512),    nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, 256),    nn.ReLU(inplace=True),
            nn.Linear(256, num_kp * 3)   # predict 3D (x, y, z)
        )
        self.num_kp = num_kp

    def forward(self, coords_2d, radar):
        """
        Args:
            coords_2d : (B, 17, 2) — normalised 2D predictions
            radar     : (B, 2, H, W) — radar heatmaps
        Returns:
            coords_3d : (B, 17, 3)
        """
        B = coords_2d.shape[0]
        flat_2d   = coords_2d.view(B, -1)       # (B, 34)
        depth_cue = radar.mean(dim=[2, 3])       # (B, 2)
        x   = torch.cat([flat_2d, depth_cue], dim=1)   # (B, 36)
        out = self.mlp(x)                               # (B, num_kp*3)
        return out.view(B, self.num_kp, 3)              # (B, 17, 3)
