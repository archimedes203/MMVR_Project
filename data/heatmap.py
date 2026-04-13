"""
Gaussian heatmap generation for ground-truth keypoints.
"""

import numpy as np
from config import cfg


def generate_gaussian_heatmap(heatmap_size, keypoints, sigma=2):
    """
    Generate 2D Gaussian heatmaps for ground-truth keypoints.

    keypoints: (17, 3) — [x=col, y=row, visibility] in absolute image pixels
    Returns:   (17, H, W) float32 heatmaps
    """
    H = W = heatmap_size
    heatmaps = np.zeros((17, H, W), dtype=np.float32)

    for k, (x, y, v) in enumerate(keypoints):
        if v < 0.1:          # skip low-confidence keypoints
            continue
        # x=col → maps to heatmap width axis
        # y=row → maps to heatmap height axis
        cx = int(x / cfg.IMG_W * W)
        cy = int(y / cfg.IMG_H * H)
        cx = np.clip(cx, 0, W-1)
        cy = np.clip(cy, 0, H-1)

        xs = np.arange(W)
        ys = np.arange(H)
        xx, yy = np.meshgrid(xs, ys)
        g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        heatmaps[k] = np.maximum(heatmaps[k], g)

    return heatmaps
