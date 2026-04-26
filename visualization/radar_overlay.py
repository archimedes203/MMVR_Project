"""
Radar heatmap overlay visualisation.
"""

import os
import matplotlib.pyplot as plt

from config import cfg


def visualise_radar_overlay(loader, n=4, save_dir=None):
    """Show hori and vert radar heatmaps side by side with keypoints overlaid."""
    save_dir = save_dir or cfg.RESULTS_DIR
    batch   = next(iter(loader))
    radar_t = batch['radar']
    coords  = batch['coords'].numpy()
    vis     = batch['vis'].numpy()

    fig, axes = plt.subplots(n, 2, figsize=(10, 4*n))
    for i in range(n):
        hori = radar_t[i, 0].numpy()
        vert = radar_t[i, 1].numpy()

        axes[i,0].imshow(hori, cmap='plasma', aspect='auto')
        axes[i,0].set_title(f'Sample {i+1} — Horizontal heatmap', fontsize=9)
        axes[i,0].axis('off')

        axes[i,1].imshow(vert, cmap='plasma', aspect='auto')
        # overlay keypoints scaled to radar space
        for ki in range(17):
            if vis[i,ki] > 0.1:
                rx = coords[i,ki,0] * cfg.RADAR_W
                ry = coords[i,ki,1] * cfg.RADAR_H
                axes[i,1].plot(rx, ry, 'o', color='lime', ms=4)
        axes[i,1].set_title(f'Sample {i+1} — Vertical heatmap + KP', fontsize=9)
        axes[i,1].axis('off')

    plt.suptitle('Radar Heatmaps with Keypoint Overlay', fontsize=13)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'radar_overlay.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
