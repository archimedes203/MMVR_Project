"""
Raw sample visualisation and skeleton drawing utilities.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import cfg


def draw_skeleton(ax, coords, vis, kp_colour, bone_colour, label_kps=False):
    """
    Draw skeleton bonds and keypoints on a matplotlib Axes.

    coords: (17, 2) — [col_norm, row_norm] in [0,1]
    vis:    (17,)   — visibility scores
    """
    for a, b in cfg.SKELETON:
        if vis[a] > 0.1 and vis[b] > 0.1:
            ax.plot(
                [coords[a, 0] * cfg.IMG_W, coords[b, 0] * cfg.IMG_W],
                [coords[a, 1] * cfg.IMG_H, coords[b, 1] * cfg.IMG_H],
                '-', color=bone_colour, lw=2, zorder=2
            )
    for ki in range(17):
        if vis[ki] > 0.1:
            cx = coords[ki, 0] * cfg.IMG_W
            cy = coords[ki, 1] * cfg.IMG_H
            ax.plot(cx, cy, 'o', color=kp_colour, ms=5,
                    markeredgecolor='black', zorder=3)
            if label_kps:
                ax.text(cx + 4, cy - 4, cfg.KP_NAMES[ki],
                        fontsize=5, color='yellow',
                        bbox=dict(boxstyle='round,pad=0.1',
                                  fc='black', alpha=0.5))


def visualise_sample(sample, mask_path=None):
    """Visualise a single dataset sample (radar heatmaps + keypoint overlay)."""
    radar   = sample['radar'].numpy()
    coords  = sample['coords'].numpy()   # (17, 2) [col_norm, row_norm]
    vis     = sample['vis'].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Horizontal radar
    axes[0].imshow(radar[0], cmap='plasma', aspect='auto')
    axes[0].set_title('Horizontal Radar Heatmap')
    axes[0].axis('off')

    # Vertical radar
    axes[1].imshow(radar[1], cmap='plasma', aspect='auto')
    axes[1].set_title('Vertical Radar Heatmap')
    axes[1].axis('off')

    # Skeleton overlay on blank canvas
    canvas = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)
    axes[2].imshow(canvas)
    draw_skeleton(axes[2], coords, vis, 'white', 'lime', label_kps=True)
    axes[2].set_xlim(0, cfg.IMG_W)
    axes[2].set_ylim(cfg.IMG_H, 0)
    axes[2].set_title('Keypoint Skeleton')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
