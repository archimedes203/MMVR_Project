"""
3D pose skeleton visualisation.
"""

import os
import matplotlib.pyplot as plt
import torch

from config import cfg


@torch.no_grad()
def visualise_3d_pose(fusion_model, lifter, loader, device,
                      n=2, save_dir=None):
    """Reconstruct and plot 3D skeleton from fusion model + PoseLiftingMLP."""
    save_dir = save_dir or cfg.RESULTS_DIR
    if loader is None:
        return
    fusion_model.eval(); lifter.eval()
    batch = next(iter(loader))
    radar = batch['radar'].to(device)

    _, pred_2d  = fusion_model(radar)         # (B, 17, 2)
    pred_3d     = lifter(pred_2d, radar).cpu().numpy()  # (B, 17, 3)

    fig = plt.figure(figsize=(5*n, 5))
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1, projection='3d')
        kp = pred_3d[i]   # (17, 3)

        ax.scatter(kp[:,0], kp[:,1], kp[:,2], c='red', s=30)
        for a, b in cfg.SKELETON:
            ax.plot([kp[a,0],kp[b,0]], [kp[a,1],kp[b,1]],
                    [kp[a,2],kp[b,2]], 'b-', lw=1.5)

        ax.set_title(f'3D Pose {i+1}', fontsize=10)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.suptitle('3D Pose Reconstruction', fontsize=13)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '3d_pose.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
