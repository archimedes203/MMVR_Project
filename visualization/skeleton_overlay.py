"""
Skeleton overlay visualisations — predicted vs GT.
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt
import torch

from config import cfg


@torch.no_grad()
def visualise_predictions(model, loader, device, model_name,
                          n_samples=6, save_dir=None):
    """
    Show predicted skeleton (red) vs GT skeleton (green)
    on a blank canvas for n_samples from loader.
    """
    save_dir = save_dir or cfg.RESULTS_DIR
    model.eval()
    batch   = next(iter(loader))
    radar   = batch['radar'].to(device)
    gt_kp   = batch['coords'].numpy()    # (B, 17, 2)
    vis     = batch['vis'].numpy()       # (B, 17)

    # _, pred_kp          = model(radar)   # original (no log_var)
    _, pred_kp, _log_var = model(radar)
    pred_kp = pred_kp.cpu().numpy()

    n    = min(n_samples, radar.shape[0])
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = np.array(axes).flatten()

    for i in range(n):
        canvas = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)
        ax = axes[i]
        ax.imshow(canvas)
        # GT (green)
        for a, b in cfg.SKELETON:
            if vis[i,a] > 0.1 and vis[i,b] > 0.1:
                ax.plot([gt_kp[i,a,0]*cfg.IMG_W, gt_kp[i,b,0]*cfg.IMG_W],
                        [gt_kp[i,a,1]*cfg.IMG_H, gt_kp[i,b,1]*cfg.IMG_H],
                        '-o', color='lime', lw=1.5, ms=3)
        # Prediction (red)
        for a, b in cfg.SKELETON:
            if vis[i,a] > 0.1 and vis[i,b] > 0.1:
                ax.plot([pred_kp[i,a,0]*cfg.IMG_W, pred_kp[i,b,0]*cfg.IMG_W],
                        [pred_kp[i,a,1]*cfg.IMG_H, pred_kp[i,b,1]*cfg.IMG_H],
                        '-o', color='red', lw=1.5, ms=3, alpha=0.7)
        ax.set_xlim(0, cfg.IMG_W); ax.set_ylim(cfg.IMG_H, 0)
        ax.set_title(f'Sample {i+1}', fontsize=9)
        ax.axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    handles = [plt.Line2D([0],[0],color='lime',lw=2,label='GT'),
               plt.Line2D([0],[0],color='red', lw=2,label='Pred')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10)
    plt.suptitle(f'Skeleton Predictions — {model_name}', fontsize=13)
    plt.tight_layout(rect=[0,0.04,1,1])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'predictions_{model_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


@torch.no_grad()
def visualise_predictions_with_mask(model, loader, device, model_name,
                                     n_samples=6, save_dir=None):
    """
    Side-by-side GT vs prediction panels.
    coords[:,0] = x/col (width axis) → matplotlib X
    coords[:,1] = y/row (height axis) → matplotlib Y
    """
    save_dir = save_dir or cfg.RESULTS_DIR
    model.eval()
    batch   = next(iter(loader))
    n       = min(n_samples, batch['radar'].shape[0])
    radar   = batch['radar'][:n].to(device)
    gt_kp   = batch['coords'][:n].numpy()   # (n, 17, 2) [x_norm, y_norm]
    vis     = batch['vis'][:n].numpy()

    # _, pred_kp          = model(radar)   # original (no log_var)
    _, pred_kp, _log_var = model(radar)
    pred_kp = pred_kp.cpu().numpy()

    ds      = loader.dataset
    cols    = min(n, 3)
    rows    = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols*2, figsize=(5*cols*2, 5*rows))
    axes    = np.array(axes).reshape(rows, cols*2)

    bone_colours = ['#ff6b6b','#ffd93d','#6bcb77','#4d96ff',
                    '#ff922b','#cc5de8','#20c997','#f06595',
                    '#74c0fc','#a9e34b','#ffa94d','#da77f2']

    def draw(ax, kps, vis_arr, title, sample_idx, label_names=False):
        try:
            meta      = ds.samples[sample_idx]
            mask_path = meta['pose_path'].replace('_pose.npz','_mask.npz')
            masks     = np.load(mask_path)['mask']
            canvas    = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)
            for pi, m in enumerate(masks):
                c = [(0,180,0),(0,120,255),(255,80,0)][pi%3]
                for ch, cv in enumerate(c): canvas[:,:,ch][m] = cv
        except Exception:
            canvas = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)

        ax.imshow(canvas)
        # Draw bones — x=col, y=row
        for bi, (a, b) in enumerate(cfg.SKELETON):
            if vis_arr[a] > 0.1 and vis_arr[b] > 0.1:
                ax.plot(
                    [kps[a,0]*cfg.IMG_W, kps[b,0]*cfg.IMG_W],  # x=col
                    [kps[a,1]*cfg.IMG_H, kps[b,1]*cfg.IMG_H],  # y=row
                    '-', color=bone_colours[bi % len(bone_colours)], lw=2, zorder=2
                )
        # Draw keypoints
        for ki in range(17):
            if vis_arr[ki] > 0.1:
                cx = kps[ki,0] * cfg.IMG_W   # x=col → horizontal
                cy = kps[ki,1] * cfg.IMG_H   # y=row → vertical
                ax.plot(cx, cy, 'o', color='white', ms=5,
                        markeredgecolor='black', zorder=3)
                if label_names:
                    ax.text(cx+4, cy-4, cfg.KP_NAMES[ki],
                            fontsize=5, color='yellow',
                            bbox=dict(boxstyle='round,pad=0.1',
                                      fc='black', alpha=0.5))
        ax.set_xlim(0, cfg.IMG_W); ax.set_ylim(cfg.IMG_H, 0)  # row 0=top
        ax.set_title(title, fontsize=8); ax.axis('off')

    for i in range(n):
        r   = i // cols
        c   = (i %  cols) * 2
        draw(axes[r, c],   gt_kp[i],   vis[i],
             f'Sample {i+1} — GT', i, label_names=True)
        draw(axes[r, c+1], pred_kp[i], vis[i],
             f'Sample {i+1} — {model_name}', i)

    for i in range(n, rows*cols):
        r = i // cols
        axes[r,(i%cols)*2  ].axis('off')
        axes[r,(i%cols)*2+1].axis('off')

    plt.suptitle(f'GT vs Predicted Poses — {model_name}', fontsize=11)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'predictions_{model_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/predictions_{model_name}.png")
