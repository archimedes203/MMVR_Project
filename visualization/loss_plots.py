"""
Training & validation loss curve plots.
"""

import os
import matplotlib.pyplot as plt

from config import cfg


def plot_loss_curves(histories, save_dir=None):
    """Plot train/val loss curves for each model."""
    save_dir = save_dir or cfg.RESULTS_DIR

    if not histories:
        print("histories is empty — has training been run?")
        return

    has_data = any(len(h.get('train_loss', [])) > 0 for h in histories.values())
    if not has_data:
        print("All loss lists are empty — training did not complete an epoch.")
        return

    colours = {'custom_cnn': '#e63946', 'resnet18': '#457b9d', 'fusion': '#2a9d8f'}
    panel_pairs = [
        ('Total Loss',      'train_loss',  'val_loss'),
        ('Coordinate Loss', 'train_coord', 'val_coord'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (title, tr_key, va_key) in zip(axes, panel_pairs):
        for mname, hist in histories.items():
            if not hist.get(tr_key):
                continue
            c       = colours.get(mname, 'grey')
            ep      = list(range(1, len(hist[tr_key]) + 1))
            tr_vals = hist[tr_key]
            va_vals = hist[va_key]

            ax.plot(ep, tr_vals, 'o-', color=c, linestyle='-',
                    label=f'{mname} train', markersize=5)
            ax.plot(ep, va_vals, 's--', color=c, linestyle='--',
                    label=f'{mname} val', markersize=5, alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss',  fontsize=11)
        ax.set_title(title,    fontsize=13)
        ax.set_xticks(range(1, max(
            len(h.get(tr_key, [])) for h in histories.values()) + 1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Training & Validation Loss Curves', fontsize=14)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/loss_curves.png")
