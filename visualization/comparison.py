"""
Model comparison bar charts and per-keypoint PCK heatmaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import cfg


def plot_model_comparison(eval_results, save_dir=None):
    """Bar chart comparing all models across all metrics."""
    save_dir = save_dir or cfg.RESULTS_DIR
    if not eval_results:
        print("No evaluation results to plot."); return

    metrics = ['PCK@0.05', 'OKS', 'F1', 'Precision']
    models  = list(eval_results.keys())
    n_m, n_metrics = len(models), len(metrics)

    x   = np.arange(n_metrics)
    w   = 0.25
    colours = ['#e63946', '#457b9d', '#2a9d8f']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (mname, c) in enumerate(zip(models, colours)):
        vals = [eval_results[mname][m] for m in metrics]
        bars = ax.bar(x + i*w - w, vals, w, label=mname, color=c, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.1)
    ax.set_title('Model Comparison — All Metrics', fontsize=14)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_pck_per_keypoint(eval_results, save_dir=None):
    """Heatmap of per-keypoint PCK for each model."""
    save_dir = save_dir or cfg.RESULTS_DIR
    if not eval_results:
        print("No data."); return

    matrix = np.array([r['pck_per_kp'] for r in eval_results.values()])  # (M, 17)
    fig, ax = plt.subplots(figsize=(16, 3))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGn',
                xticklabels=cfg.KP_NAMES,
                yticklabels=list(eval_results.keys()),
                vmin=0, vmax=1, ax=ax, linewidths=0.4)
    ax.set_title('Per-Keypoint PCK@0.05 by Model', fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'pck_per_keypoint.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
