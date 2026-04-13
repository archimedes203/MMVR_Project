"""
Adverse condition robustness plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import cfg


def plot_adverse_robustness(eval_results, adverse_results,
                            save_dir=None):
    """Bar chart: Fusion model PCK under normal vs adverse radar conditions."""
    save_dir = save_dir or cfg.RESULTS_DIR
    if not eval_results or not adverse_results:
        print("No data."); return

    labels     = ['Normal', 'Noise', 'Dropout', 'Low Power']
    conditions = ['noise', 'dropout', 'low_power']
    fusion_pck = [eval_results.get('fusion', {}).get('PCK@0.05', 0)]
    for cond in conditions:
        fusion_pck.append(adverse_results.get(cond, {}).get('PCK@0.05', 0))

    rn_normal = eval_results.get('resnet18', {}).get('PCK@0.05', 0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, fusion_pck, 0.5, color='#2a9d8f', label='Fusion Model', alpha=0.85)
    ax.axhline(rn_normal, color='#457b9d', linestyle='--', lw=2,
               label=f'ResNet-18 (normal) = {rn_normal:.3f}')
    for b, v in zip(bars, fusion_pck):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{v:.3f}', ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('PCK@0.05'); ax.set_ylim(0, 1.1)
    ax.set_title('Fusion Model Robustness under Adverse Radar Conditions', fontsize=13)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'adverse_robustness.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
