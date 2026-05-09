#!/usr/bin/env python3
"""
Hyperparameter tuning via random search over LR, WEIGHT_DECAY, lambda_hm, lambda_coord.

Usage:
    python tune.py                        # random search, 30 combos, 15 epochs, all models
    python tune.py --n-samples 50         # try 50 random combos per model
    python tune.py --epochs 20            # use 20 epochs per trial
    python tune.py --model resnet18       # tune one model only
    python tune.py --grid                 # full grid search (240 combos — slow!)
    python tune.py --plot-only            # regenerate plots from existing CSV, skip training

Results are saved incrementally to ./results/tuning_results.csv so the run can
be interrupted and resumed safely.

After tuning, retrain the winning config with full epochs:
    python main.py train  (after updating config.py with the best values)
"""

import argparse
import copy
import csv
import gc
import itertools
import os
import random
import time
import warnings

import torch

warnings.filterwarnings('ignore', category=UserWarning)

from config import cfg, DEVICE
from data.splits import load_mmvr_samples_split
from data.loader import create_dataloaders_from_splits
from models.custom_cnn import CustomCNN
from models.resnet_baseline import ResNet18PoseModel
from models.fusion import FusionModel
from training.loss import PoseLoss
from training.train import run_training

# ── Parameter grid ───────────────────────────────────────────────────────────
PARAM_GRID = {
    'lr':           [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'lambda_hm':    [1.0, 2.0, 5.0, 10.0],
    'lambda_coord': [0.5, 1.0, 2.0],
}

MODEL_SPECS = {
    'custom_cnn': lambda: CustomCNN(num_kp=cfg.NUM_KEYPOINTS, in_channels=2),
    'resnet18':   lambda: ResNet18PoseModel(num_kp=cfg.NUM_KEYPOINTS, in_channels=2),
    'fusion':     lambda: FusionModel(num_kp=cfg.NUM_KEYPOINTS),
}

RESULTS_CSV = './results/tuning_results.csv'
CSV_FIELDS  = ['model', 'lr', 'weight_decay', 'lambda_hm', 'lambda_coord',
               'best_val_loss', 'final_val_loss', 'epochs', 'duration_s']
TUNING_CKPT_DIR = './checkpoints/tuning'


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning via random search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--n-samples', type=int, default=30,
                        help='Random combos to try per model (default: 30)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Training epochs per trial (default: 15)')
    parser.add_argument('--model', type=str, default='all',
                        choices=list(MODEL_SPECS) + ['all'],
                        help='Which model(s) to tune (default: all)')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for random sampling (default: 0)')
    parser.add_argument('--grid', action='store_true',
                        help='Full grid search instead of random (240 combos — slow!)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training; regenerate plots from existing CSV')
    return parser.parse_args()


# ── Combo generation ─────────────────────────────────────────────────────────

def build_combos(n_samples, seed, grid):
    all_combos = [
        dict(zip(PARAM_GRID.keys(), vals))
        for vals in itertools.product(*PARAM_GRID.values())
    ]
    if grid:
        return all_combos
    rng = random.Random(seed)
    return rng.sample(all_combos, min(n_samples, len(all_combos)))


# ── CSV helpers ──────────────────────────────────────────────────────────────

def load_completed(csv_path):
    """Return set of (model, lr, wd, lhm, lc) string-tuples already in the CSV."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((row['model'], row['lr'], row['weight_decay'],
                      row['lambda_hm'], row['lambda_coord']))
    return done


def append_result(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Trial ────────────────────────────────────────────────────────────────────

def make_trial_cfg(lr, weight_decay, n_epochs):
    """Shallow copy of global cfg with per-trial overrides as instance attrs."""
    trial_cfg = copy.copy(cfg)
    trial_cfg.LR             = lr
    trial_cfg.WEIGHT_DECAY   = weight_decay
    trial_cfg.NUM_EPOCHS     = n_epochs
    trial_cfg.CHECKPOINT_DIR = TUNING_CKPT_DIR
    return trial_cfg


def run_trial(model_fn, model_name, combo, n_epochs, train_loader, val_loader,
              run_id, total_runs):
    lr           = combo['lr']
    weight_decay = combo['weight_decay']
    lambda_hm    = combo['lambda_hm']
    lambda_coord = combo['lambda_coord']

    print(f"\n[{run_id}/{total_runs}] {model_name} | "
          f"lr={lr:.0e}  wd={weight_decay:.0e}  "
          f"lambda_hm={lambda_hm}  lambda_coord={lambda_coord}")

    trial_name = (f"{model_name}__lr{lr:.0e}__wd{weight_decay:.0e}"
                  f"__lhm{lambda_hm}__lc{lambda_coord}")
    model      = model_fn().to(DEVICE)
    criterion  = PoseLoss(lambda_hm=lambda_hm, lambda_coord=lambda_coord)
    trial_cfg  = make_trial_cfg(lr, weight_decay, n_epochs)

    t0 = time.time()
    try:
        history   = run_training(model, trial_name, train_loader, val_loader,
                                 trial_cfg, criterion,
                                 is_fusion=(model_name == 'fusion'))
        best_val  = min(history['val_loss'])
        final_val = history['val_loss'][-1]
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  [ERROR] Trial failed: {e}")
        best_val = final_val = float('nan')
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_val, final_val, round(time.time() - t0, 1)


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(csv_path, model_names):
    if not os.path.exists(csv_path):
        print("No results file found.")
        return

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    print(f"\n{'='*70}")
    print(" TUNING SUMMARY — best val loss per model")
    print(f"{'='*70}")

    for mname in model_names:
        valid = [r for r in rows
                 if r['model'] == mname and r['best_val_loss'] not in ('', 'nan')]
        if not valid:
            print(f"\n{mname}: no completed trials")
            continue
        best = min(valid, key=lambda r: float(r['best_val_loss']))
        print(f"\n{mname}")
        print(f"  best_val_loss : {float(best['best_val_loss']):.4f}")
        print(f"  lr            : {best['lr']}")
        print(f"  weight_decay  : {best['weight_decay']}")
        print(f"  lambda_hm     : {best['lambda_hm']}")
        print(f"  lambda_coord  : {best['lambda_coord']}")

    print(f"\nFull results → {csv_path}")


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_results(csv_path, model_names, out_dir='./results'):
    """Save three tuning report plots from the results CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    if not os.path.exists(csv_path):
        print("No results CSV found; skipping plots.")
        return

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if r['best_val_loss'] not in ('', 'nan')]
    if not valid:
        print("No valid results to plot.")
        return

    for r in valid:
        r['best_val_loss']  = float(r['best_val_loss'])
        r['lr']             = float(r['lr'])
        r['weight_decay']   = float(r['weight_decay'])
        r['lambda_hm']      = float(r['lambda_hm'])
        r['lambda_coord']   = float(r['lambda_coord'])

    os.makedirs(out_dir, exist_ok=True)
    for style in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid'):
        try:
            plt.style.use(style)
            break
        except OSError:
            pass

    MODEL_COLORS = {'custom_cnn': '#4C72B0', 'resnet18': '#DD8452', 'fusion': '#55A868'}

    # ── 1. Best val loss per model ────────────────────────────────────────────
    model_bests = {}
    for mname in model_names:
        m_rows = [r for r in valid if r['model'] == mname]
        if m_rows:
            model_bests[mname] = min(r['best_val_loss'] for r in m_rows)

    if model_bests:
        fig, ax = plt.subplots(figsize=(max(4, len(model_bests) * 1.8), 4))
        names  = list(model_bests)
        values = [model_bests[n] for n in names]
        colors = [MODEL_COLORS.get(n, '#888') for n in names]
        bars   = ax.bar(names, values, color=colors, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Model')
        ax.set_ylabel('Best Validation Loss')
        ax.set_title('Best Validation Loss by Model')
        ax.set_ylim(0, max(values) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'tuning_model_comparison.png'), dpi=150)
        plt.close()

    # ── 2. Hyperparameter sensitivity (2 × 2 box plots) ──────────────────────
    hp_keys   = ['lr', 'weight_decay', 'lambda_hm', 'lambda_coord']
    hp_labels = ['Learning Rate', 'Weight Decay', r'$\lambda_{hm}$', r'$\lambda_{coord}$']
    fmt_sci   = {'lr', 'weight_decay'}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, hp, label in zip(axes.flatten(), hp_keys, hp_labels):
        unique_vals = sorted(set(r[hp] for r in valid))
        data        = [[r['best_val_loss'] for r in valid if r[hp] == v]
                       for v in unique_vals]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor('#4C72B0')
            patch.set_alpha(0.55)
        tick_labels = ([f'{v:.0e}' for v in unique_vals] if hp in fmt_sci
                       else [str(v) for v in unique_vals])
        ax.set_xticklabels(tick_labels, rotation=20, ha='right')
        ax.set_xlabel(label)
        ax.set_ylabel('Best Validation Loss')
        ax.set_title(f'Effect of {label}')

    fig.suptitle('Hyperparameter Sensitivity', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tuning_hp_sensitivity.png'), dpi=150)
    plt.close()

    # ── 3. Top-15 trials horizontal bar chart ────────────────────────────────
    top = sorted(valid, key=lambda r: r['best_val_loss'])[:15]
    labels = [
        f"{r['model']}  lr={r['lr']:.0e}  wd={r['weight_decay']:.0e}"
        f"  λhm={r['lambda_hm']}  λc={r['lambda_coord']}"
        for r in top
    ]
    vals       = [r['best_val_loss'] for r in top]
    bar_colors = [MODEL_COLORS.get(r['model'], '#888') for r in top]

    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.45 + 1)))
    ax.barh(range(len(vals)), vals, color=bar_colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('Best Validation Loss')
    ax.set_title('Top 15 Trials by Best Validation Loss')

    present_models = sorted({r['model'] for r in top})
    legend_handles = [Patch(facecolor=MODEL_COLORS.get(m, '#888'), label=m)
                      for m in present_models]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tuning_top_trials.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to {out_dir}/")
    for name in ('tuning_model_comparison.png',
                 'tuning_hp_sensitivity.png',
                 'tuning_top_trials.png'):
        print(f"  {name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs('./results', exist_ok=True)
    os.makedirs(TUNING_CKPT_DIR, exist_ok=True)

    models_to_tune = (list(MODEL_SPECS.items()) if args.model == 'all'
                      else [(args.model, MODEL_SPECS[args.model])])
    model_names    = [m for m, _ in models_to_tune]

    if args.plot_only:
        print_summary(RESULTS_CSV, model_names)
        plot_results(RESULTS_CSV, model_names)
        return

    print("Loading data...")
    train_samples, val_samples, test_samples = load_mmvr_samples_split(
        cfg.DATA_ROOT, cfg.SPLIT_FILE, cfg.PROTOCOL)
    train_loader, val_loader, _, _, _, _ = create_dataloaders_from_splits(
        train_samples, val_samples, test_samples, cfg)

    combos     = build_combos(args.n_samples, args.seed, args.grid)
    total_runs = len(models_to_tune) * len(combos)
    done       = load_completed(RESULTS_CSV)

    print(f"\nSearch       : {'grid' if args.grid else 'random'}")
    print(f"Combos/model : {len(combos)}")
    print(f"Epochs/trial : {args.epochs}")
    print(f"Models       : {model_names}")
    print(f"Total runs   : {total_runs}  "
          f"({len(done)} already completed, will skip)")

    run_id = 0
    try:
        for model_name, model_fn in models_to_tune:
            for combo in combos:
                run_id += 1
                key = (model_name,
                       str(combo['lr']), str(combo['weight_decay']),
                       str(combo['lambda_hm']), str(combo['lambda_coord']))
                if key in done:
                    print(f"[{run_id}/{total_runs}] Skip (done): {key}")
                    continue

                best_val, final_val, duration = run_trial(
                    model_fn, model_name, combo, args.epochs,
                    train_loader, val_loader, run_id, total_runs)

                append_result(RESULTS_CSV, {
                    'model':         model_name,
                    'lr':            combo['lr'],
                    'weight_decay':  combo['weight_decay'],
                    'lambda_hm':     combo['lambda_hm'],
                    'lambda_coord':  combo['lambda_coord'],
                    'best_val_loss': best_val,
                    'final_val_loss':final_val,
                    'epochs':        args.epochs,
                    'duration_s':    duration,
                })
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving summary of completed trials...")

    print_summary(RESULTS_CSV, model_names)
    plot_results(RESULTS_CSV, model_names)


if __name__ == '__main__':
    main()
