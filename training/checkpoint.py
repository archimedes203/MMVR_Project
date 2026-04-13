"""
Checkpoint save/load and model export utilities.
"""

import os
import json
import shutil

import numpy as np
import torch

from config import cfg, DEVICE


def load_checkpoint(model, model_name):
    """
    Load the best saved checkpoint for a model from the checkpoints directory.
    Safe to call even if no checkpoint exists yet — prints a warning and
    returns the model with its current (random) weights in that case.
    """
    path = os.path.join(cfg.CHECKPOINT_DIR, f'{model_name}_best.pth')
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded {path}  "
              f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    else:
        print(f"[WARNING] No checkpoint at {path} — using random weights.")
    return model


def save_histories(histories, path='./results/histories.json'):
    """Save training histories dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(histories, f, indent=2)
    print(f"Histories saved → {path}")
    for mname, h in histories.items():
        n = len(h.get('train_loss', []))
        best_val = min(h.get('val_loss', [float('inf')]))
        print(f"  {mname}: {n} epochs, best val loss = {best_val:.4f}")


def load_histories(path='./results/histories.json'):
    """Load training histories from JSON."""
    if not os.path.exists(path):
        print(f"[WARNING] No saved histories at '{path}'. Run training first.")
        return {}
    with open(path) as f:
        h = json.load(f)
    print(f"Histories loaded from {path}")
    for mname, hist in h.items():
        n        = len(hist.get('train_loss', []))
        best_val = min(hist.get('val_loss', [float('inf')]))
        print(f"  {mname}: {n} epochs, best val loss = {best_val:.4f}")
    return h


def save_eval_results(eval_results, path='./results/eval_results.json'):
    """Save evaluation results dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert numpy arrays to lists for JSON serialisation
    serialisable = {}
    for mname, r in eval_results.items():
        serialisable[mname] = {
            k: (v.tolist() if hasattr(v, 'tolist') else v)
            for k, v in r.items()
        }
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"Eval results saved → {path}")
    for mname, r in eval_results.items():
        print(f"  {mname}: PCK={r.get('PCK@0.05',0)*100:.2f}%  "
              f"OKS={r.get('OKS',0):.4f}  MAE={r.get('MAE_px',0):.1f}px")


def load_eval_results(path='./results/eval_results.json'):
    """Load evaluation results from JSON."""
    if not os.path.exists(path):
        print(f"[WARNING] No saved eval_results at '{path}'. Run evaluation first.")
        return {}
    with open(path) as f:
        raw = json.load(f)
    for mname in raw:
        if 'pck_per_kp' in raw[mname]:
            raw[mname]['pck_per_kp'] = np.array(raw[mname]['pck_per_kp'])
    print(f"Eval results loaded from {path}")
    for mname, r in raw.items():
        print(f"  {mname}: PCK={r.get('PCK@0.05',0)*100:.2f}%  "
              f"OKS={r.get('OKS',0):.4f}  MAE={r.get('MAE_px',0):.1f}px")
    return raw


def check_checkpoints():
    """Print status of all model checkpoints."""
    print("Checkpoint status:")
    for mname in ['custom_cnn', 'resnet18', 'fusion']:
        path = os.path.join(cfg.CHECKPOINT_DIR, f'{mname}_best.pth')
        if os.path.exists(path):
            ckpt    = torch.load(path, map_location='cpu')
            size_mb = os.path.getsize(path) / 1024**2
            print(f"  {mname}: epoch {ckpt['epoch']}, "
                  f"val_loss={ckpt['val_loss']:.4f}, "
                  f"size={size_mb:.1f} MB  ✓")
        else:
            print(f"  {mname}: NOT FOUND at {path}  ✗")


def export_final_models(cfg):
    """Copy best checkpoints into results dir for easy submission."""
    for mname in ['custom_cnn', 'resnet18', 'fusion']:
        src = os.path.join(cfg.CHECKPOINT_DIR, f'{mname}_best.pth')
        dst = os.path.join(cfg.RESULTS_DIR,    f'{mname}_final.pth')
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  Exported: {dst}")
        else:
            print(f"  [SKIP] {src} not found")
