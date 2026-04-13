"""
Training and validation loops.
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import cfg, DEVICE
from data.loader import PrefetchLoader


def train_one_epoch(model, loader, optimizer, criterion, device,
                    is_fusion=False):
    model.train()
    total_loss = total_hm = total_coord = 0.0
    n_batches  = len(loader)
    n_skipped  = 0

    # Wrap with prefetch so next batch loads while GPU processes current one
    prefetch = PrefetchLoader(loader, device)

    for batch in tqdm(prefetch, desc='  Train', leave=False, total=n_batches):
        radar = batch['radar']    # already on GPU via prefetch
        gt_hm = batch['heatmap']
        gt_kp = batch['coords']
        vis   = batch['vis']

        if not torch.isfinite(radar).all():
            n_skipped += 1
            continue

        optimizer.zero_grad()
        pred_hm, pred_kp = model(radar)

        if not torch.isfinite(pred_hm).all():
            n_skipped += 1
            optimizer.zero_grad()
            continue

        loss, lhm, lc = criterion(pred_hm, pred_kp, gt_hm, gt_kp, vis)

        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_hm    += lhm
        total_coord += lc

    n_good = n_batches - n_skipped
    if n_skipped > 0:
        print(f"  [WARNING] Skipped {n_skipped}/{n_batches} batches due to nan/inf")
    denom = max(n_good, 1)
    return total_loss/denom, total_hm/denom, total_coord/denom


@torch.no_grad()
def validate(model, loader, criterion, device, is_fusion=False):
    model.eval()
    total_loss = total_hm = total_coord = 0.0
    n_batches  = len(loader)
    n_skipped  = 0

    # Prefetch for validation too
    prefetch = PrefetchLoader(loader, device)

    for batch in tqdm(prefetch, desc='  Val  ', leave=False, total=n_batches):
        radar = batch['radar']
        gt_hm = batch['heatmap']
        gt_kp = batch['coords']
        vis   = batch['vis']

        if not torch.isfinite(radar).all():
            n_skipped += 1
            continue

        pred_hm, pred_kp = model(radar)

        if not torch.isfinite(pred_hm).all():
            n_skipped += 1
            continue

        loss, lhm, lc = criterion(pred_hm, pred_kp, gt_hm, gt_kp, vis)

        if not torch.isfinite(loss):
            n_skipped += 1
            continue

        total_loss  += loss.item()
        total_hm    += lhm
        total_coord += lc

    n_good = n_batches - n_skipped
    denom  = max(n_good, 1)
    return total_loss/denom, total_hm/denom, total_coord/denom


def run_training(model, model_name, train_loader, val_loader,
                 cfg, criterion, is_fusion=False):
    model     = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.LR_STEP, gamma=cfg.LR_GAMMA)

    history   = {'train_loss':[], 'val_loss':[], 'train_hm':[],
                 'val_hm':[], 'train_coord':[], 'val_coord':[]}
    best_val  = float('inf')
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f'{model_name}_best.pth')

    print(f"\n{'='*60}")
    print(f" Training : {model_name}")
    print(f" Epochs   : {cfg.NUM_EPOCHS}  |  LR: {cfg.LR}  |  Device: {DEVICE}")
    print(f"{'='*60}")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_hm, tr_c = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        va_loss, va_hm, va_c = validate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        for k, v in zip(['train_loss','val_loss','train_hm','val_hm',
                         'train_coord','val_coord'],
                        [tr_loss,va_loss,tr_hm,va_hm,tr_c,va_c]):
            history[k].append(v)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_loss': best_val}, ckpt_path)
            flag = ' ✓'
        else:
            flag = ''

        print(f"Ep {epoch:3d}/{cfg.NUM_EPOCHS} | "
              f"Train {tr_loss:.4f} (hm {tr_hm:.4f}, kp {tr_c:.4f}) | "
              f"Val {va_loss:.4f} (hm {va_hm:.4f}, kp {va_c:.4f}) | "
              f"{time.time()-t0:.1f}s{flag}")

    print(f"\nBest val loss: {best_val:.4f}  →  {ckpt_path}")
    return history


def run_training_radar(model, model_name, train_loader, val_loader, cfg, criterion):
    """Convenience wrapper for radar-only model training."""
    return run_training(model, model_name, train_loader, val_loader,
                        cfg, criterion, is_fusion=False)
