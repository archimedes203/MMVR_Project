"""
Evaluation metrics for pose estimation: PCK, OKS, MAE, F1, Precision.
"""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score

from config import cfg


def compute_pck(pred_coords, gt_coords, visibility,
                img_h=None, img_w=None, threshold=None):
    """
    Percentage of Correct Keypoints (PCK@threshold).
    A keypoint is correct if its Euclidean distance to GT
    is within threshold * max(img_h, img_w) pixels.
    Returns (pck_overall, pck_per_kp array of shape (17,)).
    """
    img_h     = img_h     or cfg.IMG_H
    img_w     = img_w     or cfg.IMG_W
    threshold = threshold or cfg.PCK_THRESHOLD
    ref_dist  = threshold * max(img_h, img_w)

    pred_px = pred_coords.copy()
    gt_px   = gt_coords.copy()
    pred_px[:,:,0] *= img_w;  pred_px[:,:,1] *= img_h
    gt_px  [:,:,0] *= img_w;  gt_px  [:,:,1] *= img_h

    dist     = np.sqrt(((pred_px - gt_px) ** 2).sum(axis=-1))  # (N, 17)
    correct  = dist < ref_dist
    vis_mask = visibility > 0

    pck_per_kp = np.zeros(17)
    for k in range(17):
        m = vis_mask[:, k]
        if m.sum() > 0:
            pck_per_kp[k] = correct[:, k][m].mean()

    valid       = pck_per_kp > 0
    pck_overall = pck_per_kp[valid].mean() if valid.any() else 0.0
    return float(pck_overall), pck_per_kp


def compute_oks(pred_coords, gt_coords, visibility,
                sigmas=None, img_h=None, img_w=None):
    """
    Object Keypoint Similarity (OKS).
    Returns (mean_oks, per_sample_oks array of shape (N,)).
    """
    img_h  = img_h  or cfg.IMG_H
    img_w  = img_w  or cfg.IMG_W
    sigmas = sigmas if sigmas is not None else cfg.OKS_SIGMAS
    k2     = max(img_h, img_w) ** 2

    pred_px = pred_coords.copy()
    gt_px   = gt_coords.copy()
    pred_px[:,:,0] *= img_w;  pred_px[:,:,1] *= img_h
    gt_px  [:,:,0] *= img_w;  gt_px  [:,:,1] *= img_h

    d2       = ((pred_px - gt_px) ** 2).sum(axis=-1)          # (N, 17)
    s2       = (sigmas * 2) ** 2                                # (17,)
    oks_vals = np.exp(-d2 / (2 * s2[None] * k2))               # (N, 17)

    vis_mask       = (visibility > 0).astype(float)
    per_sample_oks = ((oks_vals * vis_mask).sum(1)
                      / vis_mask.sum(1).clip(min=1))
    return float(per_sample_oks.mean()), per_sample_oks


def compute_mae(pred_coords, gt_coords, visibility,
                img_h=None, img_w=None):
    """Mean Absolute Error in pixels (visible keypoints only)."""
    img_h = img_h or cfg.IMG_H
    img_w = img_w or cfg.IMG_W

    pred_px = pred_coords.copy()
    gt_px   = gt_coords.copy()
    pred_px[:,:,0] *= img_w;  pred_px[:,:,1] *= img_h
    gt_px  [:,:,0] *= img_w;  gt_px  [:,:,1] *= img_h

    diff = np.abs(pred_px - gt_px).sum(-1)   # (N, 17)
    vis  = visibility > 0
    return float(diff[vis].mean()) if vis.sum() > 0 else 0.0


def compute_f1_precision(pred_coords, gt_coords, visibility,
                         img_h=None, img_w=None, threshold=None):
    """
    F1-Score and Precision for keypoint detection.

    The model always produces a prediction for every keypoint slot.
    We therefore treat every slot as a detection attempt:
      - GT positive  : keypoint is annotated (visible > 0)
      - Pred positive: prediction is within the distance threshold
                       (regardless of visibility)

      TP: within threshold  AND  GT visible
      FP: outside threshold AND  GT visible   (wrong prediction)
      FN: visible keypoint not predicted within threshold
    """
    img_h     = img_h     or cfg.IMG_H
    img_w     = img_w     or cfg.IMG_W
    threshold = threshold or cfg.PCK_THRESHOLD
    ref_dist  = threshold * max(img_h, img_w)

    pred_px = pred_coords.copy()
    gt_px   = gt_coords.copy()
    pred_px[:,:,0] *= img_w;  pred_px[:,:,1] *= img_h
    gt_px  [:,:,0] *= img_w;  gt_px  [:,:,1] *= img_h

    dist    = np.sqrt(((pred_px - gt_px) ** 2).sum(-1))  # (N, 17)
    within  = (dist < ref_dist).flatten()                 # pred positive
    visible = (visibility > 0).flatten()                  # GT positive

    # y_true: 1 = GT has a keypoint here
    y_true = visible.astype(int)
    # y_pred: 1 = model predicted within threshold (irrespective of visibility)
    y_pred = within.astype(int)

    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    return float(f1), float(prec)


@torch.no_grad()
def evaluate_model(model, loader, device, model_name='model'):
    """Evaluate a trained model on a DataLoader. Returns metrics dict."""
    model.eval()
    all_pred, all_gt, all_vis = [], [], []

    for batch in tqdm(loader, desc=f'  Eval {model_name}'):
        radar = batch['radar'].to(device)
        gt_kp = batch['coords'].cpu().numpy()
        vis   = batch['vis'].cpu().numpy()

        _, pred_kp = model(radar)
        all_pred.append(pred_kp.cpu().numpy())
        all_gt.append(gt_kp)
        all_vis.append(vis)

    pred_all = np.concatenate(all_pred, axis=0)
    gt_all   = np.concatenate(all_gt,   axis=0)
    vis_all  = np.concatenate(all_vis,  axis=0)

    pck, pck_per_kp = compute_pck(pred_all, gt_all, vis_all)
    oks, _          = compute_oks(pred_all, gt_all, vis_all)
    mae             = compute_mae(pred_all, gt_all, vis_all)
    f1, prec        = compute_f1_precision(pred_all, gt_all, vis_all)

    results = {'model': model_name, 'PCK@0.05': pck, 'OKS': oks,
               'MAE_px': mae, 'F1': f1, 'Precision': prec,
               'pck_per_kp': pck_per_kp}

    print(f"\n── {model_name} ──────────────────────────")
    print(f"  PCK@0.05  : {pck*100:.2f}%")
    print(f"  OKS       : {oks:.4f}")
    print(f"  MAE (px)  : {mae:.2f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    return results


def print_summary_table(eval_results):
    """Pretty-print a comparison table of all models."""
    if not eval_results:
        print("No results."); return

    header = f"{'Model':<15} {'PCK@5%':>9} {'OKS':>9} {'MAE(px)':>9} {'F1':>9} {'Prec':>9}"
    print("\n" + "="*60)
    print(header)
    print("-"*60)
    for mname, r in eval_results.items():
        print(f"{mname:<15} "
              f"{r['PCK@0.05']*100:>8.2f}% "
              f"{r['OKS']:>9.4f} "
              f"{r['MAE_px']:>9.2f} "
              f"{r['F1']:>9.4f} "
              f"{r['Precision']:>9.4f}")
    print("="*60)
