"""
Combined pose estimation loss function.
"""

import torch.nn as nn
# import torch.nn.functional as F   # retained for reference; unused after heatmap loss disabled


class PoseLoss(nn.Module):
    """
    Combined pose estimation loss:
        L = λ_coord * L_coord_nll

    - L_heatmap   : MSE between predicted and GT Gaussian heatmaps.
                    DISABLED — see commented-out block in forward().
    - L_coord_nll : Gaussian negative log-likelihood between predicted and GT
                    (x,y) coordinates, using a per-keypoint log-variance tensor
                    predicted by the model, weighted by keypoint visibility.

                    NLL = 0.5 * (log σ² + (pred − gt)² / σ²)

                    The model must output pred_log_var of shape (B, 17, 2)
                    alongside pred_coords.
    """

    def __init__(self, lambda_hm=1.0, lambda_coord=5.0):
        super().__init__()
        self.lambda_hm    = lambda_hm
        self.lambda_coord = lambda_coord
        # self.smooth_l1  = nn.SmoothL1Loss(reduction='none')   # replaced by Gaussian NLL
        self.gaussian_nll = nn.GaussianNLLLoss(full=True, reduction='none')

    def forward(self, pred_hm, pred_coords, pred_log_var, gt_hm, gt_coords, visibility):
        """
        Args:
            pred_hm      : (B, 17, H_pred, W_pred)  — model heatmap output
            pred_coords  : (B, 17, 2)                — predicted keypoint coords
            pred_log_var : (B, 17, 2)                — predicted log-variance per keypoint axis
            gt_hm        : (B, 17, H_gt, W_gt)       — Gaussian GT heatmaps
            gt_coords    : (B, 17, 2)
            visibility   : (B, 17)
        """
        # ── Heatmap loss (MSE) — DISABLED ───────────────────────────────────
        # if pred_hm.shape[2:] != gt_hm.shape[2:]:
        #     gt_hm = F.interpolate(gt_hm, size=pred_hm.shape[2:],
        #                           mode='bilinear', align_corners=False)
        # loss_hm = F.mse_loss(pred_hm, gt_hm)

        # ── Coordinate loss: Gaussian NLL with learned variance ──────────────
        # NLL = 0.5 * (log σ² + (pred − gt)² / σ²)
        # pred_log_var encodes log(σ²), so σ² = exp(pred_log_var)
        var      = pred_log_var.exp().clamp(min=1e-6)       # (B, 17, 2)
        vis_mask = (visibility > 0).float().unsqueeze(-1)   # (B, 17, 1)

        # GaussianNLLLoss(input, target, var) with reduction='none' → (B, 17, 2)
        loss_nll   = self.gaussian_nll(pred_coords, gt_coords, var)
        loss_coord = (loss_nll * vis_mask).sum() / vis_mask.sum().clamp(min=1)

        # ── Coordinate loss (SmoothL1) — DISABLED ───────────────────────────
        # vis_mask   = (visibility > 0).float().unsqueeze(-1)
        # loss_coord = self.smooth_l1(pred_coords, gt_coords) * vis_mask
        # n_vis      = vis_mask.sum().clamp(min=1)
        # loss_coord = loss_coord.sum() / n_vis

        # ── Total ────────────────────────────────────────────────────────────
        # total = self.lambda_hm * loss_hm + self.lambda_coord * loss_coord   # DISABLED
        total = self.lambda_coord * loss_coord

        # Second return value is 0.0 (placeholder) since heatmap loss is disabled
        return total, 0.0, loss_coord.item()
