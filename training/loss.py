"""
Combined pose estimation loss function.
"""

import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    """
    Combined pose estimation loss:
        L = λ_hm * L_heatmap  +  λ_coord * L_coord

    - L_heatmap : MSE between predicted and GT Gaussian heatmaps.
                  GT heatmap is resized to match the model's output
                  spatial size, since the non-square radar input means
                  pred_hm dimensions vary by model.
    - L_coord   : Smooth-L1 between predicted and GT (x,y) coordinates,
                  weighted by keypoint visibility.
    """

    def __init__(self, lambda_hm=1.0, lambda_coord=5.0):
        super().__init__()
        self.lambda_hm    = lambda_hm
        self.lambda_coord = lambda_coord
        self.smooth_l1    = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_hm, pred_coords, gt_hm, gt_coords, visibility):
        """
        Args:
            pred_hm     : (B, 17, H_pred, W_pred)  — model heatmap output
            pred_coords : (B, 17, 2)
            gt_hm       : (B, 17, H_gt, W_gt)      — Gaussian GT heatmaps
            gt_coords   : (B, 17, 2)
            visibility  : (B, 17)
        """
        # Resize GT heatmap to match pred_hm spatial size if they differ
        if pred_hm.shape[2:] != gt_hm.shape[2:]:
            # gt_hm is (B, 17, H, W) — interpolate expects (N, C, H, W)
            gt_hm = F.interpolate(gt_hm, size=pred_hm.shape[2:],
                                  mode='bilinear', align_corners=False)

        # Heatmap MSE
        loss_hm = F.mse_loss(pred_hm, gt_hm)

        # Coordinate SmoothL1 masked by visibility
        vis_mask   = (visibility > 0).float().unsqueeze(-1)   # (B, 17, 1)
        loss_coord = self.smooth_l1(pred_coords, gt_coords) * vis_mask
        n_vis      = vis_mask.sum().clamp(min=1)
        loss_coord = loss_coord.sum() / n_vis

        total = self.lambda_hm * loss_hm + self.lambda_coord * loss_coord
        return total, loss_hm.item(), loss_coord.item()
