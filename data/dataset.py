"""
PyTorch Dataset classes for the MMVR P1 dataset.
"""

import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from config import cfg
from data.heatmap import generate_gaussian_heatmap


class MMVRDataset(Dataset):
    """
    PyTorch Dataset for the MMVR P1 dataset.
    Loads radar heatmaps and keypoints directly from .npz files.
    """

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load_radar(self, radar_path):
        """
        Load and sanitise hm_hori + hm_vert from _radar.npz.
        Replaces any nan/inf values with 0 before normalising.
        Returns Tensor (2, RADAR_H, RADAR_W).
        """
        data = np.load(radar_path)
        hori = data['hm_hori'].astype(np.float32)
        vert = data['hm_vert'].astype(np.float32)

        for arr in [hori, vert]:
            # Replace nan and inf with 0 before doing anything else
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalise to [0, 1]
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr[:] = (arr - mn) / (mx - mn)

        if hori.shape != (cfg.RADAR_H, cfg.RADAR_W):
            hori = cv2.resize(hori, (cfg.RADAR_W, cfg.RADAR_H),
                              interpolation=cv2.INTER_LINEAR)
            vert = cv2.resize(vert, (cfg.RADAR_W, cfg.RADAR_H),
                              interpolation=cv2.INTER_LINEAR)

        return torch.from_numpy(np.stack([hori, vert], axis=0))

    def _load_keypoints(self, pose_path, person_idx):
        """
        Load keypoints for one person from _pose.npz.
        Returns kp_norm (17,2) normalised to [0,1] and vis (17,).
        """
        data = np.load(pose_path)
        kp   = data['kp'][person_idx]       # (17, 3)
        x, y, vis = kp[:,0], kp[:,1], kp[:,2]
        # x = col (width axis), y = row (height axis)
        x_norm = np.clip(x / cfg.IMG_W, 0.0, 1.0).astype(np.float32)
        y_norm = np.clip(y / cfg.IMG_H, 0.0, 1.0).astype(np.float32)
        kp_norm = np.stack([x_norm, y_norm], axis=1)  # (17,2): [col_norm, row_norm]
        # Sanitise keypoints too
        np.nan_to_num(kp_norm, copy=False, nan=0.0)
        np.nan_to_num(vis,     copy=False, nan=0.0)
        return kp_norm, vis.astype(np.float32)

    def _augment_radar(self, radar):
        if random.random() < 0.5:
            radar = np.flip(radar, axis=2).copy()
        if random.random() < 0.3:
            radar = np.clip(
                radar + np.random.randn(*radar.shape).astype(np.float32)*0.02,
                0, 1)
        if random.random() < 0.3:
            radar = np.clip(radar * random.uniform(0.7, 1.3), 0, 1)
        return radar

    def _augment_keypoints(self, kp_norm, vis):
        FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
        kp_norm = kp_norm.copy(); vis = vis.copy()
        if random.random() < 0.5:
            kp_norm[:, 1] = 1.0 - kp_norm[:, 1]
            for l, r in FLIP_PAIRS:
                kp_norm[[l,r]] = kp_norm[[r,l]]
                vis[[l,r]]     = vis[[r,l]]
        if random.random() < 0.3:
            kp_norm = np.clip(
                kp_norm + np.random.randn(*kp_norm.shape).astype(np.float32)*0.01,
                0, 1)
        return kp_norm, vis

    def __getitem__(self, idx):
        s = self.samples[idx]
        radar            = self._load_radar(s['radar_path'])
        kp_norm, vis     = self._load_keypoints(s['pose_path'], s['person_idx'])

        if self.augment:
            radar         = self._augment_radar(radar.numpy())
            radar         = torch.from_numpy(radar)
            kp_norm, vis  = self._augment_keypoints(kp_norm, vis)

        kp_abs      = kp_norm.copy()
        kp_abs[:,0] *= cfg.IMG_W   # x = col → scale by W
        kp_abs[:,1] *= cfg.IMG_H   # y = row → scale by H
        kp_with_vis = np.concatenate([kp_abs, vis[:,None]], axis=1)
        heatmap     = generate_gaussian_heatmap(cfg.HEATMAP_SIZE,
                                                kp_with_vis, sigma=cfg.SIGMA)
        return {
            'radar'   : radar,
            'heatmap' : torch.from_numpy(heatmap),
            'coords'  : torch.from_numpy(kp_norm).float(),
            'vis'     : torch.from_numpy(vis).float(),
            'session' : s['session'],
        }


class AdverseConditionDataset(Dataset):
    """
    Wraps MMVRDataset and applies fixed perturbations to radar heatmaps
    to simulate challenging sensing conditions:
        'noise'     — strong Gaussian noise on radar
        'dropout'   — randomly zero out 30% of radar pixels (occlusion proxy)
        'low_power' — scale radar signal to 20% (weak return / dark room proxy)
    """

    def __init__(self, base_dataset, condition='noise'):
        self.base      = base_dataset
        self.condition = condition

    def __len__(self):
        return len(self.base)

    def _apply(self, radar_tensor):
        t = radar_tensor.clone()
        if self.condition == 'noise':
            noise = torch.randn_like(t) * 0.15
            return torch.clamp(t + noise, 0.0, 1.0)
        elif self.condition == 'dropout':
            mask = (torch.rand_like(t) > 0.30).float()
            return t * mask
        elif self.condition == 'low_power':
            return t * 0.2
        return t

    def __getitem__(self, idx):
        sample = dict(self.base[idx])
        sample['radar'] = self._apply(sample['radar'])
        return sample
