"""
Configuration, reproducibility, and device setup for the MMVR Pose Detection pipeline.
"""

import os
import random
import numpy as np
import torch


# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════

class Config:
    # ── Paths ────────────────────────────────────────────────────────
    DATA_ROOT       = r'C:\Users\andyd\Desktop\MIAMI\GRADUATE\SPRING 2026\CSE586 Introduction to AI\MIDTERM PROJECT\CLAUDE\P1'
    SPLIT_FILE      = r'C:\Users\andyd\Desktop\MIAMI\GRADUATE\SPRING 2026\CSE586 Introduction to AI\MIDTERM PROJECT\CLAUDE\data_split.npz'
    CHECKPOINT_DIR  = './checkpoints'
    RESULTS_DIR     = './results'

    # ── Which protocol + split to use (from data_split.npz) ─────────
    # P1S1: single subject, random split         (recommended for this project)
    # P1S2: single subject, cross-environment split
    # P2S1: multiple subjects, random split
    # P2S2: multiple subjects, cross-environment split
    PROTOCOL        = 'P1S2' # Changed to P1S1 (AD 04/18/26)

    # ── Real image dimensions from MMVR dataset ──────────────────────
    IMG_H           = 480            # actual camera image height
    IMG_W           = 640            # actual camera image width

    # ── Radar heatmap dimensions (from README) ────────────────────────
    RADAR_H         = 256
    RADAR_W         = 128
    HEATMAP_SIZE    = 64             # output heatmap spatial resolution for model
    SIGMA           = 3              # Gaussian sigma for GT heatmap generation

    # ── Keypoints ────────────────────────────────────────────────────
    NUM_KEYPOINTS   = 17
    KP_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    SKELETON = [
        (13,15),(11,13),(14,16),(12,14),(11,12),
        (5,11),(6,12),(5,6),(5,7),(6,8),(7,9),
        (8,10),(1,2),(0,1),(0,2),(1,3),(2,4),
        (3,5),(4,6)
    ]

    # ── Training ─────────────────────────────────────────────────────
    BATCH_SIZE      = 64   # Changed from 32 to 64 to draw more GPU performance via CUDA.
    NUM_EPOCHS      = 1    # Changed to 1 for testing purposes
    LR              = 1e-3   # changed from 1e-4 to 1e-3 for testing purposes
    LR_STEP         = [15, 25]
    LR_GAMMA        = 0.1
    WEIGHT_DECAY    = 1e-4
    NUM_WORKERS     = 0   # 0 required on Windows to avoid DataLoader deadlock

    # ── Evaluation ───────────────────────────────────────────────────
    PCK_THRESHOLD   = 0.05
    OKS_SIGMAS      = np.array([
        .026,.025,.025,.035,.035,
        .079,.079,.072,.072,.062,
        .062,.107,.107,.087,.087,
        .089,.089
    ])


# ── Instantiate global config ────────────────────────────────────────
cfg = Config()


# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Device ───────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Create output directories ───────────────────────────────────────
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR,    exist_ok=True)
