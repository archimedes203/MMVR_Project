# MMVR Pose Detection — CSE 486/586 Group Project

**Human Pose Detection / Keypoint Estimation using Millimeter-Wave Radar (MMVR) Dataset**

Miami University, Oxford, OH — Spring 2026

## Overview

This project implements a complete pose estimation pipeline that predicts 17 COCO-format human keypoints from **millimeter-wave radar heatmaps** (no RGB camera required at inference time). Three model architectures are compared:

| Model | Parameters | Description |
|-------|-----------|-------------|
| **CustomCNN** | 1.78M | Residual encoder-decoder with CBAM attention |
| **ResNet-18** | 3.15M | Pretrained backbone adapted for 2-channel radar input |
| **FusionModel** | 2.46M | Dual-stream (horizontal + vertical radar) with bottleneck fusion |
| **PoseLiftingMLP** | 285K | Lifts 2D predictions to 3D using radar depth cues |

## Project Structure

```
MMVR_Project/
├── config.py                     # Configuration, seeds, device setup
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusions
│
├── data/                         # Data loading & preprocessing
│   ├── dataset.py                # MMVRDataset, AdverseConditionDataset
│   ├── loader.py                 # PrefetchLoader, DataLoader factory
│   ├── heatmap.py                # Gaussian heatmap generation
│   ├── splits.py                 # Official MMVR train/val/test splits
│   └── explore.py                # Dataset structure explorer
│
├── models/                       # Model architectures
│   ├── blocks.py                 # ResidualBlock, CBAM, SoftArgmax2D
│   ├── custom_cnn.py             # Custom CNN encoder-decoder
│   ├── resnet_baseline.py        # ResNet-18 baseline
│   ├── fusion.py                 # Dual-stream radar fusion
│   └── lifter.py                 # 2D → 3D pose lifting
│
├── training/                     # Training pipeline
│   ├── loss.py                   # PoseLoss (heatmap + coordinate)
│   ├── train.py                  # Training/validation loops
│   └── checkpoint.py             # Save/load/export checkpoints
│
├── evaluation/                   # Evaluation metrics
│   └── metrics.py                # PCK, OKS, MAE, F1, full evaluation
│
└── visualization/                # Plotting & visualisation
    ├── sample_vis.py             # Raw sample + skeleton drawing
    ├── loss_plots.py             # Loss curves
    ├── comparison.py             # Model comparison charts
    ├── skeleton_overlay.py       # GT vs predicted skeleton overlays
    ├── radar_overlay.py          # Radar heatmap overlays
    ├── pose_3d.py                # 3D pose reconstruction plots
    └── adverse.py                # Adverse condition robustness plots
```

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### 2. Install PyTorch (CUDA)

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and install for your CUDA version. Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place Dataset

Extract the MMVR P1 dataset and split file into the project root:

```
MMVR_Project/
├── P1/           # Dataset directory
├── data_split.npz  # Official splits
```

## Usage

```bash
# Explore dataset structure
python main.py explore

# Train all models (CustomCNN, ResNet-18, FusionModel)
python main.py train

# Evaluate on test set (requires trained checkpoints)
python main.py evaluate

# Generate all plots (requires saved histories/eval_results)
python main.py visualize

# Export final models for submission
python main.py export

# Run full pipeline end-to-end
python main.py all
```

### CLI Options

```bash
python main.py train --epochs 50 --batch-size 16 --lr 0.0005
python main.py train --protocol P2S1
```

## Evaluation Metrics

- **PCK@0.05** — Percentage of Correct Keypoints (threshold = 5% of max image dimension)
- **OKS** — Object Keypoint Similarity (COCO-standard)
- **MAE** — Mean Absolute Error in pixels
- **F1 / Precision** — Detection-level metrics
