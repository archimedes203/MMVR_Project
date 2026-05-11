MMVR Pose Detection -- CSE 486/586 Group Project

Human Pose Detection / Keypoint Estimation using Millimeter-Wave Radar (MMVR) Dataset

Andrew Dirr - Dean DiCarlo - AJ Marin
Miami University, Oxford, OH -- Spring 2026

================================================================================
OVERVIEW
================================================================================

This project implements a complete pose estimation pipeline that predicts 17
COCO-format human keypoints from millimeter-wave radar heatmaps (no RGB camera
required at inference time). Three model architectures are compared:

  Model             Parameters   Description
  ---------------   ----------   ------------------------------------------------
  CustomCNN           1.78M      Residual encoder-decoder with CBAM attention
  ResNet-18           3.15M      Pretrained backbone adapted for 2-channel radar input
  FusionModel         2.46M      Dual-stream (horizontal + vertical radar) with
                                 bottleneck fusion
  PoseLiftingMLP       285K      Lifts 2D predictions to 3D using radar depth cues

================================================================================
PROJECT STRUCTURE
================================================================================

MMVR_Project/
├── config.py                     Configuration, seeds, device setup
├── main.py                       CLI entry point
├── tune.py                       Hyperparameter tuning script
├── requirements.txt              Python dependencies
├── .gitignore                    Git exclusions
│
├── data/                         Data loading & preprocessing
│   ├── dataset.py                MMVRDataset, AdverseConditionDataset
│   ├── loader.py                 PrefetchLoader, DataLoader factory
│   ├── heatmap.py                Gaussian heatmap generation
│   ├── splits.py                 Official MMVR train/val/test splits
│   └── explore.py                Dataset structure explorer
│
├── models/                       Model architectures
│   ├── blocks.py                 ResidualBlock, CBAM, SoftArgmax2D
│   ├── custom_cnn.py             Custom CNN encoder-decoder
│   ├── resnet_baseline.py        ResNet-18 baseline
│   ├── fusion.py                 Dual-stream radar fusion
│   └── lifter.py                 2D -> 3D pose lifting
│
├── training/                     Training pipeline
│   ├── loss.py                   PoseLoss (heatmap + coordinate)
│   ├── train.py                  Training/validation loops
│   └── checkpoint.py             Save/load/export checkpoints
│
├── evaluation/                   Evaluation metrics
│   └── metrics.py                PCK, OKS, MAE, F1, full evaluation
│
└── visualization/                Plotting & visualisation
    ├── sample_vis.py             Raw sample + skeleton drawing
    ├── loss_plots.py             Loss curves
    ├── comparison.py             Model comparison charts
    ├── skeleton_overlay.py       GT vs predicted skeleton overlays
    ├── radar_overlay.py          Radar heatmap overlays
    ├── pose_3d.py                3D pose reconstruction plots
    └── adverse.py                Adverse condition robustness plots

================================================================================
SETUP
================================================================================

1. Create Virtual Environment
------------------------------

    python3 -m venv venv
    source venv/bin/activate        # Linux/Mac
    # venv\Scripts\activate         # Windows

2. Install PyTorch (CUDA)
--------------------------

Visit https://pytorch.org/get-started/locally/ and install for your CUDA
version. Example for CUDA 12.8:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

3. Install Dependencies
------------------------

    pip install -r requirements.txt

4. Place Dataset
-----------------

Extract the MMVR P1 dataset and split file into the project root:

    MMVR_Project/
    ├── P1/               Dataset directory (~80 GB)
    └── data_split.npz    Official train/val/test splits

By default, config.py looks for the dataset at:
    /home/<user>/MMVR_Project/P1

Override this without editing code by setting an environment variable:

    export MMVR_DATA_ROOT=/path/to/your/P1      # Linux/Mac
    set MMVR_DATA_ROOT=C:\path\to\your\P1       # Windows

================================================================================
USAGE
================================================================================

    python main.py <command> [options]

Commands:
  explore       Print dataset structure and sample counts
  train         Train all three models
  evaluate      Evaluate saved checkpoints on the test set
  visualize     Generate loss curves and comparison plots
  export        Copy final .pth files into results/
  all           Run train -> evaluate -> visualize -> export  (default)

CLI options (override values from config.py):
  --epochs N        Number of training epochs     (default: 30)
  --batch-size N    Batch size                    (default: 128)
  --lr LR           Learning rate                 (default: 0.001)
  --protocol P      Data split protocol           (default: P1S1)
                    Options: P1S1, P1S2, P2S1, P2S2

Examples:
  python main.py all
  python main.py train --epochs 50 --batch-size 16 --lr 0.0005
  python main.py train --protocol P2S1
  python main.py evaluate

================================================================================
HYPERPARAMETER TUNING
================================================================================

tune.py runs a random (or full grid) search over learning rate, weight decay,
and loss function weights (lambda_hm, lambda_coord).

    python tune.py                        # random search, 30 combos, 15 epochs
    python tune.py --n-samples 50         # try 50 random combos per model
    python tune.py --epochs 20            # 20 epochs per trial
    python tune.py --model resnet18       # tune one model only
    python tune.py --grid                 # full grid search (240 combos -- slow)
    python tune.py --plot-only            # regenerate plots from existing CSV

Results are saved incrementally to results/tuning_results.csv so the run can
be interrupted and resumed. After identifying the best config, update config.py
and retrain with: python main.py train

================================================================================
EVALUATION METRICS
================================================================================

  PCK@0.05   Percentage of Correct Keypoints
             (threshold = 5% of max image dimension)
  OKS        Object Keypoint Similarity (COCO-standard)
  MAE        Mean Absolute Error in pixels
  F1         Detection-level F1 score

The FusionModel is also evaluated under simulated adverse radar conditions:
  noise, dropout, low_power

================================================================================
OUTPUT
================================================================================

  checkpoints/          Best model weights saved during training (.pth)

  results/
    loss_curves.png           Training/validation loss curves
    model_comparison.png      Side-by-side model metric comparison
    pck_per_keypoint.png      Per-keypoint PCK breakdown
    eval_results.json         Full evaluation results
    histories.json            Training history (loss per epoch)
    custom_cnn_final.pth      Exported CustomCNN weights
    resnet18_final.pth        Exported ResNet-18 weights
    fusion_final.pth          Exported FusionModel weights
    tuning_results.csv        Hyperparameter tuning log (if tune.py was run)
