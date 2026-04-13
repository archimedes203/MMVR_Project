#!/usr/bin/env python3
"""
MMVR Pose Detection — Main CLI Entry Point
===========================================

Usage:
    python main.py explore           # Explore dataset structure
    python main.py train             # Train all three models
    python main.py evaluate          # Evaluate saved models on test set
    python main.py visualize         # Generate all visualization plots
    python main.py export            # Export final models for submission
    python main.py all               # Full pipeline: train → evaluate → visualize → export

Options:
    --epochs N          Override number of training epochs
    --batch-size N      Override batch size
    --lr LR             Override learning rate
    --protocol P        Override data split protocol (P1S1, P1S2, P2S1, P2S2)
"""

import argparse
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg, DEVICE

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMVR Pose Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('command', nargs='?', default='all',
                        choices=['explore', 'train', 'evaluate', 'visualize',
                                 'export', 'all'],
                        help='Pipeline stage to run (default: all)')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Override NUM_EPOCHS (default: {cfg.NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Override BATCH_SIZE (default: {cfg.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'Override LR (default: {cfg.LR})')
    parser.add_argument('--protocol', type=str, default=None,
                        help=f'Override PROTOCOL (default: {cfg.PROTOCOL})')
    return parser.parse_args()


def apply_overrides(args):
    """Apply CLI overrides to the Config object."""
    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs
        print(f"  Override: NUM_EPOCHS = {cfg.NUM_EPOCHS}")
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
        print(f"  Override: BATCH_SIZE = {cfg.BATCH_SIZE}")
    if args.lr is not None:
        cfg.LR = args.lr
        print(f"  Override: LR = {cfg.LR}")
    if args.protocol is not None:
        cfg.PROTOCOL = args.protocol
        print(f"  Override: PROTOCOL = {cfg.PROTOCOL}")


def cmd_explore():
    """Explore the dataset structure."""
    from data.explore import explore_dataset
    print("\n" + "="*60)
    print(" EXPLORING DATASET")
    print("="*60)
    explore_dataset(cfg.DATA_ROOT)


def cmd_train():
    """Train all three models."""
    from data.splits import load_mmvr_samples_split
    from data.loader import create_dataloaders_from_splits
    from models.custom_cnn import CustomCNN
    from models.resnet_baseline import ResNet18PoseModel
    from models.fusion import FusionModel
    from training.loss import PoseLoss
    from training.train import run_training_radar
    from training.checkpoint import save_histories

    print("\n" + "="*60)
    print(" LOADING DATA")
    print("="*60)

    train_samples, val_samples, test_samples = load_mmvr_samples_split(
        cfg.DATA_ROOT, cfg.SPLIT_FILE, cfg.PROTOCOL)

    if not train_samples:
        print("[ERROR] No training samples found. Check DATA_ROOT and SPLIT_FILE.")
        return

    train_loader, val_loader, test_loader, _, _, _ = \
        create_dataloaders_from_splits(train_samples, val_samples, test_samples, cfg)

    criterion = PoseLoss(lambda_hm=1.0, lambda_coord=5.0)

    print("\n" + "="*60)
    print(" TRAINING")
    print("="*60)
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {cfg.NUM_EPOCHS}")
    print(f"  LR:     {cfg.LR}")

    histories = {}
    for model_name, model in [
        ('custom_cnn', CustomCNN(num_kp=cfg.NUM_KEYPOINTS, in_channels=2)),
        ('resnet18',   ResNet18PoseModel(num_kp=cfg.NUM_KEYPOINTS, in_channels=2)),
        ('fusion',     FusionModel(num_kp=cfg.NUM_KEYPOINTS)),
    ]:
        print(f'\n>>> Training {model_name}...')
        try:
            histories[model_name] = run_training_radar(
                model, model_name, train_loader, val_loader, cfg, criterion)
        except KeyboardInterrupt:
            print(f'\n[Interrupted] {model_name} training stopped by user.')
            break
        except Exception as e:
            print(f'\n[ERROR] {model_name} failed: {e}')
            raise

    save_histories(histories)
    print('\nTraining complete.')


def cmd_evaluate():
    """Evaluate all models on the test set."""
    from data.splits import load_mmvr_samples_split
    from data.loader import create_dataloaders_from_splits
    from models.custom_cnn import CustomCNN
    from models.resnet_baseline import ResNet18PoseModel
    from models.fusion import FusionModel
    from models.lifter import PoseLiftingMLP
    from data.dataset import AdverseConditionDataset
    from training.checkpoint import load_checkpoint, save_eval_results
    from evaluation.metrics import evaluate_model, print_summary_table

    print("\n" + "="*60)
    print(" EVALUATING")
    print("="*60)

    train_samples, val_samples, test_samples = load_mmvr_samples_split(
        cfg.DATA_ROOT, cfg.SPLIT_FILE, cfg.PROTOCOL)

    if not test_samples:
        print("[ERROR] No test samples found.")
        return

    _, _, test_loader, _, _, _ = \
        create_dataloaders_from_splits(train_samples, val_samples, test_samples, cfg)

    eval_results = {}
    for mname, ModelClass in [
        ('custom_cnn', lambda: CustomCNN(cfg.NUM_KEYPOINTS, in_channels=2)),
        ('resnet18',   lambda: ResNet18PoseModel(cfg.NUM_KEYPOINTS, in_channels=2)),
        ('fusion',     lambda: FusionModel(cfg.NUM_KEYPOINTS)),
    ]:
        print(f'\n--- Evaluating {mname} ---')
        mod = ModelClass()
        mod = load_checkpoint(mod, mname).to(DEVICE)
        eval_results[mname] = evaluate_model(mod, test_loader, DEVICE, mname)

    # Adverse condition evaluation
    print("\n--- Adverse Condition Evaluation (Fusion) ---")
    fus_eval = FusionModel(cfg.NUM_KEYPOINTS)
    fus_eval = load_checkpoint(fus_eval, 'fusion').to(DEVICE)

    adverse = {}
    for condition in ['noise', 'dropout', 'low_power']:
        print(f'\nEvaluating under: {condition}')
        adv_ds     = AdverseConditionDataset(test_loader.dataset, condition)
        adv_loader = DataLoader(adv_ds, batch_size=cfg.BATCH_SIZE,
                                shuffle=False, num_workers=0)
        adverse[condition] = evaluate_model(fus_eval, adv_loader, DEVICE,
                                            f'fusion_{condition}')

    # Print summary
    print_summary_table(eval_results)

    print('\nAdverse Condition Results (Fusion Model — radar perturbations):')
    print(f"{'Condition':<12} {'PCK@5%':>9} {'OKS':>9} {'MAE(px)':>9} {'F1':>9}")
    print('-' * 50)
    for cond, r in adverse.items():
        print(f"{cond:<12} "
              f"{r['PCK@0.05']*100:>8.2f}% "
              f"{r['OKS']:>9.4f} "
              f"{r['MAE_px']:>9.2f} "
              f"{r['F1']:>9.4f}")

    save_eval_results(eval_results)
    print('\nEvaluation complete.')


def cmd_visualize():
    """Generate all visualization plots."""
    from training.checkpoint import load_histories, load_eval_results
    from visualization.loss_plots import plot_loss_curves
    from visualization.comparison import plot_model_comparison, plot_pck_per_keypoint

    print("\n" + "="*60)
    print(" GENERATING VISUALIZATIONS")
    print("="*60)

    histories    = load_histories()
    eval_results = load_eval_results()

    if histories:
        plot_loss_curves(histories)

    if eval_results:
        plot_model_comparison(eval_results)
        plot_pck_per_keypoint(eval_results)

    print('\nVisualization complete.')


def cmd_export():
    """Export final models for submission."""
    from training.checkpoint import export_final_models, check_checkpoints
    import os

    print("\n" + "="*60)
    print(" EXPORTING MODELS")
    print("="*60)

    check_checkpoints()
    export_final_models(cfg)

    print('\nFinal output files:')
    if os.path.exists(cfg.RESULTS_DIR):
        for f in sorted(os.listdir(cfg.RESULTS_DIR)):
            size = os.path.getsize(os.path.join(cfg.RESULTS_DIR, f))
            print(f'  ./results/{f}  ({size/1024:.1f} KB)')


def main():
    args = parse_args()

    print("="*60)
    print(" MMVR Pose Detection Pipeline")
    print(f" Device : {DEVICE}")
    print(f" Command: {args.command}")
    print("="*60)

    apply_overrides(args)

    COMMANDS = {
        'explore':   cmd_explore,
        'train':     cmd_train,
        'evaluate':  cmd_evaluate,
        'visualize': cmd_visualize,
        'export':    cmd_export,
    }

    if args.command == 'all':
        for name in ['train', 'evaluate', 'visualize', 'export']:
            COMMANDS[name]()
    else:
        COMMANDS[args.command]()


if __name__ == '__main__':
    main()
