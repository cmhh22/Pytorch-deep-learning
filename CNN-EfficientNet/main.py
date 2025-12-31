"""
CNN Transfer Learning with EfficientNet-B0 on CIFAR-10.

This is the main entry point for training and evaluating the model.
Supports two commands:
    - train: Train the model with configurable hyperparameters
    - eval: Evaluate a saved checkpoint on the test set

Usage Examples:
    # Train with frozen backbone (fast, feature extraction)
    python main.py train --epochs 5 --freeze-backbone
    
    # Train with full fine-tuning (slower, potentially higher accuracy)
    python main.py train --epochs 10 --lr 1e-4
    
    # Evaluate a saved model
    python main.py eval --checkpoint models/efficientnet_cifar10_best.pth

Project Structure:
    main.py          - CLI and training orchestration (this file)
    models/model.py  - EfficientNet architecture and transfer learning setup
    src/data.py      - CIFAR-10 data loading and augmentation
    src/engine.py    - Training and evaluation loops
    src/utils.py     - Utilities (checkpoints, metrics, reproducibility)

Author: Carlos Manuel
Project: CNN-EfficientNet Transfer Learning
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from torch import nn

from models.model import build_efficientnet_b0, set_trainable, trainable_parameters
from src.data import DataConfig, create_cifar10_loaders
from src.engine import evaluate, train_one_epoch
from src.utils import Checkpoint, EarlyStopping, get_device, load_checkpoint, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training and evaluation.
    
    Returns:
        argparse.Namespace: Parsed arguments with all configuration options.
    """
    parser = argparse.ArgumentParser(
        description="CNN Transfer Learning with EfficientNet-B0 (CIFAR-10)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ==================== TRAIN SUBCOMMAND ====================
    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory for CIFAR-10 dataset (auto-downloads if missing)"
    )
    train_p.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs"
    )
    train_p.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for training and validation"
    )
    train_p.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate for AdamW optimizer"
    )
    train_p.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="L2 regularization strength"
    )
    train_p.add_argument(
        "--num-workers", type=int, default=2,
        help="Number of data loading workers"
    )
    train_p.add_argument(
        "--image-size", type=int, default=224,
        help="Input image size (EfficientNet expects 224)"
    )
    train_p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    train_p.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze backbone, train only classifier (faster, less overfitting)"
    )
    train_p.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (epochs without improvement)"
    )
    train_p.add_argument(
        "--no-early-stop", action="store_true",
        help="Disable early stopping"
    )
    train_p.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"],
        help="Learning rate scheduler: cosine, plateau, or none"
    )
    train_p.add_argument(
        "--out", type=str, default="models/efficientnet_cifar10_best.pth",
        help="Output path for best model checkpoint"
    )

    # ==================== EVAL SUBCOMMAND ====================
    eval_p = sub.add_parser("eval", help="Evaluate a saved checkpoint")
    eval_p.add_argument("--data-dir", type=str, default="data")
    eval_p.add_argument("--batch-size", type=int, default=64)
    eval_p.add_argument("--num-workers", type=int, default=2)
    eval_p.add_argument("--image-size", type=int, default=224)
    eval_p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth file)"
    )

    return parser.parse_args()


def cmd_train(args: argparse.Namespace) -> None:
    """
    Execute training pipeline.
    
    Training Flow:
    1. Set random seeds for reproducibility
    2. Create data loaders with augmentation
    3. Build model and configure trainable layers
    4. Training loop with validation after each epoch
    5. Save best model based on validation accuracy
    
    Args:
        args: Parsed command line arguments.
    """
    # Reproducibility
    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data loading
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    train_loader, val_loader = create_cifar10_loaders(data_cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model setup
    model = build_efficientnet_b0(num_classes=10, pretrained=True)
    set_trainable(model, train_classifier_only=bool(args.freeze_backbone))
    model.to(device)
    
    # Log trainable parameter count
    num_trainable = sum(p.numel() for p in trainable_parameters(model))
    print(f"Trainable parameters: {num_trainable:,}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        print(f"Using CosineAnnealingLR scheduler")
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler")
    else:
        print(f"No learning rate scheduler")

    # Early stopping
    early_stopping = None
    if not args.no_early_stop:
        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode="max")
        print(f"Early stopping enabled (patience={args.patience})")
    else:
        print(f"Early stopping disabled")

    # Training loop with best model tracking
    best_val_acc = -1.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train and evaluate
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va = evaluate(model, val_loader, loss_fn, device)

        # Update scheduler
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(va.acc)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {tr.loss:.4f}, Acc: {tr.acc:.4f} | "
            f"Val Loss: {va.loss:.4f}, Acc: {va.acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if va.acc > best_val_acc:
            best_val_acc = va.acc
            config: dict[str, Any] = {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "image_size": args.image_size,
                "freeze_backbone": bool(args.freeze_backbone),
                "seed": args.seed,
                "scheduler": args.scheduler,
                "patience": args.patience,
            }
            save_checkpoint(
                args.out,
                Checkpoint(
                    epoch=epoch,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    best_val_acc=best_val_acc,
                    config=config,
                ),
            )
            print(f"  ↳ New best model saved! (acc={best_val_acc:.4f})")

        # Check early stopping
        if early_stopping is not None and early_stopping(va.acc):
            print(f"\n⚠ Early stopping triggered at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\n{'='*50}")
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.out}")


def cmd_eval(args: argparse.Namespace) -> None:
    """
    Evaluate a saved model checkpoint on the test set.
    
    Args:
        args: Parsed command line arguments.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Data loading (only need validation loader)
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    _, val_loader = create_cifar10_loaders(data_cfg)

    # Load checkpoint and rebuild model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    
    model = build_efficientnet_b0(num_classes=10, pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    # Display checkpoint info
    print(f"Checkpoint from epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Best val acc in training: {ckpt.get('best_val_acc', 'N/A'):.4f}")

    # Evaluate
    loss_fn = nn.CrossEntropyLoss()
    res = evaluate(model, val_loader, loss_fn, device)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Loss: {res.loss:.4f}")
    print(f"  Accuracy: {res.acc:.4f} ({res.acc*100:.2f}%)")


def main() -> None:
    """Main entry point - parse args and dispatch to appropriate command."""
    args = parse_args()
    
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
