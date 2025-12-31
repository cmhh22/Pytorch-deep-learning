"""
Quick Smoke Test for CNN-EfficientNet Project.

This script verifies that all components work correctly by:
1. Loading CIFAR-10 data (small batch)
2. Building the model with pretrained weights
3. Running one complete training epoch

Purpose:
- CI/CD validation before full training
- Quick sanity check after code changes
- Verify dependencies are installed correctly

Run with: python test_quick.py

Expected output: "quick train OK | loss=X.XXXX acc=X.XXXX"

Author: Carlos Manuel
Project: CNN-EfficientNet Transfer Learning
"""
from __future__ import annotations

import torch

from models.model import build_efficientnet_b0, set_trainable
from src.data import DataConfig, create_cifar10_loaders
from src.engine import train_one_epoch


def main() -> None:
    """
    Run a minimal training iteration to verify the pipeline works.
    
    Uses CPU and small batch size to run quickly on any machine.
    This is NOT meant for actual training - just validation.
    """
    print("=" * 50)
    print("CNN-EfficientNet Quick Smoke Test")
    print("=" * 50)
    
    # Force CPU for quick testing (no GPU required)
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Minimal data config for fast execution
    cfg = DataConfig(
        data_dir="data", 
        batch_size=8,       # Small batch for speed
        num_workers=0,      # No multiprocessing (simpler)
        image_size=224
    )
    print("Loading CIFAR-10 dataset...")
    train_loader, _ = create_cifar10_loaders(cfg)
    print(f"Train batches: {len(train_loader)}")

    # Build model with frozen backbone (fewer params to update)
    print("Building EfficientNet-B0 model...")
    model = build_efficientnet_b0(num_classes=10, pretrained=True)
    set_trainable(model, train_classifier_only=True)  # Freeze backbone for speed
    model.to(device)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Setup training components
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-3
    )

    # Run one epoch as smoke test
    print("\nRunning 1 training epoch (smoke test)...")
    res = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    
    # Report results
    print("\n" + "=" * 50)
    print(f"✓ Quick train OK | loss={res.loss:.4f} acc={res.acc:.4f}")
    print("All components working correctly!")
    print("=" * 50)


if __name__ == "__main__":
    main()
