"""
Utility functions for CNN Transfer Learning project.

This module provides:
- Reproducibility helpers (seed_everything)
- Device management (get_device)
- Metrics (accuracy_top1)
- Checkpoint save/load functionality

Author: Carlos Manuel
Project: CNN-EfficientNet Transfer Learning
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This ensures that experiments can be reproduced by fixing:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU and CUDA random generators
    
    Args:
        seed: Integer seed value for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: CUDA device if GPU is available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy from model logits and ground truth targets.
    
    Args:
        logits: Raw model outputs of shape (batch_size, num_classes).
        targets: Ground truth class indices of shape (batch_size,).
    
    Returns:
        float: Accuracy as a fraction between 0 and 1.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


@dataclass
class Checkpoint:
    """
    Data class to store all information needed to resume training.
    
    Attributes:
        epoch: The epoch number when this checkpoint was saved.
        model_state: The model's state_dict (weights and biases).
        optimizer_state: The optimizer's state_dict (momentum, etc.).
        best_val_acc: Best validation accuracy achieved so far.
        config: Dictionary of hyperparameters used for training.
    """
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    best_val_acc: float
    config: dict[str, Any]


def save_checkpoint(path: str, checkpoint: Checkpoint) -> None:
    """
    Save a training checkpoint to disk.
    
    Creates parent directories if they don't exist.
    
    Args:
        path: File path where the checkpoint will be saved (.pth).
        checkpoint: Checkpoint dataclass containing all training state.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": checkpoint.epoch,
            "model_state": checkpoint.model_state,
            "optimizer_state": checkpoint.optimizer_state,
            "best_val_acc": checkpoint.best_val_acc,
            "config": checkpoint.config,
        },
        path,
    )


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """
    Load a training checkpoint from disk.
    
    Args:
        path: File path to the saved checkpoint (.pth).
        map_location: Device to map tensors to (useful for CPU/GPU transfer).
    
    Returns:
        dict: Dictionary containing all checkpoint data.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


class EarlyStopping:
    """
    Early stopping to prevent overfitting during training.
    
    Monitors a metric (e.g., validation accuracy) and stops training
    if no improvement is seen for a specified number of epochs.
    
    Attributes:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        counter: Current number of epochs without improvement.
        best_score: Best metric value observed so far.
        should_stop: Flag indicating whether training should stop.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        """
        Initialize early stopping.
        
        Args:
            patience: Epochs to wait without improvement before stopping.
            min_delta: Minimum improvement to reset patience counter.
            mode: 'max' for metrics to maximize (accuracy), 'min' for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop based on the current score.
        
        Args:
            score: Current metric value to evaluate.
        
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check for improvement based on mode
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:  # mode == "min"
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self) -> None:
        """Reset early stopping state for new training run."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
