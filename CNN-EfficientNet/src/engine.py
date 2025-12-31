"""
Training and evaluation engine for CNN models.

This module provides the core training loop functions:
- train_one_epoch: Single epoch training with gradient updates
- evaluate: Model evaluation without gradient computation

Design Pattern:
- Functions are model-agnostic (work with any nn.Module)
- Return structured results via dataclass
- Use tqdm for progress visualization

Author: Carlos Manuel
Project: CNN-EfficientNet Transfer Learning
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import accuracy_top1


@dataclass
class EpochResult:
    """
    Container for epoch-level training/evaluation metrics.
    
    Attributes:
        loss: Average loss over all batches in the epoch.
        acc: Average top-1 accuracy over all batches.
    """
    loss: float
    acc: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochResult:
    """
    Execute one complete training epoch.
    
    Training Loop Steps (per batch):
    1. Move data to device (GPU/CPU)
    2. Zero gradients from previous iteration
    3. Forward pass: compute predictions
    4. Compute loss between predictions and targets
    5. Backward pass: compute gradients
    6. Optimizer step: update weights
    
    Args:
        model: Neural network model to train.
        loader: DataLoader providing training batches.
        optimizer: Optimizer for weight updates (e.g., AdamW).
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        device: Device to run computations on.
    
    Returns:
        EpochResult: Average loss and accuracy for the epoch.
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)

    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        # Move batch to device (non_blocking for async transfer with pin_memory)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero gradients - set_to_none=True is slightly faster than zero_grad()
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        logits = model(images)
        loss = loss_fn(logits, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_acc += accuracy_top1(logits.detach(), targets)
        batches += 1

    # Return average metrics (avoid division by zero)
    return EpochResult(
        loss=total_loss / max(batches, 1), 
        acc=total_acc / max(batches, 1)
    )


@torch.no_grad()  # Decorator disables gradient computation for efficiency
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochResult:
    """
    Evaluate model on a dataset without gradient computation.
    
    Key Differences from Training:
    - model.eval() disables dropout, freezes batch norm statistics
    - @torch.no_grad() prevents gradient computation (saves memory)
    - No optimizer step (weights are not updated)
    
    Args:
        model: Neural network model to evaluate.
        loader: DataLoader providing evaluation batches.
        loss_fn: Loss function for computing validation loss.
        device: Device to run computations on.
    
    Returns:
        EpochResult: Average loss and accuracy for the dataset.
    """
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass only (no backward)
        logits = model(images)
        loss = loss_fn(logits, targets)

        total_loss += loss.item()
        total_acc += accuracy_top1(logits, targets)
        batches += 1

    return EpochResult(
        loss=total_loss / max(batches, 1), 
        acc=total_acc / max(batches, 1)
    )
