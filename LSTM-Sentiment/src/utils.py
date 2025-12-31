"""
Utility Functions for LSTM Sentiment Analysis

This module provides helper functions for:
- Reproducibility (seed setting)
- Device detection
- Checkpoint saving/loading
- Visualization (training curves, attention)
- Text prediction
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Save path
        **kwargs: Additional items to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Load a training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}" if 'loss' in checkpoint else "")
    
    return checkpoint


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    if 'val_f1' in history:
        axes[1].plot(epochs, history['val_f1'], 'g--', label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training and Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()


def visualize_attention(
    tokens: List[str],
    attention_weights: np.ndarray,
    prediction: str,
    confidence: float,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights over tokens.
    
    Args:
        tokens: List of tokens
        attention_weights: Attention weights array
        prediction: Model prediction ('Positive' or 'Negative')
        confidence: Prediction confidence
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 3))
    
    # Normalize attention weights
    weights = attention_weights[:len(tokens)]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    # Create color-coded display
    colors = plt.cm.Reds(weights)
    
    # Plot tokens with attention coloring
    x_positions = np.arange(len(tokens))
    
    for i, (token, weight, color) in enumerate(zip(tokens, weights, colors)):
        ax.text(
            i, 0.5, token,
            ha='center', va='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, edgecolor='none'),
            rotation=45 if len(token) > 5 else 0
        )
    
    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    color = 'green' if prediction == 'Positive' else 'red'
    ax.set_title(
        f"Prediction: {prediction} (confidence: {confidence:.2%})",
        fontsize=12, fontweight='bold', color=color
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape: Tuple[int, ...] = None) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for forward pass
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        print(f"{name}: {list(param.shape)} = {param_count:,} params")
    
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    device = get_device()
    
    # Test AverageMeter
    meter = AverageMeter("Loss")
    for i in range(10):
        meter.update(random.random())
    print(f"Average meter test: {meter}")
    
    print("\n✓ Utils module loaded successfully!")
