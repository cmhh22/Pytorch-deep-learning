"""
Training and Evaluation Engine for LSTM Sentiment Analysis

This module provides:
- Training loop with progress tracking
- Evaluation metrics computation
- Trainer class with early stopping
"""

import time
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        Dictionary with training metrics (loss, accuracy)
    """
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in pbar:
        texts, labels, lengths = batch
        texts = texts.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Evaluation DataLoader
        criterion: Loss function
        device: Device for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    for batch in pbar:
        texts, labels, lengths = batch
        texts = texts.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class Trainer:
    """
    Training manager with early stopping and checkpointing.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        device: Device for training
        scheduler: Learning rate scheduler (optional)
        patience: Early stopping patience
        checkpoint_dir: Directory for saving checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        patience: int = 3,
        checkpoint_dir: str = "./models"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = train_one_epoch(
                self.model, train_loader, self.optimizer,
                self.criterion, self.device, self.scheduler
            )
            
            # Validation
            val_metrics = evaluate(
                self.model, val_loader, self.criterion, self.device
            )
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Print progress
            print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self._save_checkpoint('best_model.pth')
                print(f"  ✓ New best model saved!")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
                
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                    break
            
            print()
        
        total_time = time.time() - start_time
        print(f"{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        import os
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']


@torch.no_grad()
def predict(
    model: nn.Module,
    texts: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions on input texts.
    
    Args:
        model: Trained model
        texts: Input tensor (batch_size, seq_len)
        device: Device for inference
        
    Returns:
        predictions: Class predictions (batch_size,)
        probabilities: Class probabilities (batch_size, num_classes)
    """
    model.eval()
    texts = texts.to(device)
    
    outputs = model(texts)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = outputs.argmax(dim=1)
    
    return predictions, probabilities


if __name__ == "__main__":
    # Quick test
    print("Engine module loaded successfully!")
    print("Available functions: train_one_epoch, evaluate, Trainer, predict")
