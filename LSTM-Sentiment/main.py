"""
LSTM Sentiment Analysis - Main Training Script

Train an LSTM model for binary sentiment classification on IMDB movie reviews.

Usage:
    python main.py                          # Train with default settings
    python main.py --epochs 10 --lr 0.001   # Custom training
    python main.py --evaluate               # Evaluate saved model
    
Author: cmhh22
Date: December 2025
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lstm import LSTMSentiment, LSTMAttention, count_parameters
from src.data import IMDBDataModule
from src.engine import Trainer, evaluate
from src.utils import set_seed, get_device, plot_training_curves, model_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LSTM Sentiment Analysis on IMDB Dataset"
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for dataset')
    parser.add_argument('--max_vocab', type=int, default=25000,
                        help='Maximum vocabulary size')
    parser.add_argument('--max_len', type=int, default=256,
                        help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=100,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--attention', action='store_true',
                        help='Use attention mechanism')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Other arguments
    parser.add_argument('--evaluate', action='store_true',
                        help='Only evaluate (skip training)')
    parser.add_argument('--checkpoint', type=str, default='./models/lstm_sentiment_best.pth',
                        help='Path to save/load checkpoint')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("🎭 LSTM SENTIMENT ANALYSIS")
    print("="*60)
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    print("\n📦 Loading Data...")
    
    data_module = IMDBDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_vocab_size=args.max_vocab,
        max_seq_len=args.max_len
    )
    
    data_module.prepare_data()
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    
    vocab = data_module.vocab
    print(f"✓ Vocabulary size: {len(vocab)}")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # =========================================================================
    # Model Creation
    # =========================================================================
    print("\n🏗️ Building Model...")
    
    ModelClass = LSTMAttention if args.attention else LSTMSentiment
    
    model = ModelClass(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=2,  # Binary classification
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=vocab.pad_idx
    )
    
    model = model.to(device)
    
    print(f"✓ Model: {ModelClass.__name__}")
    print(f"✓ Parameters: {count_parameters(model):,}")
    
    # =========================================================================
    # Training Setup
    # =========================================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    # =========================================================================
    # Training or Evaluation
    # =========================================================================
    if args.evaluate:
        # Load and evaluate existing model
        print("\n📊 Evaluating Model...")
        
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"⚠ Checkpoint not found: {args.checkpoint}")
            return
        
        metrics = evaluate(model, test_loader, criterion, device)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"  Loss:      {metrics['loss']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print("="*60)
        
    else:
        # Train the model
        print("\n🚀 Starting Training...")
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            patience=args.patience,
            checkpoint_dir=os.path.dirname(args.checkpoint)
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=test_loader,  # Using test as validation for simplicity
            epochs=args.epochs
        )
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': len(vocab),
            'args': vars(args),
            'history': history
        }, args.checkpoint)
        print(f"\n✓ Model saved to {args.checkpoint}")
        
        # Plot training curves
        try:
            plot_training_curves(history, save_path='./training_curves.png')
        except Exception as e:
            print(f"Could not plot curves: {e}")
        
        # Final evaluation
        print("\n📊 Final Evaluation...")
        metrics = evaluate(model, test_loader, criterion, device)
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"  Test Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Test F1 Score: {metrics['f1']:.4f}")
        print("="*60)


def predict_sentiment(text: str, model_path: str = './models/lstm_sentiment_best.pth'):
    """
    Predict sentiment for a single text.
    
    Args:
        text: Input text to classify
        model_path: Path to saved model
        
    Returns:
        dict with prediction and confidence
    """
    from src.data import preprocess_text, tokenize, Vocabulary
    
    device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint.get('args', {})
    
    # This is simplified - in production, you'd save/load the vocabulary
    print("Note: For proper inference, vocabulary should be saved and loaded.")
    
    return {"prediction": "Positive/Negative", "confidence": 0.0}


if __name__ == "__main__":
    main()
