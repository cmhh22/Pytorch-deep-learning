"""
Main script for training, validating and testing an MLP on MNIST.

Flow:
1. Load MNIST data
2. Create MLP model
3. Train for multiple epochs
4. Validate after each epoch
5. Save best model
6. Evaluate on test set
"""

import torch
import torch.optim as optim
import sys
import os
from pathlib import Path

# Add paths to system
sys.path.insert(0, str(Path(__file__).parent))

from models.mlp import MLP
from src.utils import (
    get_device,
    load_mnist_data,
    train_epoch,
    validate,
    test,
    save_model,
    load_model
)


def main():
    """Main function of the program."""
    
    print("=" * 60)
    print("MLP TRAINING ON MNIST WITH PYTORCH")
    print("=" * 60)
    
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    
    # Get script directory to ensure correct paths
    script_dir = Path(__file__).parent
    MODEL_PATH = script_dir / "models" / "mlp_mnist.pth"
    
    # Create directory to save models
    (script_dir / "models").mkdir(exist_ok=True)
    
    # 1. Get device
    device = get_device()
    print()
    
    # 2. Load data
    print("\nLOADING DATA...")
    train_loader, val_loader, test_loader = load_mnist_data(
        data_dir=script_dir / "data",
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    print()
    
    # 3. Create model
    print("CREATING MODEL...")
    model = MLP(
        input_size=784,      # 28x28
        hidden1=512,
        hidden2=256,
        num_classes=10
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MLP model created with {total_params:,} parameters")
    print(model)
    print()
    
    # 4. Define optimizer and criterion
    print("CONFIGURING OPTIMIZATION...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Optimizer: Adam")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Loss function: Cross Entropy")
    print()
    
    # 5. Train
    print("STARTING TRAINING...")
    print("=" * 60)
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEPOCH {epoch}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Show summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"   Training   -> Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"   Validation -> Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, MODEL_PATH)
            print(f"   New best model! (Accuracy: {best_val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    
    # 6. Load best model and evaluate on test
    print("\nEVALUATING ON TEST SET...")
    model = load_model(model, MODEL_PATH, device)
    
    test_acc, test_loss = test(model, test_loader, device)
    print(f"\nFINAL RESULTS:")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Show statistics
    print(f"\nGENERAL STATISTICS:")
    print(f"   Best validation accuracy: {max(val_accs):.2f}%")
    print(f"   Final training accuracy: {train_accs[-1]:.2f}%")
    print(f"   Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"   Test accuracy: {test_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
