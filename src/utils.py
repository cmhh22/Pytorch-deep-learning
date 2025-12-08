"""
Utility module for training, validation and evaluation of the MLP model.

Contains functions for:
- Loading the MNIST dataset
- Training the model
- Validating and evaluating performance
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple


def get_device():
    """
    Get the available device (GPU if available, otherwise CPU).
    
    Returns:
        torch.device: Device to use
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def load_mnist_data(batch_size: int = 64, num_workers: int = 0, data_dir = './data') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset with normalized transformations.
    
    MNIST contains:
    - 60,000 training images
    - 10,000 test images
    - Size: 28x28 grayscale pixels (values 0-255)
    
    Normalization: (pixel - mean) / std
    - Mean: 0.1307 (computed over entire MNIST)
    - Std: 0.3081
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        data_dir: Directory to store/load MNIST data
        
    Returns:
        Tuple: (train_loader, val_loader, test_loader)
    """
    
    # Transformations applied to images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to Tensor (0-1)
        transforms.Normalize(
            mean=[0.1307],      # MNIST mean
            std=[0.3081]        # MNIST standard deviation
        )
    ])
    
    print("Downloading MNIST dataset...")
    
    # Load training set
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test set
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )
    
    # Split train into train (85%) and validation (15%)
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Dataset loaded:")
    print(f"   - Training: {train_size} images")
    print(f"   - Validation: {val_size} images")
    print(f"   - Testing: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, device, epoch):
    """
    Train the model for one epoch.
    
    One epoch = passing the entire dataset through the model once.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        optimizer: Optimizer (Adam, SGD, etc.)
        device: Device (GPU/CPU)
        epoch: Current epoch number
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()  # Training mode
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Flatten images (28x28 -> 784)
        data = data.view(data.size(0), -1)
        
        # Forward pass
        outputs = model(data)
        
        # Compute loss (Cross Entropy combines LogSoftmax + NLLLoss)
        loss = F.cross_entropy(outputs, target)
        
        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Show progress
        if (batch_idx + 1) % 100 == 0:
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch}] Batch [{batch_idx + 1}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Accuracy: {accuracy:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, device):
    """
    Validate the model on the validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation DataLoader
        device: Device (GPU/CPU)
        
    Returns:
        Tuple: (average loss, accuracy)
    """
    model.eval()  # Evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't compute gradients
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def test(model, test_loader, device):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test DataLoader
        device: Device (GPU/CPU)
        
    Returns:
        Tuple: (accuracy, loss)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def save_model(model, filepath):
    """
    Save the trained model.
    
    Args:
        model: Model to save
        filepath (str): Path to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to: {filepath}")


def load_model(model, filepath, device):
    """
    Load a trained model.
    
    Args:
        model: Model architecture
        filepath (str): Path to the saved model
        device: Device to load onto
        
    Returns:
        Loaded model
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from: {filepath}")
    return model


def predict_single_image(model, image_tensor, device):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained model
        image_tensor: Image tensor (1, 784)
        device: Device
        
    Returns:
        Tuple: (prediction, probabilities)
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    return prediction.item(), probabilities[0].cpu().numpy()
