"""
Module defining the Multi-Layer Perceptron (MLP) architecture
for MNIST digit classification.

An MLP is a simple but effective neural network that contains:
- Input layer: 784 neurons (28x28 MNIST pixels)
- Hidden layers: With ReLU activation functions
- Output layer: 10 neurons (digits 0-9)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MNIST classification.
    
    Architecture:
    - Input: 784 (28*28)
    - Hidden1: 512 neurons
    - Hidden2: 256 neurons
    - Output: 10 (classes)
    
    Activation function: ReLU (Rectified Linear Unit)
    - Passes positive values, converts negatives to 0
    - Avoids vanishing gradient problem
    """
    
    def __init__(self, input_size=784, hidden1=512, hidden2=256, num_classes=10):
        """
        Initialize the MLP model.
        
        Args:
            input_size (int): Number of input features (784 for MNIST)
            hidden1 (int): Number of neurons in first hidden layer
            hidden2 (int): Number of neurons in second hidden layer
            num_classes (int): Number of output classes (10 for digits)
        """
        super(MLP, self).__init__()
        
        # Linear layers (fully connected)
        self.fc1 = nn.Linear(input_size, hidden1)      # 784 -> 512
        self.fc2 = nn.Linear(hidden1, hidden2)         # 512 -> 256
        self.fc3 = nn.Linear(hidden2, num_classes)     # 256 -> 10
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Dropout for regularization (reduces overfitting)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        """
        Forward propagation.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 784)
            
        Returns:
            torch.Tensor: Output logits with shape (batch_size, 10)
        """
        # First layer: 784 -> 512
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second layer: 512 -> 256
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer: 256 -> 10
        x = self.fc3(x)
        
        return x
