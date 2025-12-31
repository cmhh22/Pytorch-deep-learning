"""
EfficientNet-B0 model builder for transfer learning.

Transfer Learning Strategy:
1. Load EfficientNet-B0 pretrained on ImageNet (1000 classes)
2. Replace the final classifier layer for our target task (10 classes)
3. Optionally freeze backbone to train only the classifier (faster, less data needed)

EfficientNet Architecture Overview:
- Developed by Google (2019), uses Neural Architecture Search (NAS)
- Compound scaling: balances depth, width, and resolution
- B0 is the smallest variant (~5.3M parameters, ~4MB model size)
- Achieves better accuracy than ResNet-50 with 8x fewer parameters

Author: Carlos Manuel
Project: CNN-EfficientNet Transfer Learning
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B0 model with custom classifier head.
    
    The original EfficientNet-B0 has a classifier for 1000 ImageNet classes.
    We replace it with a new Linear layer for our target number of classes.
    
    Architecture of classifier:
    - classifier[0]: Dropout(p=0.2)
    - classifier[1]: Linear(1280 -> num_classes) <- We replace this
    
    Args:
        num_classes: Number of output classes (10 for CIFAR-10).
        pretrained: If True, load ImageNet pretrained weights.
    
    Returns:
        nn.Module: Modified EfficientNet-B0 model ready for fine-tuning.
    """
    # Load pretrained weights or start from scratch
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Replace final classification layer
    # EfficientNet-B0 has 1280 features before the classifier
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def set_trainable(model: nn.Module, train_classifier_only: bool) -> None:
    """
    Configure which layers should be trained (have gradients computed).
    
    Transfer Learning Strategies:
    1. train_classifier_only=True (Feature Extraction):
       - Freeze all backbone layers (convolutional layers)
       - Only train the new classifier head
       - Faster training, works well with small datasets
       - Preserves learned ImageNet features
    
    2. train_classifier_only=False (Fine-Tuning):
       - Train all layers including backbone
       - Can achieve higher accuracy with enough data
       - Risk of overfitting on small datasets
       - Slower training
    
    Args:
        model: The EfficientNet model to configure.
        train_classifier_only: If True, freeze backbone weights.
    """
    if not train_classifier_only:
        # Fine-tuning: all parameters are trainable
        for p in model.parameters():
            p.requires_grad = True
        return

    # Feature extraction: freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Then unfreeze only the classifier head
    for p in model.classifier.parameters():
        p.requires_grad = True


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    """
    Get list of parameters that require gradients.
    
    Useful for:
    - Passing to optimizer (only optimize trainable params)
    - Counting trainable parameters for logging
    
    Args:
        model: PyTorch model to inspect.
    
    Returns:
        list: Parameters with requires_grad=True.
    """
    return [p for p in model.parameters() if p.requires_grad]
