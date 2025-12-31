"""
Data loading and preprocessing module for CIFAR-10 classification.

This module handles:
- Image transformations (resize, augmentation, normalization)
- CIFAR-10 dataset loading with automatic download
- DataLoader creation for training and validation

Key Design Decisions:
- Images resized to 224x224 to match EfficientNet's expected input size
- ImageNet normalization stats used (model was pretrained on ImageNet)
- Data augmentation applied only to training set

Author: Carlos Manuel
Project: CNN-EfficientNet Transfer Learning
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    """
    Configuration for data loading pipeline.
    
    Attributes:
        data_dir: Directory where CIFAR-10 will be downloaded/stored.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        image_size: Target size for resizing images (height = width).
    """
    data_dir: str = "data"
    batch_size: int = 64
    num_workers: int = 2
    image_size: int = 224  # EfficientNet expects 224x224 input


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Build image transformation pipelines for training and validation.
    
    Training transforms include data augmentation to improve generalization:
    - Random horizontal flip (50% probability)
    - Random rotation (±10 degrees)
    
    Both pipelines include:
    - Resize to target dimensions (CIFAR-10 is 32x32, EfficientNet needs 224x224)
    - Convert PIL image to tensor
    - Normalize with ImageNet statistics (required for pretrained weights)
    
    Args:
        image_size: Target height and width for resized images.
    
    Returns:
        tuple: (train_transforms, val_transforms) composition objects.
    """
    # Training: apply data augmentation for better generalization
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Augmentation: horizontal flip
            transforms.RandomRotation(10),           # Augmentation: slight rotation
            transforms.ToTensor(),                   # Convert to tensor [0, 1]
            # Normalize with ImageNet stats (pretrained model expects this)
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Validation: no augmentation, just resize and normalize
    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return train_tfms, val_tfms


def create_cifar10_loaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 data loaders for training and validation.
    
    CIFAR-10 Dataset Info:
    - 60,000 32x32 color images in 10 classes
    - 50,000 training images, 10,000 test images
    - Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    
    Args:
        cfg: DataConfig object with loading parameters.
    
    Returns:
        tuple: (train_loader, val_loader) DataLoader objects.
    """
    train_tfms, val_tfms = build_transforms(cfg.image_size)

    # Download CIFAR-10 if not present, apply transforms
    train_ds = datasets.CIFAR10(
        root=cfg.data_dir, 
        train=True, 
        download=True, 
        transform=train_tfms
    )
    val_ds = datasets.CIFAR10(
        root=cfg.data_dir, 
        train=False,  # Use test set as validation
        download=True, 
        transform=val_tfms
    )

    # Create DataLoaders with optimized settings
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),  # Speed up CPU->GPU transfer
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
