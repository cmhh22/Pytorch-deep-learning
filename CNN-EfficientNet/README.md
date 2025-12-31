# 🧠 CNN Transfer Learning with EfficientNet-B0

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Educational project** demonstrating **transfer learning** with a state-of-the-art CNN architecture on the CIFAR-10 dataset.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [What is Transfer Learning?](#-what-is-transfer-learning)
- [EfficientNet Architecture](#-efficientnet-architecture)
- [Dataset: CIFAR-10](#-dataset-cifar-10)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Learning Guide](#-learning-guide)
- [Results](#-results)
- [References](#-references)

---

## 🎯 Overview

This project teaches you how to:

1. **Use a pretrained CNN** (EfficientNet-B0) for a new classification task
2. **Fine-tune vs Feature Extraction** — two transfer learning strategies
3. **Implement a complete training pipeline** with PyTorch
4. **Analyze model performance** with confusion matrices and Grad-CAM

### Why This Project?

| Skill | What You'll Learn |
|-------|-------------------|
| 🏗️ Architecture | How modern CNNs (EfficientNet) are structured |
| 🔄 Transfer Learning | Reuse ImageNet features for new tasks |
| 📊 Evaluation | Confusion matrix, per-class metrics, error analysis |
| 🔍 Interpretability | Grad-CAM to visualize what the model "sees" |

---

## 🔄 What is Transfer Learning?

**Transfer learning** means taking a model trained on one task and adapting it to a different task.

### The Intuition

Imagine you learned to play piano. When you try to learn guitar, you don't start from zero — you already understand music theory, rhythm, and hand coordination. Transfer learning works the same way for neural networks!

### How It Works

```
ImageNet (1.2M images, 1000 classes)
         ↓
    Pretrained CNN
    (learned general features: edges, textures, shapes)
         ↓
    Replace final layer
    (1000 classes → 10 classes for CIFAR-10)
         ↓
    Fine-tune on new dataset
```

### Two Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Feature Extraction** | Freeze backbone, train only classifier | Small dataset, limited compute |
| **Fine-Tuning** | Train all layers (smaller learning rate) | More data, need higher accuracy |

In this project, use `--freeze-backbone` for feature extraction or omit it for fine-tuning.

---

## 🏛️ EfficientNet Architecture

**EfficientNet** was developed by Google in 2019 and achieves state-of-the-art accuracy with fewer parameters than previous architectures.

### Key Innovation: Compound Scaling

Previous CNNs scaled only one dimension (depth OR width OR resolution). EfficientNet scales all three together using a compound coefficient:

```
depth:      d = α^φ
width:      w = β^φ  
resolution: r = γ^φ

where α·β²·γ² ≈ 2 (to roughly double FLOPs)
```

### EfficientNet Family

| Model | Parameters | Top-1 Acc (ImageNet) | Input Size |
|-------|------------|----------------------|------------|
| **B0** | 5.3M | 77.1% | 224×224 |
| B1 | 7.8M | 79.1% | 240×240 |
| B2 | 9.2M | 80.1% | 260×260 |
| B3 | 12M | 81.6% | 300×300 |
| B7 | 66M | 84.3% | 600×600 |

We use **B0** because it's lightweight and perfect for learning!

### Architecture Diagram

```
Input (224×224×3)
    ↓
┌─────────────────────────────────────┐
│  Stem: Conv3×3 + BatchNorm + SiLU   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  MBConv Blocks (×16)                │
│  - Depthwise Separable Convolution  │
│  - Squeeze-and-Excitation (SE)      │
│  - Skip connections                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Head: Conv1×1 + GlobalAvgPool      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Classifier: Dropout + Linear       │  ← We replace this!
│  (1280 → 10 classes)                │
└─────────────────────────────────────┘
    ↓
Output (10 class probabilities)
```

---

## 📦 Dataset: CIFAR-10

### Overview

| Property | Value |
|----------|-------|
| **Images** | 60,000 color images |
| **Size** | 32×32 pixels (we resize to 224×224) |
| **Classes** | 10 |
| **Train/Test** | 50,000 / 10,000 |
| **Balance** | Perfectly balanced (6,000 per class) |

### Classes

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | ✈️ Airplane | 5 | 🐕 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚚 Truck |

### Where to Download?

**You don't need to download manually!** 🎉

The dataset is automatically downloaded by `torchvision` when you run training:

```python
from torchvision import datasets
datasets.CIFAR10(root='data', train=True, download=True)
```

Files are saved to:
```
data/
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    ├── batches.meta
    └── readme.html
```

**Manual download** (if needed): https://www.cs.toronto.edu/~kriz/cifar.html

---

## 📁 Project Structure

```
cnn-efficientnet/
├── main.py              # 🚀 CLI entry point (train/eval commands)
├── test_quick.py        # 🧪 Smoke test (verify everything works)
├── requirements.txt     # 📦 Dependencies
├── README.md            # 📖 This file
│
├── models/
│   ├── __init__.py
│   └── model.py         # 🏗️ EfficientNet builder + transfer learning config
│
├── src/
│   ├── __init__.py
│   ├── data.py          # 📊 CIFAR-10 loaders + augmentation
│   ├── engine.py        # 🔄 Training and evaluation loops
│   └── utils.py         # 🛠️ Checkpoints, metrics, reproducibility
│
├── notebooks/
│   ├── train_colab.ipynb            # 🚀 Training notebook (Google Colab GPU)
│   └── analysis_post_training.ipynb # 📈 Confusion matrix + Grad-CAM
│
└── data/                # 📂 CIFAR-10 (auto-downloaded)
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# 1. Clone or navigate to the project
cd cnn-efficientnet

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.2          # Deep learning framework
torchvision>=0.17   # Pretrained models + datasets
numpy>=1.24         # Numerical computing
tqdm>=4.66          # Progress bars
```

For the analysis notebook, also install:
```bash
pip install matplotlib seaborn scikit-learn
```

---

## 🚀 Usage

### Quick Test (Verify Setup)

```bash
python test_quick.py
```

Expected output:
```
✓ Quick train OK | loss=X.XXXX acc=X.XXXX
All components working correctly!
```

### Train Model

**Feature Extraction** (fast, ~90% accuracy):
```bash
python main.py train --epochs 5 --freeze-backbone
```

**Fine-Tuning** (slower, higher accuracy):
```bash
python main.py train --epochs 10 --lr 1e-4
```

**Full options**:
```bash
python main.py train \
    --epochs 10 \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --freeze-backbone \
    --scheduler cosine \
    --patience 5 \
    --out models/my_model.pth
```

**Advanced options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--scheduler` | `cosine` | Learning rate scheduler: `cosine`, `plateau`, or `none` |
| `--patience` | `5` | Early stopping patience (epochs without improvement) |
| `--no-early-stop` | `False` | Disable early stopping |

### Evaluate Model

```bash
python main.py eval --checkpoint models/efficientnet_cifar10_best.pth
```

### Run Analysis Notebook

```bash
jupyter notebook notebooks/analysis_post_training.ipynb
```

### 🚀 Train on Google Colab (Recommended for GPU)

If you don't have a dedicated NVIDIA GPU, use Google Colab for faster training:

1. Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Google Colab
2. Go to **Runtime > Change runtime type > GPU**
3. Run all cells

**Expected speedup: ~40x faster than CPU!**

| Environment | Time per Epoch |
|-------------|----------------|
| CPU (Intel i5) | ~50 minutes |
| GPU (Colab T4) | ~1-2 minutes |

---

## 📖 Learning Guide

### Recommended Learning Path

1. **Read this README** — Understand the concepts
2. **Run `test_quick.py`** — See the pipeline in action
3. **Read `models/model.py`** — Understand transfer learning setup
4. **Read `src/data.py`** — Learn about data augmentation
5. **Read `src/engine.py`** — Study the training loop
6. **Train a model** — Experiment with hyperparameters
7. **Run the notebook** — Analyze your trained model

### Key Concepts to Understand

| Concept | File | Lines to Study |
|---------|------|----------------|
| Transfer learning setup | `models/model.py` | `build_efficientnet_b0()` |
| Freezing layers | `models/model.py` | `set_trainable()` |
| Data augmentation | `src/data.py` | `build_transforms()` |
| ImageNet normalization | `src/data.py` | `Normalize(mean, std)` |
| Training loop | `src/engine.py` | `train_one_epoch()` |
| Evaluation mode | `src/engine.py` | `model.eval()` |

### Experiments to Try

1. **Compare strategies**: Train with and without `--freeze-backbone`
2. **Learning rate**: Try `1e-3`, `3e-4`, `1e-4`, `1e-5`
3. **Batch size**: Compare `32` vs `64` vs `128`
4. **Epochs**: Watch for overfitting with more epochs
5. **Augmentation**: Modify `build_transforms()` in `src/data.py`

---

## 📊 Results

### Expected Performance

| Strategy | Epochs | Val Accuracy | Training Time* |
|----------|--------|--------------|----------------|
| Feature Extraction | 5 | ~85-90% | ~5 min (GPU) |
| Fine-Tuning | 10 | ~92-95% | ~15 min (GPU) |

*On NVIDIA RTX 3060 or similar

### Sample Training Output

```
Using device: cuda
Train batches: 782, Val batches: 157
Trainable parameters: 12,810 (freeze) or 4,020,618 (full)

Epoch 01/05 | Train Loss: 0.8234, Acc: 0.7123 | Val Loss: 0.4521, Acc: 0.8456
  ↳ New best model saved! (acc=0.8456)
Epoch 02/05 | Train Loss: 0.4123, Acc: 0.8567 | Val Loss: 0.3892, Acc: 0.8721
  ↳ New best model saved! (acc=0.8721)
...
Training complete! Best validation accuracy: 0.8912
```

---

## 📚 References

### Papers

1. **EfficientNet**: Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for CNNs" 
   - [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

2. **CIFAR-10**: Krizhevsky (2009). "Learning Multiple Layers of Features from Tiny Images"
   - [Technical Report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

3. **Grad-CAM**: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks"
   - [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

### Tutorials

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [EfficientNet in TorchVision](https://pytorch.org/vision/stable/models/efficientnet.html)

---

## 📄 License

MIT License — feel free to use this project for learning!

---

<div align="center">

*Part of the `pytorch-deep-learning` portfolio series*

</div>
