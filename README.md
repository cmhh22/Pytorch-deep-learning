# 🧠 PyTorch Deep Learning Portfolio

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Collection of PyTorch deep learning projects covering fundamental and advanced architectures: MLPs, CNNs, RNNs, LSTMs, and Transfer Learning.

---

## 📚 Projects

| Project | Architecture | Dataset | Accuracy | Description |
|---------|-------------|---------|----------|-------------|
| [**MLP-MNIST**](./MLP-MNIST) | Multi-Layer Perceptron | MNIST | ~98% | Handwritten digit classification with a 3-layer MLP |
| [**CNN-EfficientNet**](./CNN-EfficientNet) | EfficientNet-B0 (Transfer Learning) | CIFAR-10 | ~98% | Image classification using pretrained CNN |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/cmhh22/Pytorch-deep-learning.git
cd Pytorch-deep-learning

# Choose a project
cd MLP-MNIST     # or CNN-EfficientNet

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py train
```

---

## 📁 Repository Structure

```
Pytorch-deep-learning/
├── MLP-MNIST/              # 🔢 Multi-Layer Perceptron on MNIST
│   ├── models/
│   ├── src/
│   ├── notebooks/
│   ├── main.py
│   └── README.md
│
├── CNN-EfficientNet/       # 🖼️ Transfer Learning on CIFAR-10
│   ├── models/
│   ├── src/
│   ├── notebooks/
│   ├── main.py
│   └── README.md
│
├── LICENSE
└── README.md               # 📖 This file
```

---

## 🎯 Learning Path

### 1️⃣ Start with MLP-MNIST
- Understand basic neural network concepts
- Learn forward/backward propagation
- Master PyTorch fundamentals

### 2️⃣ Move to CNN-EfficientNet
- Learn convolutional neural networks
- Understand transfer learning
- Apply pretrained models to new tasks

### 3️⃣ Coming Soon
- 🔜 RNN/LSTM for sequence data
- 🔜 Transformers for NLP

---

## 📊 Results Summary

### MLP-MNIST
```
Dataset: MNIST (60K train, 10K test)
Architecture: 784 → 512 → 256 → 10
Test Accuracy: ~98%
Training Time: ~2 min (GPU)
```

### CNN-EfficientNet
```
Dataset: CIFAR-10 (50K train, 10K test)
Architecture: EfficientNet-B0 (pretrained ImageNet)
Test Accuracy: ~98%
Training Time: ~10 min (GPU)
```

---

## 🛠️ Technologies

- **PyTorch** - Deep learning framework
- **TorchVision** - Pretrained models & datasets
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **Scikit-learn** - Metrics & evaluation

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Created by Carlos Manuel** • Part of the Deep Learning learning journey 🚀

</div>
