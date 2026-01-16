# 🎭 LSTM Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> LSTM-based sentiment analysis classifier trained on IMDB movie reviews dataset.

## 📋 Overview

This project implements a **Long Short-Term Memory (LSTM)** neural network for binary sentiment classification (positive/negative) on movie reviews. It demonstrates:

- Text preprocessing and tokenization
- Trainable word embeddings
- Bidirectional LSTM architecture
- Attention mechanism for interpretability
- PyTorch best practices

## 🏗️ Project Structure

```
LSTM-Sentiment/
├── main.py              # Training and evaluation script
├── test_quick.py        # Quick test to verify setup
├── requirements.txt     # Dependencies
├── README.md
├── data/                # Dataset (auto-downloaded)
├── models/
│   ├── __init__.py
│   └── lstm.py          # LSTM model architecture
├── notebooks/
│   └── sentiment_analysis.ipynb  # Interactive tutorial (Colab compatible)
└── src/
    ├── __init__.py
    ├── data.py          # Data loading and preprocessing
    ├── engine.py        # Training and evaluation loops
    └── utils.py         # Utilities and helpers
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-deep-learning.git
cd pytorch-deep-learning/LSTM-Sentiment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default parameters
python main.py

# Train with custom parameters
python main.py --epochs 10 --batch_size 64 --lr 0.001 --hidden_dim 256

# Train with attention mechanism
python main.py --attention
```

### Quick Test

```bash
# Verify installation and run inference
python test_quick.py
```

### Google Colab

The notebook `notebooks/sentiment_analysis.ipynb` is **fully standalone** and works in Google Colab without uploading any files. It will:
- Auto-install dependencies
- Download the IMDB dataset from HuggingFace
- Train the model
- Auto-download the trained `.pth` file when finished

## 📊 Model Architecture

```
Input Text
    │
    ▼
┌─────────────────┐
│   Embedding     │  (vocab_size × embed_dim)
│   (Trainable)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bidirectional  │  (2 layers, hidden_dim=128)
│      LSTM       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Attention     │  (weighted sum of hidden states)
│    Layer        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fully Connected│  (hidden_dim × 2 → num_classes)
│     + Dropout   │
└────────┬────────┘
         │
         ▼
    Output (pos/neg)
```

## 📈 Expected Results

| Metric | Value |
|--------|-------|
| Accuracy | ~85-88% |
| F1-Score | ~0.85-0.88 |
| Training Time | ~5-15 min (GPU) |

*Results may vary depending on hyperparameters and training epochs.*

## 🎯 Features

- **Trainable Embeddings**: Word embeddings learned during training
- **Bidirectional LSTM**: Captures context from both directions
- **Attention Mechanism**: Highlights important words for prediction
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Colab Compatible**: Notebook runs standalone in Google Colab

## 📚 Dataset

**IMDB Movie Reviews** (50,000 reviews)
- 25,000 training samples
- 25,000 test samples
- Binary labels: positive (1) / negative (0)

The dataset is automatically downloaded from **HuggingFace Datasets** on first run.

## 🔧 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 100 | Embedding dimension |
| `hidden_dim` | 128 | LSTM hidden size |
| `num_layers` | 2 | Number of LSTM layers |
| `dropout` | 0.5 | Dropout rate |
| `batch_size` | 32 | Batch size |
| `lr` | 0.001 | Learning rate |
| `epochs` | 5 | Training epochs |
| `max_vocab` | 25000 | Maximum vocabulary size |
| `max_len` | 256 | Maximum sequence length |

## 📖 Learning Resources

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Part of the PyTorch Deep Learning Portfolio**
