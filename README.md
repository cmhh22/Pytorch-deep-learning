# MLP MNIST - Multi-Layer Perceptron with PyTorch

An educational and professional project implementing a **Multi-Layer Perceptron (MLP)** for MNIST digit classification using PyTorch.

## Table of Contents

- [What is an MLP?](#what-is-an-mlp)
- [MNIST Dataset](#mnist-dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)
- [Key Concepts Explained](#key-concepts-explained)
- [Future Improvements](#future-improvements)

---

## What is an MLP?

A **Multi-Layer Perceptron (MLP)** is an artificial neural network with multiple layers:

### Model Layers

```
INPUT (784) → FC1 (512) → ReLU → Dropout 
                                     ↓
                FC2 (256) → ReLU → Dropout
                                     ↓
                         FC3 (10) → OUTPUT
```

**Explanation:**
- **Input Layer (784)**: A 28×28 image flattened to a vector of 784 values
- **Hidden Layer 1 (512)**: Captures basic features
- **Hidden Layer 2 (256)**: Captures more complex features
- **Output Layer (10)**: Digit prediction (0-9)
- **ReLU**: Activation function that introduces non-linearity
- **Dropout**: Regularization to prevent overfitting

---

## MNIST Dataset

**MNIST** = Modified National Institute of Standards and Technology database

- **60,000** training images
- **10,000** test images
- Size: **28×28 pixels** in grayscale
- Values: **0-255** (then normalized)
- Goal: Classify handwritten digits (0-9)

### Normalization

```python
Normalized = (pixel - mean) / standard_deviation
           = (pixel - 0.1307) / 0.3081
```

---

## Requirements

- Python >= 3.8
- CUDA 11.8+ (optional, for GPU)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MLP-MNIST.git
cd MLP-MNIST
```

### 2. Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
MLP-MNIST/
├── data/                    # MNIST dataset (downloaded automatically)
├── models/                  # Saved trained models
│   ├── mlp.py              # Model architecture
│   └── mlp_mnist.pth       # Trained model (generated)
├── src/
│   └── utils.py            # Utility functions
├── notebooks/
│   └── mnist_tutorial.ipynb # Educational notebook
├── main.py                  # Main script
├── requirements.txt         # Dependencies
├── .gitignore              # Files to ignore in git
└── README.md               # This file
```

### File Descriptions

| File | Description |
|------|-------------|
| `models/mlp.py` | Defines the MLP architecture |
| `src/utils.py` | Data loading, training and evaluation functions |
| `main.py` | Orchestrates the training flow |
| `notebooks/mnist_tutorial.ipynb` | Interactive analysis and visualization |

---

## Usage

### Train the Model

```bash
python main.py
```

**This will:**
1. Download MNIST automatically
2. Create and train the model for 20 epochs
3. Save the best model to `models/mlp_mnist.pth`
4. Display training and test metrics

### Run the Educational Notebook

```bash
jupyter notebook notebooks/mnist_tutorial.ipynb
```

---

## How It Works

### 1. Data Loading (`src/utils.py` - `load_mnist_data()`)

```python
# Normalize images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

# Split: 85% train, 15% validation
train_size = int(0.85 * len(train_dataset))
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
```

### 2. Forward Pass

```python
def forward(self, x):  # x: (batch_size, 784)
    x = self.fc1(x)     # → (batch_size, 512)
    x = self.relu(x)    # Non-linear activation
    x = self.dropout(x) # Regularization
    
    x = self.fc2(x)     # → (batch_size, 256)
    x = self.relu(x)
    x = self.dropout(x)
    
    x = self.fc3(x)     # → (batch_size, 10)
    return x            # Logits without activation
```

### 3. Loss Function: Cross Entropy

Combines Log-Softmax + Negative Log-Likelihood

```
CE = -Σ(yi * log(pi))

Where:
- yi: true label (one-hot)
- pi: predicted probability
```

### 4. Optimizer: Adam

Adaptive Moment Estimation

```
m_t = β₁*m_{t-1} + (1-β₁)*∇L      # Momentum
v_t = β₂*v_{t-1} + (1-β₂)*∇L²     # RMSprop
θ_t = θ_{t-1} - α * m_t / √v_t    # Update
```

### 5. Validation and Early Stopping

After each epoch:
- Evaluates on validation set
- Saves model if accuracy improves
- Prevents overfitting

---

## Expected Results

After training 20 epochs:

```
Train Accuracy: ~98-99%
Validation Accuracy: ~97-98%
Test Accuracy: ~97-98%
```

**Note:** Results may vary depending on:
- Random weight initialization
- Data order in batches
- Device (GPU vs CPU)

---

## Key Concepts Explained

### 1. ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)

Advantages:
- Introduces non-linearity
- Computes fast
- Mitigates vanishing gradient problem
```

### 2. Dropout

```
During training: Randomly deactivates 20% of neurons
During inference: Uses all neurons

Benefit: Regularization, reduces overfitting
```

### 3. Backpropagation

```
Forward Pass:  x → MLP → ŷ
Compute Loss: L = CE(ŷ, y)
Backward Pass: ∂L/∂w = ∂L/∂y * ∂y/∂w
Update:    w_new = w - α * ∂L/∂w
```

### 4. Epoch vs Batch

```
Epoch = passing entire dataset once
Batch = small subset (64 images)

With batch_size=64 and 51,000 training images:
Batches per epoch = 51,000 / 64 ≈ 797 batches
```

---

## Customization

### Change Hyperparameters in `main.py`

```python
BATCH_SIZE = 64           # Change to 32, 128, etc.
LEARNING_RATE = 0.001     # Change learning rate
NUM_EPOCHS = 20           # More epochs = more training
```

### Modify Architecture in `models/mlp.py`

```python
# Example: Larger model
model = MLP(
    input_size=784,
    hidden1=1024,   # Larger
    hidden2=512,    # Larger
    num_classes=10
)
```

---

## Suggested Learning Flow

1. **Read** the code in `models/mlp.py`
   - Understand the layer architecture
   - Examine the forward pass

2. **Explore** `src/utils.py`
   - How data is loaded and normalized
   - How the training loop works

3. **Run** `main.py`
   - Watch the model train in real-time
   - Observe how loss decreases

4. **Experiment** with hyperparameters
   - Change learning_rate
   - Try different architectures
   - Observe how results change

5. **Visualize** with the notebook
   - See predictions on real images
   - Analyze error distribution

---

## Training Monitoring

The script shows in real-time:

```
Using device: cuda
Downloading MNIST dataset...
Dataset loaded:
   - Training: 51000 images
   - Validation: 9000 images
   - Testing: 10000 images

MLP TRAINING ON MNIST WITH PYTORCH
STARTING TRAINING...

EPOCH 1/20
Epoch [1] Batch [100] Loss: 0.2547 | Accuracy: 92.12%
Epoch [1] Batch [200] Loss: 0.1823 | Accuracy: 94.87%
...
Epoch 1 Summary:
   Training   -> Loss: 0.3122 | Accuracy: 91.45%
   Validation -> Loss: 0.2841 | Accuracy: 92.63%
   New best model!

[Epochs 2-20...]

TRAINING COMPLETED
EVALUATING ON TEST SET...

FINAL RESULTS:
   Test Accuracy: 97.52%
   Test Loss: 0.0948
```

---

## Troubleshooting

### Problem: "CUDA out of memory"

**Solution:**
```python
# Reduce batch size in main.py
BATCH_SIZE = 32  # from 64 to 32
```

### Problem: Slow MNIST download

**Solution:** Data is cached after the first download

### Problem: Model not improving

**Solution:**
- Increase learning_rate
- Train more epochs
- Verify there are no bugs in the code

---

## Educational References

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [PyTorch Official Tutorial](https://pytorch.org/tutorials/)
- [Backpropagation Explained](https://brilliant.org/wiki/backpropagation/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

## Future Improvements

- [ ] Add CNN (Convolutional Neural Network)
- [ ] Implement batch normalization
- [ ] Add data augmentation
- [ ] Create REST API to serve predictions
- [ ] Visualize learned features
- [ ] Compare with other models (KNN, SVM, Forests)

---

## License

This project is under MIT license. See `LICENSE` file for more details.

---

## Author

Created as an educational project to learn Deep Learning with PyTorch.

---

## Contributions

Contributions are welcome. For large changes, open an issue first.

---

## Questions?

If you have questions about the code or concepts, check the comments in the code and the educational notebook.

**Happy learning!**
