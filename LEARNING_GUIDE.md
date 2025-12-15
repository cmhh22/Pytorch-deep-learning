# Learning Guide - MLP MNIST

## How to Learn from This Project

This project is designed to teach you Deep Learning from scratch. Follow this order:

---

## Phase 1: Understand the Theory (1-2 hours)

### 1. Read the README.md
- Understand what an MLP is
- Learn about MNIST
- Familiarize yourself with the concepts

### 2. Review the architecture
```
INPUT (784) 
  ↓
FC1 + ReLU + Dropout (512 neurons)
  ↓
FC2 + ReLU + Dropout (256 neurons)
  ↓
FC3 (10 neurons)
  ↓
OUTPUT (prediction 0-9)
```

---

## Phase 2: Explore the Code (2-3 hours)

### File 1: `models/mlp.py` (Architecture)
```python
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden1=512, hidden2=256, num_classes=10):
        # Layers are defined here
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Forward pass happens here
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        ...
```

**What you learn:**
- How to create a neural network in PyTorch
- What forward pass is
- How fully-connected layers work

### File 2: `src/utils.py` (Helper functions)

**`load_mnist_data()`** - Load data
```python
# Normalize images
transform = transforms.Normalize(mean=[0.1307], std=[0.3081])
# Split into train/val/test
train_loader, val_loader, test_loader
```

**`train_epoch()`** - One training epoch
```python
# 1. Forward pass
outputs = model(data)

# 2. Calculate loss
loss = F.cross_entropy(outputs, target)

# 3. Backward pass
loss.backward()

# 4. Update weights
optimizer.step()
```

**`validate()` and `test()`** - Evaluation

### File 3: `main.py` (Orchestration)

```python
# Complete flow:
for epoch in range(NUM_EPOCHS):
    # 1. Train
    train_loss, train_acc = train_epoch(...)
    
    # 2. Validate
    val_loss, val_acc = validate(...)
    
    # 3. Save if improved
    if val_acc > best_acc:
        save_model(...)
```

---

## Phase 3: Run and Train (1-2 hours)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python main.py

# Expected result:
# Epoch 1: Train Loss: 0.3122 | Train Acc: 91.45% | Val Loss: 0.2841 | Val Acc: 92.63%
# Epoch 2: Train Loss: 0.2456 | Train Acc: 93.21% | Val Loss: 0.2345 | Val Acc: 94.18%
# ...
# Epoch 20: Train Loss: 0.0234 | Train Acc: 98.76% | Val Loss: 0.1023 | Val Acc: 97.82%
```

**What's happening?**
- Each epoch, the model processes 51,000 images
- Loss is decreasing = the model is learning
- Accuracy is increasing = more correct predictions
- Model is saved when validation improves

---

## Phase 4: Explore the Jupyter Notebook (2-3 hours)

```bash
jupyter notebook notebooks/mnist_tutorial.ipynb
```

The notebook has **8 sections**:

1. **Import libraries** - Understand each library
2. **Load MNIST** - Visualize real data
3. **Prepare DataLoaders** - Understand batches
4. **Define MLP** - Build the model
5. **Configure training** - Loss and optimizer
6. **Training loop** - Watch it train
7. **Evaluation** - Confusion matrix
8. **Predictions** - See specific examples

**Interact:**
- Modify values and see what happens
- Change the architecture
- Visualize incorrect predictions

---

## Phase 5: Experiment and Learn (2+ hours)

### Experiment 1: Change Learning Rate
```python
# In main.py
LEARNING_RATE = 0.0001  # Slower
# vs
LEARNING_RATE = 0.01    # Faster

# What happened? Did the model learn better or worse?
```

### Experiment 2: Change Architecture
```python
# Smaller model
model = MLP(hidden1=256, hidden2=128)

# Larger model
model = MLP(hidden1=1024, hidden2=512)

# Which is faster? Which has better accuracy?
```

### Experiment 3: More Epochs
```python
NUM_EPOCHS = 50  # vs 20

# Does accuracy improve? When does it stop improving?
```

### Experiment 4: Without Dropout
```python
# In models/mlp.py, comment out:
# x = self.dropout(x)

# Overfitting? Worse accuracy?
```

---

## Key Concepts to Understand

### 1. Forward Pass
```
Input (784) → FC1 (×512) → ReLU → Dropout →
FC2 (×256) → ReLU → Dropout →
FC3 (×10) → Output
```
**It's like:**
- Raw material input
- Processed through machines (layers)
- Final product comes out (prediction)

### 2. Backward Pass / Backpropagation
```
Error = (prediction - reality)
∂L/∂w = how error affects each weight

w_new = w_previous - learning_rate × ∂L/∂w
```
**It's like:**
- Measure how much error you had
- Calculate who is responsible for each error
- Adjust weights to reduce error

### 3. Loss Function (Cross Entropy)
```
CE = -Σ(true × log(predicted))
```
**Interpretation:**
- If predicts correctly: low loss
- If predicts incorrectly: high loss

### 4. Overfitting
```
Train Accuracy: 99.5%
Val Accuracy: 94.2%

Why is there a difference? 
→ Model "memorized" training data
→ Doesn't generalize well to new data
```

**Solution: Dropout**
```
During training: Randomly turn off 20% of neurons
During inference: Use all neurons

Effect: Forces model to learn robust features
```

---

## Frequently Asked Questions

### Why 784 in input?
```
28 × 28 = 784 pixels
An image is "flattened" into a vector of 784 values
```

### Why ReLU?
```
ReLU(x) = max(0, x)

Advantages:
- Introduces non-linearity
- Computes very fast
- Avoids vanishing gradients
```

### Why 10 in output?
```
There are 10 digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
Each neuron predicts the probability of a digit
```

### What is CrossEntropyLoss?
```
It's the "penalty" for incorrect predictions

If model says "it's a 3" but it's a "5":
→ High loss

If model says "it's a 5" (correct):
→ Low loss
```

### What is Adam optimizer?
```
It's like a "personal trainer" for the model

Features:
- Adapts learning rate for each weight
- Uses momentum (memory of previous changes)
- Converges fast
```

---

## Learning Checklist

- [ ] I understand what an MLP is
- [ ] I understand what MNIST is
- [ ] I can explain the forward pass
- [ ] I understand what backpropagation is
- [ ] I know what Dropout does
- [ ] I can read and understand `models/mlp.py`
- [ ] I can read and understand `src/utils.py`
- [ ] I have successfully run `main.py`
- [ ] I have explored the notebook
- [ ] I have done at least 3 experiments
- [ ] I can modify the architecture
- [ ] I understand the difference between train/val/test

---

## External Resources

### Videos
- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- [Andrew Ng - Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

### Papers
- [MNIST Dataset Original](http://yann.lecun.com/exdb/mnist/)
- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)

### Documentation
- [PyTorch Official](https://pytorch.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## Next Steps After Mastering MLP

1. **CNN (Convolutional Neural Networks)**
   - Better for images
   - Maintains spatial structure
   - Fewer parameters

2. **Data Augmentation**
   - Rotations
   - Translations
   - Zooms
   - Improves generalization

3. **Ensemble Models**
   - Combine multiple models
   - Better accuracy

4. **Transfer Learning**
   - Use pretrained models
   - Fine-tuning

5. **Production**
   - Serve model with FastAPI
   - Containerize with Docker
   - Deploy to cloud

---

## Good luck with your learning!

If you have questions, check:
1. The README.md
2. The comments in the code
3. The Jupyter notebook

**Happy learning!**
