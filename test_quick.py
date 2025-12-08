"""
Quick test script to verify everything works.
Trains the model for 2 epochs just to verify there are no errors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("=" * 70)
print("QUICK TEST - MLP MNIST")
print("=" * 70)

# 1. Verify PyTorch
print("\nVerifying PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

# 2. Load data
print("\nLoading MNIST data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use only subset for quick test
train_subset = torch.utils.data.Subset(train_dataset, range(5000))
test_subset = torch.utils.data.Subset(test_dataset, range(1000))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

print(f"Data loaded (subset for quick test)")
print(f"   Training batches: {len(train_loader)}")
print(f"   Test batches: {len(test_loader)}")

# 3. Create model
print("\nCreating model...")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model created")
print(f"   Parameters: {total_params:,}")

# 4. Train 2 epochs (quick test)
print("\nTraining for 2 epochs (quick test)...")
print("-" * 70)

for epoch in range(1, 3):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device).view(images.size(0), -1)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    avg_loss = train_loss / len(train_loader)
    avg_acc = 100 * train_correct / train_total
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device).view(images.size(0), -1)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    
    print(f"Epoch {epoch}/2 | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}% | Test Acc: {test_acc:.2f}%")

print("-" * 70)
print("\nQUICK TEST COMPLETED SUCCESSFULLY")
print("\nVerdict: All components are working correctly")
print("\n   - PyTorch working")
print("   - MNIST downloaded correctly")
print("   - DataLoaders working")
print("   - Model created correctly")
print("   - Training working")
print("   - Evaluation working")

print("\n" + "=" * 70)
print("Now run main.py to train the full model!")
print("=" * 70)
