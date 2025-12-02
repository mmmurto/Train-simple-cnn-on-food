import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# WINDOWS FIX
os.environ['OMP_NUM_THREADS'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 11)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def main():
    print("Training SimpleCNN")
    
    # Check data
    if not os.path.exists("./food-11/training"):
        print("ERROR: food-11/training not found!")
        return
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    train_data = datasets.ImageFolder("./food-11/training", transform=transform)
    
    # FIXED: Use train_data.classes, NOT train_data.dataset.classes
    print(f"Classes: {train_data.classes}")
    print(f"Training samples: {len(train_data)}")
    
    # DataLoader with Windows fix
    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=0,      # WINDOWS MUST BE 0
        pin_memory=False    # WINDOWS MUST BE FALSE
    )
    
    # Create model
    model = SimpleCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, Acc {acc:.1f}%")
    
    # Save
    torch.save(model.state_dict(), "simple_cnn_best.pth")
    print("Model saved as simple_cnn_best.pth")

if __name__ == "__main__":
    main()