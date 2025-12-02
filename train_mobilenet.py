"""
MobileNet v1 Training Script for Food-11 Classification
Optimized for CPU training.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ========================== Configuration ==========================
BATCH_SIZE = 64           # bigger batch for CPU efficiency
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
DATA_DIR = './food-11'
NUM_CLASSES = 11
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================== Dataset ==========================
class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, folder_name, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        class_folders = sorted(os.listdir(folder_name))
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(folder_name, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_path in sorted(glob(os.path.join(class_path, '*.jpg'))):
                self.data.append(img_path)
                self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def get_dataloader(mode='training', batch_size=BATCH_SIZE):
    transform = train_transform if mode == 'training' else test_transform
    dataset = FoodDataset(os.path.join(DATA_DIR, mode), transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode=='training'),
        num_workers=0,
        pin_memory=False
    )
    return dataloader

# ========================== Model ==========================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, base_channels=16, width_mult=1.0):
        super().__init__()
        multiplier = [1,2,4,8,16,16,16,16]
        channels = [int(base_channels * m * width_mult) for m in multiplier]
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2,2),
            DepthwiseSeparableConv(channels[0], channels[1]),
            nn.MaxPool2d(2,2),
            DepthwiseSeparableConv(channels[1], channels[2]),
            nn.MaxPool2d(2,2),
            DepthwiseSeparableConv(channels[2], channels[3]),
            nn.MaxPool2d(2,2),
            DepthwiseSeparableConv(channels[3], channels[4]),
            DepthwiseSeparableConv(channels[4], channels[5]),
            DepthwiseSeparableConv(channels[5], channels[6]),
            DepthwiseSeparableConv(channels[6], channels[7]),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(channels[7], NUM_CLASSES)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ========================== Training/Evaluation ==========================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), torch.LongTensor(labels).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += outputs.argmax(1).eq(labels).sum().item()
        total_samples += images.size(0)
    return total_loss/total_samples, total_correct/total_samples

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), torch.LongTensor(labels).to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += outputs.argmax(1).eq(labels).sum().item()
            total_samples += images.size(0)
    return total_loss/total_samples, total_correct/total_samples

# ========================== Main ==========================
def main():
    print(f"Using device: {DEVICE}")
    train_loader = get_dataloader('training')
    val_loader = get_dataloader('validation')
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = MobileNetV1(base_channels=16).to(DEVICE)
    print(f"Model: MobileNet v1")
    print(f"Total parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_acc = 0.0
    print("Starting training...")
    print("-"*70)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'mobilenet_v1_best.pth')
            print(f"  -> New best model saved! (Val Acc: {val_acc:.4f})")

    print("-"*70)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")

    # Final evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('mobilenet_v1_best.pth'))
    _, eval_acc = evaluate(model, get_dataloader('evaluation'), criterion)
    print(f"Evaluation Accuracy: {eval_acc:.4f}")

if __name__ == '__main__':
    main()
