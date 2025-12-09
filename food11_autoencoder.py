import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm

from huggingface_hub import login
import os

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("HF_TOKEN environment variable is not set.")


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "food-11/training"
VAL_DIR = "food-11/validation"

# ---------------------------------------------------------
# LOAD PRETRAINED GOOGLE VIT
# ---------------------------------------------------------
print("Loading Google ViT Base Patch16...")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
vit.to(DEVICE)
vit.eval()  # freeze transformer weights

# ---------------------------------------------------------
# DATA TRANSFORM (224x224 FOR VIT)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_ds = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_ds.classes)
print("Classes:", train_ds.classes)

# ---------------------------------------------------------
# CLASSIFIER ON TOP OF VIT
# ---------------------------------------------------------
class ViTClassifier(nn.Module):
    def __init__(self, vit_model, num_classes):
        super().__init__()
        self.vit = vit_model
        self.classifier = nn.Linear(768, num_classes)  # ViT hidden size = 768

        # freeze vit backbone
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_token)

model = ViTClassifier(vit, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
print("Training classifier head...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/vit_food11_classifier.pth")
print("Saved: checkpoints/vit_food11_classifier.pth")
