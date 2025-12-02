import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# USE THE SAME MODEL STRUCTURE AS train_simple_cnn.py
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
    print("=" * 50)
    print("NETWORK PRUNING")
    print("=" * 50)
    
    # Load the model you just trained
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("simple_cnn_best.pth", map_location=device))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Original model: {total_params:,} parameters")
    
    # Create pruned versions
    print("\nCreating pruned models...")
    
    for prune_rate in [0.1, 0.3, 0.5]:  # 10%, 30%, 50% pruning
        # Create new model
        pruned_model = SimpleCNN().to(device)
        pruned_model.load_state_dict(torch.load("simple_cnn_best.pth", map_location=device))
        
        # Simple pruning: set smallest weights to zero
        with torch.no_grad():
            for name, param in pruned_model.named_parameters():
                if 'weight' in name:
                    flat = param.abs().flatten()
                    threshold = torch.quantile(flat, prune_rate)
                    mask = param.abs() > threshold
                    param.data *= mask.float()
        
        # Count non-zero
        non_zero = sum(torch.count_nonzero(p).item() for p in pruned_model.parameters())
        sparsity = (1 - non_zero/total_params) * 100
        
        # Save
        filename = f"pruned_{int(prune_rate*100)}percent.pth"
        torch.save(pruned_model.state_dict(), filename)
        
        print(f"Pruned {int(prune_rate*100)}%: {non_zero:,} non-zero params ({sparsity:.1f}% sparsity)")
    
    print("\n✅ Pruning complete!")

if __name__ == "__main__":
    main()