"""
Weight Quantization Experiment for Food-11 Classification

=============================================================================
WEIGHT QUANTIZATION OVERVIEW
=============================================================================

Weight Quantization is a model compression technique that reduces the number
of bits used to represent each weight, thereby reducing model size and
potentially speeding up inference.

How It Works:
-------------
1. ORIGINAL: Weights stored as 32-bit floats (4 bytes per weight)
2. QUANTIZED: Weights stored as 16-bit, 8-bit, or even fewer bits

Quantization Methods:
--------------------
1. FLOAT16 (16-bit): Direct conversion from float32 to float16
   - Simple: Just cast the data type
   - Size reduction: ~50%
   - Accuracy loss: Minimal (float16 has good precision)

2. INT8 (8-bit): Min-Max quantization
   - Formula: W' = round((W - min(W)) / (max(W) - min(W)) * 255)
   - Stores: min_val, max_val, and uint8 array
   - Size reduction: ~75%
   - Accuracy loss: Small (depends on weight distribution)

3. INT4 (4-bit): More aggressive quantization
   - Formula: W' = round((W - min(W)) / (max(W) - min(W)) * 15)
   - Size reduction: ~87.5%
   - Accuracy loss: Moderate

Trade-offs:
-----------
- Lower bits = Smaller model size
- Lower bits = More quantization error
- Lower bits = Potentially faster inference (on specialized hardware)

This Experiment:
---------------
1. Quantize pre-trained SimpleCNN and MobileNetV1 models
2. Quantize pruned models (from pruning experiment)
3. Compare: Original vs Quantized vs Pruned+Quantized
4. Measure: Size reduction, accuracy retention

Reference: "Quantization and Training of Neural Networks for Efficient
           Integer-Arithmetic-Only Inference" (Jacob et al., CVPR 2018)
=============================================================================
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ========================== Configuration ==========================
BATCH_SIZE = 32
NUM_CLASSES = 11
DATA_DIR = './food-11'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-trained model paths
SIMPLE_CNN_PATH = 'simple_cnn_best.pth'
MOBILENET_PATH = 'mobilenet_v1_best.pth'

# Pruned model paths (from train_network_pruning.py)
# These will be searched dynamically based on available files
PRUNED_CNN_PATTERN = 'pruned_simplecnn_*.pth'
PRUNED_MOBILENET_PATTERN = 'pruned_mobilenet_*.pth'


# ========================== Dataset ==========================
# Class name to index mapping (alphabetically sorted)
CLASS_NAMES = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
               'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class FoodDataset(torch.utils.data.Dataset):
    """Custom Dataset for Food-11 classification task."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(glob(os.path.join(root, '*', '*.jpg')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        # Extract label from path: .../ClassName/image.jpg -> index
        class_name = os.path.basename(os.path.dirname(img_path))
        label = CLASS_TO_IDX[class_name]

        if self.transform:
            image = self.transform(image)
        return image, label


# Data transforms
test_transform = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_dataloader(mode='evaluation', batch_size=32):
    """Create dataloader for specified mode."""
    dataset = FoodDataset(os.path.join(DATA_DIR, mode), transform=test_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


# ========================== Model Definitions ==========================
# Must match the architectures in train_simple_cnn.py and train_mobilenet.py

class SimpleCNN(nn.Module):
    """Simple CNN classifier (matches train_simple_cnn.py architecture - flat Sequential)."""
    def __init__(self, width_mult=1.0):
        super(SimpleCNN, self).__init__()
        channels = [int(c * width_mult) for c in [64, 128, 256, 512, 512]]

        # FLAT Sequential structure (matches trained model)
        self.cnn = nn.Sequential(
            # Block 0
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 1
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(channels[4] * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution Block (matches train_mobilenet.py)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    """MobileNet v1 style network (matches train_mobilenet.py architecture)."""
    def __init__(self, base_channels=16, width_mult=1.0):
        super(MobileNetV1, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]
        channels = [max(1, int(base_channels * m * width_mult)) for m in multiplier]

        self.features = nn.Sequential(
            # Initial standard convolution
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Depthwise separable blocks
            DepthwiseSeparableConv(channels[0], channels[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DepthwiseSeparableConv(channels[1], channels[2]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DepthwiseSeparableConv(channels[2], channels[3]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DepthwiseSeparableConv(channels[3], channels[4]),
            DepthwiseSeparableConv(channels[4], channels[5]),
            DepthwiseSeparableConv(channels[5], channels[6]),
            DepthwiseSeparableConv(channels[6], channels[7]),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(channels[7], NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ========================== Quantization Functions ==========================

def encode16(params, fname):
    """
    Encode model parameters to 16-bit floating point.

    This is the simplest quantization method:
    - Convert float32 -> float16
    - Size reduction: ~50%
    - Precision loss: Minimal (float16 has 5 exponent bits, 10 mantissa bits)

    Args:
        params: Model state_dict
        fname: Output filename for quantized model
    """
    custom_dict = {}
    for name, param in params.items():
        param = np.float64(param.cpu().numpy())
        if isinstance(param, np.ndarray):
            # Convert to float16 (half precision)
            custom_dict[name] = np.float16(param)
        else:
            # Scalar values: keep as-is
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode16(fname):
    """
    Decode 16-bit quantized model back to PyTorch tensors.

    Note: PyTorch will convert float16 back to float32 during computation.
    """
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for name, param in params.items():
        custom_dict[name] = torch.tensor(param)
    return custom_dict


def encode8(params, fname):
    """
    Encode model parameters to 8-bit unsigned integers.

    Uses Min-Max quantization:
    - Formula: W' = round((W - min) / (max - min) * 255)
    - Stores: (min_val, max_val, uint8_array) for each weight
    - Size reduction: ~75% (plus small overhead for min/max)

    Args:
        params: Model state_dict
        fname: Output filename for quantized model
    """
    custom_dict = {}
    for name, param in params.items():
        param = np.float64(param.cpu().numpy())
        if isinstance(param, np.ndarray):
            min_val = np.min(param)
            max_val = np.max(param)
            # Handle edge case: all values are the same
            if max_val - min_val < 1e-10:
                quantized = np.zeros_like(param, dtype=np.uint8)
            else:
                # Min-Max normalization to [0, 255]
                quantized = np.round((param - min_val) / (max_val - min_val) * 255)
                quantized = np.uint8(quantized)
            custom_dict[name] = (min_val, max_val, quantized)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode8(fname):
    """
    Decode 8-bit quantized model back to PyTorch tensors.

    Reverses the Min-Max quantization:
    - Formula: W = W' / 255 * (max - min) + min
    """
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for name, param in params.items():
        if isinstance(param, tuple):
            min_val, max_val, quantized = param
            # Dequantize: uint8 -> float
            dequantized = np.float64(quantized)
            if max_val - min_val < 1e-10:
                dequantized = np.full_like(dequantized, min_val)
            else:
                dequantized = (dequantized / 255) * (max_val - min_val) + min_val
            custom_dict[name] = torch.tensor(dequantized)
        else:
            custom_dict[name] = torch.tensor(param)
    return custom_dict




# ========================== Evaluation Functions ==========================

def evaluate(model, dataloader):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = torch.LongTensor(labels).to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += images.size(0)

    return total_correct / total_samples


def get_model_size(path):
    """Get file size in bytes."""
    return os.stat(path).st_size


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def quantize_and_evaluate(model, model_name, original_path, eval_loader):
    """
    Quantize a model using different bit widths and evaluate accuracy.

    Process:
    1. Load original model and evaluate (32-bit baseline)
    2. Quantize to 16-bit, evaluate
    3. Quantize to 8-bit, evaluate
    4. Report size reduction and accuracy for each

    Args:
        model: The model architecture (with loaded weights)
        model_name: Name for display and file naming
        original_path: Path to original .pth file
        eval_loader: DataLoader for evaluation

    Returns:
        Dict with results for each bit width
    """
    results = {}
    base_name = os.path.splitext(original_path)[0]

    # ==================== 32-bit (Original) ====================
    original_size = get_model_size(original_path)
    original_acc = evaluate(model, eval_loader)

    results['32-bit'] = {
        'size': original_size,
        'accuracy': original_acc,
        'reduction': 1.0
    }
    print(f"  32-bit: Size={original_size:,} bytes, Accuracy={original_acc:.4f}")

    # ==================== 16-bit Quantization ====================
    params = torch.load(original_path, map_location=DEVICE)
    fname_16 = f"{base_name}_16bit.pkl"
    encode16(params, fname_16)

    # Load quantized weights and evaluate
    quantized_params = decode16(fname_16)
    model.load_state_dict(quantized_params)
    acc_16 = evaluate(model, eval_loader)
    size_16 = get_model_size(fname_16)

    results['16-bit'] = {
        'size': size_16,
        'accuracy': acc_16,
        'reduction': original_size / size_16
    }
    print(f"  16-bit: Size={size_16:,} bytes ({size_16/original_size*100:.1f}%), "
          f"Accuracy={acc_16:.4f} ({(acc_16-original_acc)*100:+.2f}%)")

    # ==================== 8-bit Quantization ====================
    params = torch.load(original_path, map_location=DEVICE)
    fname_8 = f"{base_name}_8bit.pkl"
    encode8(params, fname_8)

    # Load quantized weights and evaluate
    quantized_params = decode8(fname_8)
    model.load_state_dict(quantized_params)
    acc_8 = evaluate(model, eval_loader)
    size_8 = get_model_size(fname_8)

    results['8-bit'] = {
        'size': size_8,
        'accuracy': acc_8,
        'reduction': original_size / size_8
    }
    print(f"  8-bit:  Size={size_8:,} bytes ({size_8/original_size*100:.1f}%), "
          f"Accuracy={acc_8:.4f} ({(acc_8-original_acc)*100:+.2f}%)")

    return results


# ========================== Main Experiment ==========================

def main():
    """
    Main quantization experiment.

    Experiment Design:
    ==================
    1. BASELINE QUANTIZATION
       - Quantize original SimpleCNN (32-bit -> 16-bit -> 8-bit)
       - Quantize original MobileNetV1 (32-bit -> 16-bit -> 8-bit)

    2. PRUNED MODEL QUANTIZATION
       - Quantize pruned SimpleCNN (if available from pruning experiment)
       - Quantize pruned MobileNetV1 (if available from pruning experiment)

    3. COMPARISON
       - Compare compression ratios: Original vs Pruned vs Pruned+Quantized
       - Compare accuracy retention across all methods

    Expected Insights:
    - Quantization provides consistent size reduction (~50% for 16-bit, ~75% for 8-bit)
    - Combining pruning + quantization gives multiplicative compression
    - Different architectures may respond differently to quantization
    """
    print("=" * 70)
    print("WEIGHT QUANTIZATION EXPERIMENT")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print("\nQuantization Levels:")
    print("  32-bit: Original float32 (4 bytes per weight)")
    print("  16-bit: Half precision float16 (2 bytes per weight)")
    print("  8-bit:  Min-Max quantized uint8 (1 byte per weight)")

    # Load evaluation data
    print("\nLoading evaluation dataset...")
    eval_loader = get_dataloader('evaluation', BATCH_SIZE)
    print(f"Evaluation samples: {len(eval_loader.dataset)}")

    # Check available models
    cnn_exists = os.path.exists(SIMPLE_CNN_PATH)
    mobilenet_exists = os.path.exists(MOBILENET_PATH)

    print(f"\nPre-trained models:")
    print(f"  SimpleCNN ({SIMPLE_CNN_PATH}): {'Found' if cnn_exists else 'NOT FOUND'}")
    print(f"  MobileNetV1 ({MOBILENET_PATH}): {'Found' if mobilenet_exists else 'NOT FOUND'}")

    if not cnn_exists and not mobilenet_exists:
        print("\nERROR: No pre-trained models found!")
        print("Please run train_simple_cnn.py and train_mobilenet.py first.")
        return

    all_results = {}

    # ==================== Part 1: Quantize Original Models ====================
    print("\n" + "=" * 70)
    print("PART 1: QUANTIZING ORIGINAL MODELS")
    print("=" * 70)

    if cnn_exists:
        print(f"\n--- SimpleCNN ---")
        model = SimpleCNN(width_mult=1.0).to(DEVICE)
        model.load_state_dict(torch.load(SIMPLE_CNN_PATH, map_location=DEVICE))
        params = count_parameters(model)
        print(f"Parameters: {params:,}")

        results = quantize_and_evaluate(model, "SimpleCNN", SIMPLE_CNN_PATH, eval_loader)
        all_results['SimpleCNN_Original'] = results

    if mobilenet_exists:
        print(f"\n--- MobileNetV1 ---")
        model = MobileNetV1(base_channels=16, width_mult=1.0).to(DEVICE)
        model.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE))
        params = count_parameters(model)
        print(f"Parameters: {params:,}")

        results = quantize_and_evaluate(model, "MobileNetV1", MOBILENET_PATH, eval_loader)
        all_results['MobileNetV1_Original'] = results

    # ==================== Part 2: Quantize Pruned Models ====================
    print("\n" + "=" * 70)
    print("PART 2: QUANTIZING PRUNED MODELS")
    print("=" * 70)

    # Look for pruned models (from train_network_pruning.py output)
    pruned_cnn_files = sorted(glob('pruned_simplecnn_*.pth'))
    pruned_mobile_files = sorted(glob('pruned_mobilenet_*.pth'))

    # Also check for alternative naming patterns
    if not pruned_cnn_files:
        pruned_cnn_files = sorted(glob('pruned_model_width_*.pth'))

    print(f"\nPruned models found:")
    print(f"  SimpleCNN pruned: {len(pruned_cnn_files)} file(s)")
    print(f"  MobileNetV1 pruned: {len(pruned_mobile_files)} file(s)")

    if not pruned_cnn_files and not pruned_mobile_files:
        print("\nNo pruned models found.")
        print("Run train_network_pruning.py first to generate pruned models.")
    else:
        # Quantize the most compressed (last) pruned model for each architecture
        if pruned_cnn_files:
            pruned_path = pruned_cnn_files[-1]  # Most pruned version
            print(f"\n--- SimpleCNN Pruned ({os.path.basename(pruned_path)}) ---")

            # Extract width_mult from filename if possible
            try:
                width_str = pruned_path.split('_')[-1].replace('.pth', '')
                width_mult = float(width_str)
            except ValueError:
                width_mult = 0.77  # Default after 5 iterations of 0.95

            model = SimpleCNN(width_mult=width_mult).to(DEVICE)
            model.load_state_dict(torch.load(pruned_path, map_location=DEVICE))
            params = count_parameters(model)
            print(f"Parameters: {params:,}")

            results = quantize_and_evaluate(model, "SimpleCNN_Pruned", pruned_path, eval_loader)
            all_results['SimpleCNN_Pruned'] = results

        if pruned_mobile_files:
            pruned_path = pruned_mobile_files[-1]
            print(f"\n--- MobileNetV1 Pruned ({os.path.basename(pruned_path)}) ---")

            try:
                width_str = pruned_path.split('_')[-1].replace('.pth', '')
                width_mult = float(width_str)
            except ValueError:
                width_mult = 0.77

            model = MobileNetV1(base_channels=16, width_mult=width_mult).to(DEVICE)
            model.load_state_dict(torch.load(pruned_path, map_location=DEVICE))
            params = count_parameters(model)
            print(f"Parameters: {params:,}")

            results = quantize_and_evaluate(model, "MobileNetV1_Pruned", pruned_path, eval_loader)
            all_results['MobileNetV1_Pruned'] = results

    # ==================== Part 3: Summary Comparison ====================
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'Bit Width':<10} {'Size (bytes)':>15} {'Reduction':>12} {'Accuracy':>10}")
    print("-" * 75)

    for model_name, results in all_results.items():
        for bit_width, data in results.items():
            reduction_str = f"{data['reduction']:.2f}x" if data['reduction'] > 1 else "1.00x"
            print(f"{model_name:<25} {bit_width:<10} {data['size']:>15,} {reduction_str:>12} {data['accuracy']:>9.4f}")
        print()

    # ==================== Part 4: Compression Pipeline Comparison ====================
    if 'SimpleCNN_Original' in all_results and 'SimpleCNN_Pruned' in all_results:
        print("\n" + "=" * 70)
        print("COMPRESSION PIPELINE COMPARISON: SimpleCNN")
        print("=" * 70)

        orig_32 = all_results['SimpleCNN_Original']['32-bit']
        pruned_32 = all_results['SimpleCNN_Pruned']['32-bit']
        pruned_8 = all_results['SimpleCNN_Pruned']['8-bit']

        print(f"\n{'Stage':<30} {'Size (bytes)':>15} {'Total Reduction':>18} {'Accuracy':>10}")
        print("-" * 75)
        print(f"{'Original (32-bit)':<30} {orig_32['size']:>15,} {'1.00x':>18} {orig_32['accuracy']:>9.4f}")
        print(f"{'+ Pruning (32-bit)':<30} {pruned_32['size']:>15,} {orig_32['size']/pruned_32['size']:>17.2f}x {pruned_32['accuracy']:>9.4f}")
        print(f"{'+ Pruning + Quantization (8-bit)':<30} {pruned_8['size']:>15,} {orig_32['size']/pruned_8['size']:>17.2f}x {pruned_8['accuracy']:>9.4f}")

    if 'MobileNetV1_Original' in all_results and 'MobileNetV1_Pruned' in all_results:
        print("\n" + "=" * 70)
        print("COMPRESSION PIPELINE COMPARISON: MobileNetV1")
        print("=" * 70)

        orig_32 = all_results['MobileNetV1_Original']['32-bit']
        pruned_32 = all_results['MobileNetV1_Pruned']['32-bit']
        pruned_8 = all_results['MobileNetV1_Pruned']['8-bit']

        print(f"\n{'Stage':<30} {'Size (bytes)':>15} {'Total Reduction':>18} {'Accuracy':>10}")
        print("-" * 75)
        print(f"{'Original (32-bit)':<30} {orig_32['size']:>15,} {'1.00x':>18} {orig_32['accuracy']:>9.4f}")
        print(f"{'+ Pruning (32-bit)':<30} {pruned_32['size']:>15,} {orig_32['size']/pruned_32['size']:>17.2f}x {pruned_32['accuracy']:>9.4f}")
        print(f"{'+ Pruning + Quantization (8-bit)':<30} {pruned_8['size']:>15,} {orig_32['size']/pruned_8['size']:>17.2f}x {pruned_8['accuracy']:>9.4f}")

    # Final insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. 16-bit Quantization:
   - Reduces model size by ~50%
   - Minimal accuracy loss (float16 has sufficient precision)

2. 8-bit Quantization:
   - Reduces model size by ~75%
   - Small accuracy loss due to quantization error

3. Pruning + Quantization:
   - Multiplicative compression effect
   - E.g., 2x from pruning + 4x from 8-bit = 8x total compression

4. Architecture Comparison:
   - SimpleCNN: More parameters, more room for compression
   - MobileNetV1: Already efficient, still benefits from quantization
""")


if __name__ == '__main__':
    main()