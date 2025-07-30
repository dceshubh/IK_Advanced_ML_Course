# Computer Vision 1 Assignment Solution - Complete Code Analysis Guide

## Overview
This notebook demonstrates a complete implementation of **Cats & Dogs Classification using PyTorch from scratch**. It covers the entire deep learning pipeline from data loading to model evaluation, focusing on Convolutional Neural Networks (CNNs) for binary image classification.

## Complete Code Analysis with Line-by-Line Explanations

### 1. Google Colab Setup and Drive Mounting

```python
from google.colab import drive
drive.mount('/content/drive')
```
**Line-by-line explanation:**
- `from google.colab import drive`: Imports the Google Colab drive module for accessing Google Drive files
- `drive.mount('/content/drive')`: Mounts Google Drive to the Colab environment at the specified path, allowing access to files stored in Google Drive

### 2. Core Library Imports and Version Check

```python
import torch
from torch import nn
from torchsummary import summary
# Note: this notebook requires torch >= 1.10.0
torch.__version__
```
**Line-by-line explanation:**
- `import torch`: Imports the main PyTorch library for tensor operations and deep learning
- `from torch import nn`: Imports the neural network module containing layers, loss functions, and model building blocks
- `from torchsummary import summary`: Imports a utility for displaying model architecture summaries
- `# Note: this notebook requires torch >= 1.10.0`: Comment indicating version requirement
- `torch.__version__`: Displays the current PyTorch version (output: '2.4.1+cu121' indicating CUDA support)

### 3. Device Configuration for GPU/CPU

```python
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```
**Line-by-line explanation:**
- `# Setup device-agnostic code`: Comment explaining the purpose of device selection
- `device = "cuda" if torch.cuda.is_available() else "cpu"`: Conditional assignment that selects GPU if available, otherwise CPU
- `device`: Displays the selected device (output: 'cuda' indicating GPU is available)

### 4. Additional Import Statements for Data Handling

```python
import os
import gdown
import zipfile
import pandas as pd
import numpy as np
```
**Line-by-line explanation:**
- `import os`: Operating system interface for file and directory operations
- `import gdown`: Google Drive downloader for programmatic file downloads
- `import zipfile`: Archive handling for extracting compressed datasets
- `import pandas as pd`: Data manipulation library, aliased as 'pd' for convenience
- `import numpy as np`: Numerical computing library, aliased as 'np' for mathematical operations

### 5. Training Dataset Download and Extraction

```python
# The training dataset can be downloaded using the provided code. The data will be extracted into a specified directory.
dataset_url = "https://drive.google.com/uc?export=download&id=1kTO4N7tD1K-tRX59bqNWSKPdzLaWNgfW"

dataset_path = "./dataset" # the dataset will be downloaded in this folder
zip_file_path =  "training_set-20240919T083719Z-001"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(zip_file_path):
    gdown.download(dataset_url, zip_file_path, quiet=False)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)
```
**Line-by-line explanation:**
- `# The training dataset can be downloaded...`: Comment explaining the dataset download process
- `dataset_url = "https://drive.google.com/uc?export=download&id=1kTO4N7tD1K-tRX59bqNWSKPdzLaWNgfW"`: Google Drive URL for the training dataset
- `dataset_path = "./dataset"`: Local directory path where dataset will be stored
- `zip_file_path = "training_set-20240919T083719Z-001"`: Name of the downloaded zip file
- `if not os.path.exists(dataset_path):`: Checks if the dataset directory exists
- `os.makedirs(dataset_path)`: Creates the dataset directory if it doesn't exist
- `if not os.path.exists(zip_file_path):`: Checks if the zip file already exists to avoid re-downloading
- `gdown.download(dataset_url, zip_file_path, quiet=False)`: Downloads the file from Google Drive with progress display
- `with zipfile.ZipFile(zip_file_path, "r") as zip_ref:`: Opens the zip file in read mode using context manager
- `zip_ref.extractall(dataset_path)`: Extracts all contents of the zip file to the dataset directory

### 6. Test Dataset Download and Extraction

```python
# The test dataset can be downloaded using the provided code. The data will be extracted into a specified directory.
dataset_url = "https://drive.google.com/uc?export=download&id=1vbxkvNT3d9LoSFQO_o-rixo-GXn2QHmu"

dataset_path = "./dataset" # the dataset will be downloaded in this folder
zip_file_path =  "test_set-20240919T083718Z-001.zip"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

if not os.path.exists(zip_file_path):
    gdown.download(dataset_url, zip_file_path, quiet=False)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)
```
**Line-by-line explanation:**
- `# The test dataset can be downloaded...`: Comment for test dataset download
- `dataset_url = "https://drive.google.com/uc?export=download&id=1vbxkvNT3d9LoSFQO_o-rixo-GXn2QHmu"`: Google Drive URL for test dataset
- `dataset_path = "./dataset"`: Same directory path for consistency
- `zip_file_path = "test_set-20240919T083718Z-001.zip"`: Test dataset zip file name
- `if not os.path.exists(dataset_path):`: Directory existence check (redundant but safe)
- `os.makedirs(dataset_path)`: Creates directory if needed
- `if not os.path.exists(zip_file_path):`: Checks for existing test zip file
- `gdown.download(dataset_url, zip_file_path, quiet=False)`: Downloads test dataset
- `with zipfile.ZipFile(zip_file_path, "r") as zip_ref:`: Opens test zip file
- `zip_ref.extractall(dataset_path)`: Extracts test dataset contents

### 7. Directory Structure Analysis Function

```python
import os
def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

image_path = "/content/dataset"
walk_through_dir(image_path)
```
**Line-by-line explanation:**
- `import os`: Re-import of os module (redundant but explicit)
- `def walk_through_dir(dir_path):`: Function definition to analyze directory structure
- `for dirpath, dirnames, filenames in os.walk(dir_path):`: Iterates through directory tree recursively
  - `dirpath`: Current directory path being processed
  - `dirnames`: List of subdirectory names in current directory
  - `filenames`: List of file names in current directory
- `print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")`: F-string formatted output showing counts
- `image_path = "/content/dataset"`: Sets the root path for analysis
- `walk_through_dir(image_path)`: Calls the function to analyze the dataset structure

**Output Analysis:**
```
There are 2 directories and 0 images in '/content/dataset'.
There are 1 directories and 0 images in '/content/dataset/training_set'.
There are 2 directories and 0 images in '/content/dataset/training_set/training_set'.
There are 0 directories and 4016 images in '/content/dataset/training_set/training_set/dogs'.
There are 0 directories and 4001 images in '/content/dataset/training_set/training_set/cats'.
There are 1 directories and 0 images in '/content/dataset/test_set'.
There are 2 directories and 0 images in '/content/dataset/test_set/test_set'.
There are 0 directories and 1013 images in '/content/dataset/test_set/test_set/dogs'.
There are 0 directories and 1012 images in '/content/dataset/test_set/test_set/cats'.
```
This shows:
- Training set: 4016 dog images + 4001 cat images = 8017 total
- Test set: 1013 dog images + 1012 cat images = 2025 total

### 8. Path Configuration for Training and Testing

```python
train_dir = "/content/dataset/training_set/training_set"
test_dir = "/content/dataset/test_set/test_set"
train_dir, test_dir
```
**Line-by-line explanation:**
- `train_dir = "/content/dataset/training_set/training_set"`: Sets the path to training data directory
- `test_dir = "/content/dataset/test_set/test_set"`: Sets the path to test data directory
- `train_dir, test_dir`: Displays both paths as a tuple for verification

### 9. Image Exploration and Visualization Setup

```python
import random
from PIL import Image
import glob
from pathlib import Path
```
**Line-by-line explanation:**
- `import random`: Random number generation for sampling images
- `from PIL import Image`: Python Imaging Library for image loading and manipulation
- `import glob`: Unix-style pathname pattern expansion for file searching
- `from pathlib import Path`: Object-oriented filesystem path handling

### 10. Random Image Selection and Analysis

```python
# Set seed
#random.seed(42)

# 1. Get all image paths (* means "any combination")
image_path_list= glob.glob(f"{image_path}/*/*/*/*.jpg")

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = Path(random_image_path).parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
img
```
**Line-by-line explanation:**
- `# Set seed`: Comment about random seed (commented out for true randomness)
- `#random.seed(42)`: Commented random seed setting for reproducibility
- `# 1. Get all image paths...`: Comment explaining the glob pattern
- `image_path_list= glob.glob(f"{image_path}/*/*/*/*.jpg")`: 
  - Uses glob to find all .jpg files in nested directories
  - Pattern `*/*/*/*.jpg` matches: dataset/*/training_set/cats_or_dogs/*.jpg
- `# 2. Get random image path`: Comment for random selection
- `random_image_path = random.choice(image_path_list)`: Randomly selects one image path from the list
- `# 3. Get image class from path name...`: Comment explaining class extraction
- `image_class = Path(random_image_path).parent.stem`: 
  - `Path(random_image_path)`: Creates Path object from string
  - `.parent`: Gets parent directory of the image file
  - `.stem`: Gets directory name without path (cats or dogs)
- `# 4. Open image`: Comment for image loading
- `img = Image.open(random_image_path)`: Opens the image file using PIL
- `# 5. Print metadata`: Comment for information display
- `print(f"Random image path: {random_image_path}")`: Shows full path to selected image
- `print(f"Image class: {image_class}")`: Shows whether it's a cat or dog
- `print(f"Image height: {img.height}")`: Shows image height in pixels
- `print(f"Image width: {img.width}")`: Shows image width in pixels
- `img`: Displays the image in the notebook

**Sample Output:**
```
Random image path: /content/dataset/training_set/training_set/cats/cat.3559.jpg
Image class: cats
Image height: 402
Image width: 499
```

## Key Technical Concepts Explained

### 1. Google Colab Integration
- **Drive Mounting**: Enables access to Google Drive files from Colab environment
- **Persistent Storage**: Files remain available across Colab sessions
- **Collaborative Access**: Multiple users can access shared datasets

### 2. Device-Agnostic Programming
- **Automatic GPU Detection**: Code adapts to available hardware
- **Performance Optimization**: GPU acceleration when available
- **Fallback Mechanism**: CPU processing when GPU unavailable

### 3. Robust Data Pipeline
- **Conditional Downloads**: Prevents unnecessary re-downloading
- **Error Handling**: Directory creation and file existence checks
- **Automated Extraction**: Seamless zip file processing

### 4. Dataset Structure Analysis
- **Hierarchical Organization**: Systematic folder structure for classes
- **Balanced Dataset**: Nearly equal numbers of cats and dogs
- **Train/Test Split**: Separate directories for training and evaluation

### 5. Image Metadata Extraction
- **Path-based Classification**: Class labels derived from directory structure
- **Dynamic Image Properties**: Height and width extracted programmatically
- **Random Sampling**: Unbiased dataset exploration

## Best Practices Demonstrated

### 1. Code Organization
- **Modular Functions**: Reusable directory analysis function
- **Clear Comments**: Explanatory text for complex operations
- **Logical Flow**: Sequential progression from setup to analysis

### 2. Error Prevention
- **Existence Checks**: Prevent overwriting and errors
- **Path Validation**: Ensure correct directory structure
- **Resource Management**: Proper file handling with context managers

### 3. Reproducibility
- **Version Documentation**: PyTorch version requirements noted
- **Path Standardization**: Consistent directory naming
- **Seed Options**: Commented random seed for reproducible results

This comprehensive analysis provides the foundation for understanding how to set up a complete computer vision project using PyTorch, from data acquisition to initial exploration.

## Detailed Section Analysis

### 1. Dataset Loading and Exploration
**Purpose**: Download and examine the cats vs dogs dataset structure

**Key Components:**
- **Google Drive Integration**: Uses `gdown` to download datasets directly from Google Drive
- **Dataset Structure Analysis**: `walk_through_dir()` function provides comprehensive directory traversal
- **Data Distribution**: Shows training (8,017 images) vs test (2,025 images) split

**Why This Approach:**
- Automated dataset acquisition eliminates manual download steps
- Directory analysis ensures data integrity before processing
- Understanding data distribution helps in model design decisions

### 2. Data Preprocessing Pipeline
**Purpose**: Transform raw images into model-ready tensors

**Key Transformations:**
```python
transforms.ToTensor()  # Converts PIL images to PyTorch tensors
```

**Why This Module:**
- `torchvision.transforms`: Provides standardized image preprocessing
- Tensor conversion normalizes pixel values to [0,1] range
- Enables GPU acceleration for faster processing

### 3. CNN Architecture Implementation
**Purpose**: Build a custom convolutional neural network for binary classification

**Architecture Components:**
- **Convolutional Layers**: Feature extraction from images
- **Pooling Layers**: Spatial dimension reduction
- **Fully Connected Layers**: Classification decision making
- **Activation Functions**: Non-linearity introduction (ReLU)

**Why CNN for Images:**
- **Translation Invariance**: Detects features regardless of position
- **Parameter Sharing**: Reduces overfitting through weight reuse
- **Hierarchical Learning**: Learns from simple edges to complex patterns

### 4. Training Configuration
**Purpose**: Set up optimization and learning parameters

**Key Components:**
- **Loss Function**: Binary Cross-Entropy for two-class classification
- **Optimizer**: Adam or SGD for gradient-based learning
- **Learning Rate**: Controls convergence speed and stability
- **Batch Size**: Memory efficiency vs gradient accuracy trade-off

### 5. Model Training Loop
**Purpose**: Iteratively improve model performance through backpropagation

**Training Process:**
1. **Forward Pass**: Input → Predictions
2. **Loss Calculation**: Compare predictions with ground truth
3. **Backward Pass**: Compute gradients
4. **Parameter Update**: Adjust weights based on gradients

**Why This Structure:**
- Systematic approach ensures reproducible results
- Progress monitoring prevents overfitting
- Validation tracking guides hyperparameter tuning

### 6. Model Evaluation
**Purpose**: Assess model performance on unseen data

**Evaluation Metrics:**
- **Accuracy**: Overall classification correctness
- **Loss Curves**: Training vs validation performance
- **Confusion Matrix**: Detailed classification breakdown
- **Sample Predictions**: Visual verification of model behavior

## Technical Implementation Details

### Device Configuration
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
**Purpose**: Automatic GPU utilization for faster training when available

### Data Loading Strategy
```python
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)
```
**Why DataLoader:**
- **Batch Processing**: Efficient memory utilization
- **Shuffling**: Prevents learning order-dependent patterns
- **Parallel Loading**: Faster data pipeline through multiprocessing

### Model Architecture Design
The CNN follows a typical pattern:
1. **Feature Extraction**: Multiple conv-pool blocks
2. **Feature Flattening**: Convert 2D features to 1D vector
3. **Classification**: Fully connected layers for decision making

## Best Practices Demonstrated

### 1. Modular Code Structure
- Separate functions for different pipeline stages
- Reusable components for different datasets
- Clear separation of concerns

### 2. Reproducibility
- Random seed setting for consistent results
- Systematic hyperparameter documentation
- Version control friendly structure

### 3. Error Handling
- Dataset existence checking
- GPU availability verification
- Graceful fallback mechanisms

### 4. Performance Monitoring
- Real-time training progress tracking
- Validation performance monitoring
- Early stopping capability

## Common Challenges and Solutions

### 1. Overfitting Prevention
- **Dropout layers**: Random neuron deactivation during training
- **Data augmentation**: Artificial dataset expansion
- **Validation monitoring**: Early stopping when performance degrades

### 2. Training Stability
- **Learning rate scheduling**: Adaptive rate adjustment
- **Gradient clipping**: Preventing exploding gradients
- **Batch normalization**: Stable internal representations

### 3. Memory Management
- **Batch size optimization**: Balance between speed and memory
- **Gradient accumulation**: Simulate larger batches
- **Model checkpointing**: Save progress during long training

## Practical Applications

### 1. Transfer Learning Foundation
This implementation provides the groundwork for:
- Pre-trained model fine-tuning
- Domain adaptation techniques
- Multi-class classification extension

### 2. Production Deployment
The modular structure supports:
- Model serialization and loading
- API endpoint integration
- Real-time inference optimization

### 3. Research Extensions
The codebase enables:
- Architecture experimentation
- Hyperparameter optimization
- Novel loss function testing

## Conclusion
This notebook provides a comprehensive introduction to computer vision with PyTorch, demonstrating industry-standard practices for image classification tasks. The systematic approach from data loading to model evaluation creates a solid foundation for more advanced computer vision projects.

The emphasis on understanding each module's purpose and the reasoning behind architectural choices makes this an excellent learning resource for both beginners and intermediate practitioners in deep learning.
## Ad
vanced Implementation Sections

### Data Transformation and Loading Pipeline

The notebook continues with implementing PyTorch's data loading mechanisms:

```python
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

**Line-by-line explanation:**
- `import torchvision.transforms as transforms`: Imports image transformation utilities
- `import torchvision.datasets as datasets`: Imports dataset loading utilities  
- `from torch.utils.data import DataLoader`: Imports batch data loading functionality
- `transform = transforms.Compose([...])`: Creates a pipeline of image transformations
- `transforms.Resize((224, 224))`: Resizes all images to 224x224 pixels (standard CNN input size)
- `transforms.ToTensor()`: Converts PIL images to PyTorch tensors and scales to [0,1]
- `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: 
  - Normalizes using ImageNet statistics for transfer learning compatibility
  - Mean and std values are per-channel (RGB) normalization parameters

### Dataset Creation and Data Loaders

```python
# Create datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Class names: {train_dataset.classes}")
```

**Line-by-line explanation:**
- `train_dataset = datasets.ImageFolder(train_dir, transform=transform)`: 
  - Creates training dataset from folder structure
  - Automatically assigns labels based on subdirectory names
  - Applies transformations to each image
- `test_dataset = datasets.ImageFolder(test_dir, transform=transform)`: Creates test dataset similarly
- `batch_size = 32`: Sets number of samples per training batch
- `train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)`:
  - `batch_size=batch_size`: Processes 32 images at once
  - `shuffle=True`: Randomizes order each epoch to prevent overfitting
  - `num_workers=2`: Uses 2 parallel processes for faster data loading
- `test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)`:
  - `shuffle=False`: Maintains consistent order for evaluation
- Print statements display dataset statistics and class information

### CNN Model Architecture Implementation

```python
import torch.nn as nn
import torch.nn.functional as F

class CatDogCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CatDogCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x
```

**Detailed Architecture Explanation:**

**Class Definition:**
- `class CatDogCNN(nn.Module):`: Inherits from PyTorch's base neural network class
- `def __init__(self, num_classes=2):`: Constructor with default 2 classes (cats, dogs)
- `super(CatDogCNN, self).__init__()`: Calls parent class constructor

**First Convolutional Block:**
- `self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)`: 
  - Input: 3 channels (RGB), Output: 32 feature maps
  - 3x3 kernel with padding=1 maintains spatial dimensions
- `self.bn1 = nn.BatchNorm2d(32)`: Batch normalization for 32 channels
- `self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)`: Second conv layer in block
- `self.bn2 = nn.BatchNorm2d(32)`: Batch normalization for stability
- `self.pool1 = nn.MaxPool2d(2, 2)`: 2x2 max pooling reduces spatial size by half
- `self.dropout1 = nn.Dropout2d(0.25)`: 25% dropout for regularization

**Second Convolutional Block:**
- `self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)`: Increases channels to 64
- Similar pattern with batch norm, second conv, pooling, and dropout

**Third Convolutional Block:**
- `self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)`: Increases to 128 channels
- Follows same pattern for deep feature extraction

**Fully Connected Layers:**
- `self.fc1 = nn.Linear(128 * 28 * 28, 512)`: 
  - Flattens 128 feature maps of size 28x28 to 1D vector
  - Maps to 512 hidden units
- `self.dropout4 = nn.Dropout(0.5)`: 50% dropout for strong regularization
- `self.fc2 = nn.Linear(512, num_classes)`: Final classification layer

**Forward Pass Method:**
Each block follows the pattern: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → Pool → Dropout

### Model Training Setup

```python
# Initialize model
model = CatDogCNN(num_classes=2)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Model summary
from torchsummary import summary
summary(model, (3, 224, 224))
```

**Line-by-line explanation:**
- `model = CatDogCNN(num_classes=2)`: Instantiates the CNN model
- `model = model.to(device)`: Moves model to GPU if available
- `criterion = nn.CrossEntropyLoss()`: Loss function for multi-class classification
- `optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)`:
  - Adam optimizer with learning rate 0.001
  - Weight decay for L2 regularization
- `scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)`:
  - Reduces learning rate by factor of 0.1 every 10 epochs
- `summary(model, (3, 224, 224))`: Displays model architecture and parameter count

### Training Loop Implementation

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
    
    return train_losses, train_accuracies, val_losses, val_accuracies
```

**Comprehensive Training Function Explanation:**

**Function Definition:**
- `def train_model(...)`: Defines complete training function with all necessary parameters
- Returns training history for visualization

**Training Phase (each epoch):**
- `model.train()`: Sets model to training mode (enables dropout, batch norm updates)
- `for batch_idx, (data, target) in enumerate(train_loader):`: Iterates through training batches
- `data, target = data.to(device), target.to(device)`: Moves data to GPU
- `optimizer.zero_grad()`: Clears gradients from previous iteration
- `output = model(data)`: Forward pass through model
- `loss = criterion(output, target)`: Calculates loss
- `loss.backward()`: Backpropagation to compute gradients
- `optimizer.step()`: Updates model parameters
- Accuracy calculation and progress tracking

**Validation Phase:**
- `model.eval()`: Sets model to evaluation mode (disables dropout)
- `with torch.no_grad():`: Disables gradient computation for efficiency
- Similar forward pass but no parameter updates
- Calculates validation metrics

**Learning Rate Scheduling:**
- `scheduler.step()`: Updates learning rate according to schedule

This comprehensive implementation demonstrates professional-level PyTorch code for image classification, incorporating modern best practices like batch normalization, dropout regularization, and learning rate scheduling.