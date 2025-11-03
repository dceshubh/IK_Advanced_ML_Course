# Student Copy - Autoencoders Emoji - Coding Guide

## Overview
This notebook demonstrates **Convolutional Autoencoders** for **image compression and reconstruction** using an emoji dataset. It shows how to build encoder-decoder architectures for dimensionality reduction of image data and visualizes the learned compressed representations using PCA.

## Key Learning Objectives
- Build Convolutional Autoencoders for image data
- Understand encoder-decoder architecture for image compression
- Compare original vs compressed image representations
- Visualize high-dimensional embeddings using PCA
- Learn image preprocessing techniques for neural networks

---

## 1. Library Installation and Imports

```python
# Install HuggingFace datasets library
!pip install -U datasets

# Core TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model

# Data handling and visualization
import numpy as np
from datasets import load_dataset
from IPython.display import HTML
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Dimensionality reduction
from sklearn.decomposition import PCA
```

### Why These Libraries?
- **datasets**: HuggingFace library for easy access to ML datasets
- **tensorflow.keras**: High-level API for building neural networks
- **Conv2D/MaxPooling2D**: Convolutional layers for spatial feature extraction
- **UpSampling2D**: Upsampling layers for image reconstruction
- **PIL (Pillow)**: Python Imaging Library for image processing
- **PCA**: Principal Component Analysis for visualization

---

## 2. Dataset Loading and Exploration

### 2.1 Emoji Dataset

```python
# Load emoji dataset from HuggingFace
dataset = load_dataset("valhalla/emoji-dataset")

# Explore dataset structure
dataset['train']
```

**Dataset Details:**
- **Source**: HuggingFace emoji dataset collection
- **Content**: Various emoji images in different styles
- **Format**: PIL Image objects
- **Size**: Multiple emoji variations and styles
- **Task**: Learn to compress and reconstruct emoji images

### 2.2 Image Visualization

```python
def view_images(images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 8))
    
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Display first 8 emoji images
images = [emoji for emoji in dataset['train'][0:8]['image']]
view_images(images)
```

**Visualization Purpose:**
- **Data exploration**: Understand image content and variety
- **Quality check**: Ensure images are properly loaded
- **Preprocessing planning**: Determine necessary transformations

---

## 3. Data Preprocessing

### 3.1 Image Conversion and Normalization

```python
def pil_to_np_array(images):
    """Convert PIL images to grayscale numpy arrays"""
    grayscale_images = []
    
    for image in images:
        # Convert to grayscale (L mode)
        img = image.convert('L')
        
        # Convert to numpy array and normalize to [0,1]
        img = img_to_array(img) / 255.0
        
        grayscale_images.append(img)
    
    # Stack into single numpy array
    img_array = np.array(grayscale_images)
    return img_array
```

**Preprocessing Steps:**
1. **Grayscale conversion**: Reduce from RGB (3 channels) to grayscale (1 channel)
2. **Normalization**: Scale pixel values from [0,255] to [0,1] range
3. **Array conversion**: Convert PIL Images to numpy arrays for TensorFlow
4. **Batch creation**: Stack individual images into batch format

**Why Grayscale?**
- **Simplicity**: Reduces complexity for learning demonstration
- **Computational efficiency**: 3x fewer parameters than RGB
- **Focus on structure**: Emphasizes shape over color information

### 3.2 Train-Test Split

```python
def list_splitter(list_to_split, ratio):
    """Split list into train/test based on ratio"""
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]

# Process all images and split
images = dataset['train']['image']
images_np = pil_to_np_array(images)
X_train, X_test = list_splitter(images_np, 0.7)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
```

**Split Strategy:**
- **70% training**: Used for learning autoencoder parameters
- **30% testing**: Used for evaluation and visualization
- **No validation set**: Simple demonstration setup
- **Sequential split**: Takes first 70% for training (could be randomized)

---

## 4. Convolutional Autoencoder Architecture

### 4.1 Encoder Design

```python
# Input shape for 256x256 grayscale images
input_shape = (256, 256, 1)

# Encoder Input Layer
encoder_input = Input(shape=input_shape, name='encoder_input')

# Encoder Layer 1: Convolution + Pooling
encoder_hidden = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
encoder_hidden = MaxPooling2D((2, 2), padding='same')(encoder_hidden)
# Output: (128, 128, 16)

# Encoder Layer 2: Convolution + Pooling  
encoder_hidden = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_hidden)
encoder_hidden = MaxPooling2D((2, 2), padding='same')(encoder_hidden)
# Output: (64, 64, 8)

# Encoder Layer 3: Final Compression
encoder_hidden = Conv2D(1, (3, 3), activation='relu', padding='same')(encoder_hidden)
encoder_output = MaxPooling2D((2, 2), padding='same', name='encoder_output')(encoder_hidden)
# Output: (32, 32, 1) - Compressed representation
```

**Encoder Architecture Breakdown:**

**Layer-by-layer Compression:**
1. **Input**: (256, 256, 1) → 65,536 pixels
2. **Conv2D(16) + MaxPool**: (128, 128, 16) → 262,144 values
3. **Conv2D(8) + MaxPool**: (64, 64, 8) → 32,768 values  
4. **Conv2D(1) + MaxPool**: (32, 32, 1) → 1,024 values

**Compression Ratio**: 65,536 → 1,024 = **64:1 compression**

**Key Design Choices:**
- **Conv2D layers**: Extract spatial features while preserving locality
- **MaxPooling2D**: Downsample spatial dimensions (2x2 → 4x reduction)
- **ReLU activation**: Non-linear feature learning
- **'same' padding**: Maintain spatial dimensions through convolutions
- **Decreasing filters**: 16 → 8 → 1 (progressive compression)

### 4.2 Decoder Design

```python
# Decoder Input (starts from encoder output)
decoder_hidden = Conv2D(1, (3, 3), activation='relu', padding='same', 
                       name='decoder_input')(encoder_output)

# Decoder Layer 1: Upsampling + Convolution
decoder_hidden = UpSampling2D((2, 2))(decoder_hidden)
decoder_hidden = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_hidden)
# Output: (64, 64, 8)

# Decoder Layer 2: Upsampling + Convolution
decoder_hidden = UpSampling2D((2, 2))(decoder_hidden)
decoder_hidden = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_hidden)
# Output: (128, 128, 16)

# Decoder Layer 3: Final Reconstruction
decoder_hidden = UpSampling2D((2, 2))(decoder_hidden)
decoder_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same', 
                       name='decoder_output')(decoder_hidden)
# Output: (256, 256, 1) - Reconstructed image
```

**Decoder Architecture Breakdown:**

**Layer-by-layer Reconstruction:**
1. **Input**: (32, 32, 1) → 1,024 values
2. **UpSample + Conv2D(8)**: (64, 64, 8) → 32,768 values
3. **UpSample + Conv2D(16)**: (128, 128, 16) → 262,144 values
4. **UpSample + Conv2D(1)**: (256, 256, 1) → 65,536 pixels

**Key Design Choices:**
- **UpSampling2D**: Increase spatial dimensions (reverse of MaxPooling)
- **Symmetric architecture**: Mirror of encoder for balanced reconstruction
- **Increasing filters**: 1 → 8 → 16 (progressive feature expansion)
- **Sigmoid output**: Ensures pixel values in [0,1] range
- **Same spatial dimensions**: Input and output both (256, 256, 1)

---

## 5. Model Compilation and Training

### 5.1 Model Creation and Compilation

```python
# Create complete autoencoder model
autoencoder = Model(inputs=encoder_input, outputs=decoder_output)

# Compile with appropriate loss and optimizer
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Visualize model architecture
plot_model(autoencoder, to_file='autoencoder_architecture.png', show_shapes=True)
```

**Model Configuration:**
- **Input**: encoder_input layer
- **Output**: decoder_output layer  
- **Optimizer**: Adam (adaptive learning rate)
- **Loss function**: MSE (Mean Squared Error) for pixel-wise reconstruction
- **Architecture visualization**: Shows layer connections and shapes

**Why MSE Loss?**
- **Pixel-wise comparison**: Measures difference between original and reconstructed pixels
- **Continuous values**: Appropriate for normalized pixel values [0,1]
- **Differentiable**: Enables gradient-based optimization
- **Intuitive**: Lower MSE = better reconstruction quality

### 5.2 Training Process

```python
# Train autoencoder (input = target for reconstruction)
history = autoencoder.fit(
    X_train, X_train,           # Input and target are the same
    epochs=50,                  # Number of training iterations
    batch_size=128,             # Process 128 images per batch
    validation_data=(X_test, X_test)  # Validation on test set
)
```

**Training Configuration:**
- **Self-supervised**: Input images are also the targets
- **50 epochs**: Sufficient for convergence on small dataset
- **Batch size 128**: Balance between memory usage and gradient stability
- **Validation monitoring**: Track overfitting during training

**Training Objective:**
```
minimize: MSE(original_image, reconstructed_image)
```

### 5.3 Training Visualization

```python
# Extract training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training curves
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```

**Training Curve Analysis:**
- **Decreasing loss**: Model learning to reconstruct images
- **Convergence**: Loss stabilizes after sufficient epochs
- **Overfitting check**: Large gap between train/validation indicates overfitting
- **Optimal stopping**: Best validation performance point

---

## 6. Results Visualization and Analysis

### 6.1 Image Reconstruction Comparison

```python
# Select random test images for reconstruction
num_images = 5
sample_indices = np.random.choice(len(X_test), num_images, replace=False)
sample_images = X_test[sample_indices]

# Reshape for model input
sample_images_reshaped = sample_images.reshape(num_images, 256, 256, 1)

# Generate reconstructions
reconstructed_images = autoencoder.predict(sample_images_reshaped)

# Visualize original vs reconstructed
plt.figure(figsize=(10, 4))
for i in range(num_images):
    # Original image
    plt.subplot(2, num_images, i + 1)
    plt.imshow(sample_images[i], cmap='gray')
    plt.axis('off')
    plt.title('Original')
    
    # Reconstructed image
    plt.subplot(2, num_images, num_images + i + 1)
    plt.imshow(reconstructed_images[i], cmap='gray')
    plt.axis('off')
    plt.title('Reconstructed')

plt.tight_layout()
plt.show()
```

**Reconstruction Quality Assessment:**
- **Visual comparison**: Side-by-side original vs reconstructed
- **Detail preservation**: How well fine details are maintained
- **Compression artifacts**: Blurring or loss of sharp edges
- **Overall similarity**: Global structure and recognizability

---

## 7. Embedding Analysis and Visualization

### 7.1 Encoder Feature Extraction

```python
# Extract encoder portion of the model
encoder = Model(inputs=autoencoder.input, 
               outputs=autoencoder.get_layer('encoder_output').output)

# Generate compressed embeddings
compressed_embeddings = encoder.predict(X_test)
print(f"Original shape: {X_test.shape}")
print(f"Compressed shape: {compressed_embeddings.shape}")
```

**Embedding Extraction:**
- **Encoder isolation**: Extract only the encoding portion
- **Compressed representation**: (32, 32, 1) = 1,024 dimensions
- **Feature learning**: Encoder learns meaningful compressed features
- **Dimensionality reduction**: 65,536 → 1,024 (64x compression)

### 7.2 PCA Visualization of Embeddings

```python
# Prepare original embeddings for PCA
original_embeddings_flat = X_test.reshape(X_test.shape[0], -1)
pca = PCA(n_components=2)
original_pca_embeddings = pca.fit_transform(original_embeddings_flat)

# Prepare compressed embeddings for PCA  
compressed_embeddings_flat = compressed_embeddings.reshape(compressed_embeddings.shape[0], -1)
pca = PCA(n_components=2)
compressed_pca_embeddings = pca.fit_transform(compressed_embeddings_flat)

print(f"Original flattened: {original_embeddings_flat.shape}")
print(f"Compressed flattened: {compressed_embeddings_flat.shape}")
print(f"Original PCA: {original_pca_embeddings.shape}")
print(f"Compressed PCA: {compressed_pca_embeddings.shape}")
```

**PCA Analysis Purpose:**
- **Dimensionality reduction**: Reduce to 2D for visualization
- **Pattern discovery**: Identify clusters and relationships
- **Compression evaluation**: Compare original vs compressed distributions
- **Feature quality**: Assess if compressed features preserve structure

### 7.3 Embedding Visualization

```python
# Visualize original vs compressed embeddings
plt.figure(figsize=(10, 5))

# Plot original images in 2D PCA space
plt.subplot(1, 2, 1)
plt.scatter(original_pca_embeddings[:, 0], original_pca_embeddings[:, 1], 
           c='blue', alpha=0.5)
plt.title("Original Images")

# Plot compressed images in 2D PCA space
plt.subplot(1, 2, 2)
plt.scatter(compressed_pca_embeddings[:, 0], compressed_pca_embeddings[:, 1], 
           c='red', alpha=0.5)
plt.title("Compressed Images")

plt.tight_layout()
plt.show()
```

**Visualization Insights:**
- **Cluster preservation**: Similar images should cluster together
- **Structure maintenance**: Compressed embeddings should preserve relationships
- **Dimensionality effect**: How compression affects data distribution
- **Feature quality**: Tight clusters indicate good feature learning

---

## 8. Advanced Analysis and Concepts

### 8.1 Compression Ratio Analysis

```python
# Calculate compression statistics
original_size = np.prod(X_test.shape[1:])  # 256 * 256 * 1 = 65,536
compressed_size = np.prod(compressed_embeddings.shape[1:])  # 32 * 32 * 1 = 1,024

compression_ratio = original_size / compressed_size
space_savings = (1 - compressed_size / original_size) * 100

print(f"Original size per image: {original_size:,} pixels")
print(f"Compressed size per image: {compressed_size:,} values")
print(f"Compression ratio: {compression_ratio:.1f}:1")
print(f"Space savings: {space_savings:.1f}%")
```

### 8.2 Reconstruction Quality Metrics

```python
# Calculate reconstruction metrics
def calculate_metrics(original, reconstructed):
    # Mean Squared Error
    mse = np.mean((original - reconstructed) ** 2)
    
    # Peak Signal-to-Noise Ratio
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Structural Similarity Index (if available)
    return mse, psnr

# Evaluate on test set
test_reconstructed = autoencoder.predict(X_test)
mse, psnr = calculate_metrics(X_test, test_reconstructed)

print(f"Mean Squared Error: {mse:.6f}")
print(f"Peak Signal-to-Noise Ratio: {psnr:.2f} dB")
```

### 8.3 Latent Space Interpolation

```python
# Interpolate between two images in latent space
def interpolate_images(img1, img2, steps=10):
    # Encode both images
    encoded1 = encoder.predict(img1.reshape(1, 256, 256, 1))
    encoded2 = encoder.predict(img2.reshape(1, 256, 256, 1))
    
    # Create interpolation path
    interpolations = []
    for i in range(steps):
        alpha = i / (steps - 1)
        interpolated = (1 - alpha) * encoded1 + alpha * encoded2
        
        # Decode interpolated representation
        decoder = Model(inputs=autoencoder.get_layer('decoder_input').input,
                       outputs=autoencoder.output)
        reconstructed = decoder.predict(interpolated)
        interpolations.append(reconstructed[0])
    
    return interpolations

# Example interpolation between two test images
img1, img2 = X_test[0], X_test[1]
interpolated_sequence = interpolate_images(img1, img2, steps=8)

# Visualize interpolation
plt.figure(figsize=(12, 2))
for i, img in enumerate(interpolated_sequence):
    plt.subplot(1, len(interpolated_sequence), i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'Step {i}')
plt.show()
```

---

## 9. Applications and Extensions

### 9.1 Image Denoising

```python
# Add noise to images for denoising task
def add_noise(images, noise_factor=0.3):
    noise = np.random.normal(0, noise_factor, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)

# Create noisy test images
noisy_test = add_noise(X_test, noise_factor=0.2)

# Train denoising autoencoder (noisy input, clean target)
denoising_history = autoencoder.fit(
    noisy_test, X_test,
    epochs=20,
    batch_size=128,
    validation_split=0.2
)
```

### 9.2 Anomaly Detection

```python
# Use reconstruction error for anomaly detection
def detect_anomalies(images, threshold_percentile=95):
    reconstructed = autoencoder.predict(images)
    reconstruction_errors = np.mean((images - reconstructed) ** 2, axis=(1, 2, 3))
    
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    anomalies = reconstruction_errors > threshold
    
    return anomalies, reconstruction_errors

# Example anomaly detection
anomalies, errors = detect_anomalies(X_test)
print(f"Detected {np.sum(anomalies)} anomalies out of {len(X_test)} images")
```

### 9.3 Feature Learning for Classification

```python
# Use encoder features for downstream classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Extract features using trained encoder
train_features = encoder.predict(X_train).reshape(len(X_train), -1)
test_features = encoder.predict(X_test).reshape(len(X_test), -1)

# Train classifier on compressed features (if labels available)
# classifier = SVC()
# classifier.fit(train_features, train_labels)
# predictions = classifier.predict(test_features)
```

---

## 10. Key Takeaways and Best Practices

### Technical Insights
1. **Convolutional layers**: Better for images than fully connected layers
2. **Symmetric architecture**: Encoder-decoder symmetry aids reconstruction
3. **Progressive compression**: Gradual dimensionality reduction works better
4. **Activation functions**: ReLU for hidden layers, sigmoid for output

### Practical Considerations
1. **Preprocessing**: Normalization crucial for stable training
2. **Loss function**: MSE appropriate for continuous pixel values
3. **Batch size**: Balance between memory and gradient quality
4. **Validation**: Monitor overfitting during training

### Common Issues and Solutions
1. **Blurry reconstructions**: Try different loss functions (perceptual loss)
2. **Mode collapse**: Ensure sufficient model capacity
3. **Training instability**: Reduce learning rate or add regularization
4. **Poor compression**: Increase bottleneck size or add more layers

### Extensions and Improvements
1. **Variational Autoencoders**: Add probabilistic latent space
2. **Adversarial training**: Use GANs for sharper reconstructions
3. **Attention mechanisms**: Focus on important image regions
4. **Multi-scale architectures**: Process different resolutions

---

## Summary

This notebook demonstrates the fundamental concepts of Convolutional Autoencoders through a practical emoji image compression task. Key achievements include:

1. **64:1 compression ratio** while maintaining visual quality
2. **Learned feature representations** that preserve image structure
3. **Visualization techniques** for understanding compressed embeddings
4. **Practical implementation** of encoder-decoder architectures

The autoencoder successfully learns to compress emoji images into a compact representation and reconstruct them with reasonable fidelity, demonstrating the power of deep learning for unsupervised feature learning and dimensionality reduction in computer vision tasks.