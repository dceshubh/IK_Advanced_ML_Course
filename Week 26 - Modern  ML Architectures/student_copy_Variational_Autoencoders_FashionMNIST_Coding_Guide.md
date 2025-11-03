# Student Copy - Variational Autoencoders FashionMNIST - Coding Guide

## Overview
This notebook demonstrates **Variational Autoencoders (VAEs)** for **generative modeling** using the Fashion-MNIST dataset. Unlike regular autoencoders, VAEs learn a probabilistic latent space that enables generation of new, realistic images by sampling from the learned distribution.

## Key Learning Objectives
- Understand the difference between Autoencoders and Variational Autoencoders
- Implement the reparameterization trick for backpropagation through stochastic layers
- Build encoder-decoder architectures with probabilistic latent spaces
- Generate new images by sampling from the learned latent distribution
- Visualize and explore the continuous latent space structure

---

## 1. Library Installation and Imports

```python
# Install specific TensorFlow version
!pip install tensorflow

# Core TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

# Neural network layers
from tensorflow.keras.layers import (Input, Dense, Conv2D, MaxPooling2D, 
                                   UpSampling2D, Flatten, Lambda, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model

# Data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check TensorFlow version
print(tf.__version__)
```

### Why These Libraries?
- **tensorflow/keras**: High-level deep learning framework
- **keras.backend**: Low-level operations for custom loss functions
- **Lambda layer**: Execute custom functions within the model
- **Fashion-MNIST**: Built-in dataset for clothing item classification
- **matplotlib**: Visualization of generated images and latent space

---

## 2. Dataset Loading and Preprocessing

### 2.1 Fashion-MNIST Dataset

```python
# Load Fashion-MNIST dataset
(X_train, _), (X_test, _) = keras.datasets.fashion_mnist.load_data()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Normalize pixel values to [0,1] range
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape to add channel dimension
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))
```

**Dataset Details:**
- **Fashion-MNIST**: 70,000 grayscale images of clothing items
- **Categories**: T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, ankle boots
- **Image size**: 28×28 pixels (same as MNIST)
- **Channels**: 1 (grayscale)
- **Task**: Learn to generate new fashion items

**Preprocessing Steps:**
1. **Normalization**: Scale pixels from [0,255] to [0,1]
2. **Reshape**: Add channel dimension for CNN compatibility
3. **Type conversion**: Float32 for neural network training

### 2.2 Data Visualization

```python
def view_images(images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(8, 8))
    
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Display first 5 training images
images = [img for img in X_train[0:5]]
view_images(images)
```

---

## 3. Variational Autoencoder Architecture

### 3.1 Encoder Design

```python
# Architecture parameters
input_shape = (28, 28, 1)
latent_dim = 2  # 2D latent space for easy visualization

# Encoder Input
encoder_input = Input(shape=input_shape)

# Convolutional layers for feature extraction
encoder_hidden = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
encoder_hidden = MaxPooling2D((2, 2), padding='same')(encoder_hidden)
# Output: (14, 14, 16)

encoder_hidden = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_hidden)
encoder_hidden = MaxPooling2D((2, 2), padding='same')(encoder_hidden)
# Output: (7, 7, 8)

encoder_hidden = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder_hidden)
encoder_hidden = MaxPooling2D((2, 2), padding='same', name='encoder_output')(encoder_hidden)
# Output: (4, 4, 4) - Note: 7/2 rounded up = 4

# Flatten for dense layers
encoder_hidden = Flatten()(encoder_hidden)
# Output: 64 values (4×4×4)
```

**Encoder Architecture:**
- **Progressive downsampling**: 28×28 → 14×14 → 7×7 → 4×4
- **Feature extraction**: Convolutional layers learn spatial patterns
- **Dimensionality reduction**: From 784 pixels to 64 features
- **Flattening**: Prepare for probabilistic layers

### 3.2 Probabilistic Latent Space

```python
# Mean and log-variance vectors for latent distribution
z_mean = Dense(latent_dim, name='z_mean')(encoder_hidden)
z_log_var = Dense(latent_dim, name='z_log_var')(encoder_hidden)
```

**Key Innovation of VAEs:**
- **Regular Autoencoder**: Deterministic encoding z = f(x)
- **Variational Autoencoder**: Probabilistic encoding z ~ N(μ(x), σ²(x))

**Latent Distribution Parameters:**
- **z_mean**: Mean vector μ of the latent Gaussian distribution
- **z_log_var**: Log-variance vector log(σ²) for numerical stability
- **Latent dimension**: 2D for visualization (typically 10-100 in practice)

### 3.3 Reparameterization Trick

```python
def sampling(args):
    """Reparameterization trick for sampling from latent distribution"""
    z_mean, z_log_var = args
    
    # Sample random noise from standard normal distribution
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    
    # Convert log-variance to standard deviation
    standard_deviation = tf.exp(0.5 * z_log_var)
    
    # Reparameterized sample: z = μ + σ * ε
    return z_mean + standard_deviation * epsilon

# Lambda layer to execute sampling function
latent_space = Lambda(sampling, name='z')([z_mean, z_log_var])
```

**Reparameterization Trick Explained:**

**Problem**: Cannot backpropagate through random sampling
```python
# This doesn't work for gradient descent:
z = sample_from_normal(mean=μ, variance=σ²)
```

**Solution**: Express sampling as deterministic function + external noise
```python
# This works for backpropagation:
z = μ + σ * ε, where ε ~ N(0, 1)
```

**Mathematical Foundation:**
- **Original**: z ~ N(μ, σ²)
- **Reparameterized**: z = μ + σ * ε, where ε ~ N(0, 1)
- **Gradient flow**: ∇μ and ∇σ can be computed through deterministic operations

### 3.4 Complete Encoder

```python
# Create encoder model
encoder = Model(encoder_input, latent_space, name='encoder')
encoder.summary()

# Visualize encoder architecture
plot_model(encoder, to_file='encoder_architecture.png', show_shapes=True)
```

---

## 4. Decoder Architecture

### 4.1 Decoder Design

```python
# Decoder Input (latent vector)
decoder_input = Input(shape=(latent_dim,))

# Dense layer to expand latent vector
decoder_hidden = Dense(7 * 7 * 64, activation='relu')(decoder_input)

# Reshape to spatial dimensions
decoder_hidden = Reshape((7, 7, 64))(decoder_hidden)

# Convolutional layers for image reconstruction
decoder_hidden = Conv2D(4, (3, 3), activation='relu', padding='same', 
                       name='decoder_input')(decoder_hidden)

# Upsampling layers (reverse of encoder)
decoder_hidden = UpSampling2D((2, 2))(decoder_hidden)
# Output: (14, 14, 4)

decoder_hidden = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_hidden)
decoder_hidden = UpSampling2D((2, 2))(decoder_hidden)
# Output: (28, 28, 8)

decoder_hidden = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_hidden)

# Final output layer
decoder_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same', 
                       name='decoder_output')(decoder_hidden)
# Output: (28, 28, 1)

# Create decoder model
decoder = Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

# Visualize decoder architecture
plot_model(decoder, to_file='decoder_architecture.png', show_shapes=True)
```

**Decoder Architecture:**
- **Latent expansion**: 2D → 7×7×64 (3,136 values)
- **Spatial reconstruction**: Reshape to spatial dimensions
- **Progressive upsampling**: 7×7 → 14×14 → 28×28
- **Feature synthesis**: Generate image features from latent code
- **Sigmoid output**: Ensure pixel values in [0,1] range

---

## 5. Complete VAE Model and Loss Function

### 5.1 VAE Model Assembly

```python
# Connect encoder and decoder
encoder_output = encoder(encoder_input)
decoder_output = decoder(encoder_output)

# Create complete VAE model
vae = Model(encoder_input, decoder_output, name='vae')
vae.summary()
```

### 5.2 Custom VAE Loss Function

```python
from keras import layers, ops

class VAELossLayer(layers.Layer):
    def __init__(self, reconstruction_loss_factor=100, **kwargs):
        super().__init__(**kwargs)
        self.reconstruction_loss_factor = reconstruction_loss_factor
    
    def call(self, inputs):
        input_image, output_image, z_mean, z_log_var = inputs
        
        # Reconstruction loss (MSE between input and output)
        reconstruction_loss = self.reconstruction_loss_factor * ops.mean(
            ops.square(input_image - output_image), axis=[1, 2, 3]
        )
        
        # KL divergence loss (regularization term)
        kl_loss = -0.5 * ops.sum(
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1
        )
        
        # Total VAE loss
        total_loss = reconstruction_loss + kl_loss
        
        # Register loss with the model
        self.add_loss(ops.mean(total_loss))
        
        # Forward the output
        return output_image

# Apply loss layer
output_with_loss = VAELossLayer()([encoder_input, decoder_output, z_mean, z_log_var])

# Create final VAE model with loss
vae = Model(inputs=encoder_input, outputs=output_with_loss)
vae.compile(optimizer='adam')
```

**VAE Loss Components:**

**1. Reconstruction Loss:**
```python
L_reconstruction = ||x - x̂||²
```
- Measures how well the decoder reconstructs the input
- Higher weight (×100) emphasizes reconstruction quality

**2. KL Divergence Loss:**
```python
L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```
- Regularizes latent space to be close to standard normal N(0,1)
- Prevents overfitting and enables sampling

**3. Total VAE Loss:**
```python
L_VAE = L_reconstruction + L_KL
```

**Mathematical Intuition:**
- **Reconstruction term**: Encourages accurate image reconstruction
- **KL term**: Keeps latent space well-structured and continuous
- **Balance**: Trade-off between reconstruction quality and latent regularity

---

## 6. Training Process

### 6.1 Model Training

```python
# Train the VAE model
history = vae.fit(
    X_train, X_train,           # Input = target for reconstruction
    batch_size=128,             # Process 128 images per batch
    epochs=20,                  # Number of training iterations
    validation_data=(X_test, X_test)  # Validation monitoring
)
```

**Training Configuration:**
- **Self-supervised**: Input images are reconstruction targets
- **Batch size 128**: Balance between memory and gradient stability
- **20 epochs**: Sufficient for Fashion-MNIST convergence
- **Validation monitoring**: Track overfitting during training

### 6.2 Training Visualization

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
- **Decreasing loss**: Model learning to reconstruct and regularize
- **Convergence**: Loss stabilizes after sufficient training
- **Overfitting check**: Gap between train/validation loss
- **VAE-specific**: Loss includes both reconstruction and KL terms

---

## 7. Results and Generation

### 7.1 Image Reconstruction

```python
# Select random test images for reconstruction
num_images = 5
sample_indices = np.random.choice(len(X_test), num_images, replace=False)
sample_images = X_test[sample_indices]

# Reshape for model input
sample_images_reshaped = sample_images.reshape(num_images, 28, 28, 1)

# Generate reconstructions
reconstructed_images = vae.predict(sample_images_reshaped)

# Visualize original vs reconstructed
plt.figure(figsize=(5, 4))
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
    plt.title('Reconstructed', fontsize=8)

plt.tight_layout()
plt.show()
```

### 7.2 New Image Generation

```python
# Generate new images by sampling from latent space
num_vectors = 50
latent_dim = 2

# Sample random vectors from standard normal distribution
random_vectors = np.random.randn(num_vectors, latent_dim)

# Generate images using the decoder
generated_images = decoder(random_vectors)
generated_images = generated_images.numpy().reshape(num_vectors, 28, 28)

# Display generated images
fig, axes = plt.subplots(5, 10, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

**Generation Process:**
1. **Sample latent codes**: z ~ N(0, 1)
2. **Decode to images**: x̂ = decoder(z)
3. **Novel images**: Not in training set, but similar style

### 7.3 Latent Space Exploration

```python
# Create a grid of latent space samples
nx = ny = 10
meshgrid = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))
meshgrid = np.array(meshgrid).reshape(2, nx*ny).T

# Generate images from grid points
x_grid = decoder(meshgrid)
x_grid = x_grid.numpy().reshape(nx, ny, 28, 28, 1)

# Create canvas for visualization
canvas = np.zeros((nx*28, ny*28))
for xi in range(nx):
    for yi in range(ny):
        canvas[xi*28:xi*28+28, yi*28:yi*28+28] = x_grid[xi, yi, :, :, :].squeeze()

# Display latent space traversal
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(canvas, cmap=plt.cm.Greys)
ax.axis('off')
```

**Latent Space Visualization:**
- **Grid sampling**: Systematic exploration of 2D latent space
- **Continuous transitions**: Smooth changes between generated images
- **Semantic organization**: Similar items cluster together
- **Interpolation**: Gradual morphing between different clothing types

---

## 8. Advanced Analysis and Applications

### 8.1 Latent Space Interpolation

```python
def interpolate_images(img1, img2, steps=10):
    """Interpolate between two images in latent space"""
    # Encode both images to latent space
    z1 = encoder.predict(img1.reshape(1, 28, 28, 1))
    z2 = encoder.predict(img2.reshape(1, 28, 28, 1))
    
    # Create interpolation path
    interpolations = []
    for i in range(steps):
        alpha = i / (steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # Decode interpolated latent vector
        img_interp = decoder.predict(z_interp)
        interpolations.append(img_interp[0])
    
    return interpolations

# Example interpolation
img1, img2 = X_test[0], X_test[100]
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

### 8.2 Latent Space Arithmetic

```python
def latent_arithmetic(img1, img2, img3, operation='add'):
    """Perform arithmetic operations in latent space"""
    # Encode images to latent space
    z1 = encoder.predict(img1.reshape(1, 28, 28, 1))
    z2 = encoder.predict(img2.reshape(1, 28, 28, 1))
    z3 = encoder.predict(img3.reshape(1, 28, 28, 1))
    
    # Perform operation: z1 - z2 + z3
    if operation == 'analogy':
        z_result = z1 - z2 + z3
    
    # Decode result
    img_result = decoder.predict(z_result)
    return img_result[0]

# Example: "dress - formal + casual = casual dress"
# (This is conceptual - actual results depend on learned representations)
```

### 8.3 Clustering Analysis

```python
# Encode test set to latent space
latent_representations = encoder.predict(X_test)

# Visualize latent space distribution
plt.figure(figsize=(8, 6))
plt.scatter(latent_representations[:, 0], latent_representations[:, 1], 
           alpha=0.6, s=1)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space Distribution')
plt.show()

# Analyze latent space statistics
print(f"Latent space mean: {np.mean(latent_representations, axis=0)}")
print(f"Latent space std: {np.std(latent_representations, axis=0)}")
```

---

## 9. Comparison: Autoencoder vs VAE

### Key Differences

| Aspect | Autoencoder | Variational Autoencoder |
|--------|-------------|-------------------------|
| **Latent Space** | Deterministic | Probabilistic |
| **Encoding** | z = f(x) | z ~ N(μ(x), σ²(x)) |
| **Loss Function** | Reconstruction only | Reconstruction + KL |
| **Generation** | Cannot generate | Can generate new samples |
| **Latent Structure** | Arbitrary | Regularized (Gaussian) |
| **Interpolation** | May have gaps | Smooth interpolation |

### When to Use Each

**Use Autoencoders When:**
- Dimensionality reduction for downstream tasks
- Denoising corrupted data
- Feature learning for classification
- Compression for storage

**Use VAEs When:**
- Generating new samples
- Exploring data manifolds
- Semi-supervised learning
- Controllable generation

---

## 10. Practical Considerations and Extensions

### 10.1 Hyperparameter Tuning

```python
# Key hyperparameters to experiment with:
latent_dim = 10                    # Higher for more complex data
reconstruction_loss_factor = 100   # Balance reconstruction vs regularization
learning_rate = 0.001             # Adam optimizer learning rate
batch_size = 128                  # Memory vs gradient quality trade-off
```

### 10.2 Architecture Improvements

**β-VAE (Beta-VAE):**
```python
# Adjustable KL weight for disentanglement
beta = 4.0
total_loss = reconstruction_loss + beta * kl_loss
```

**Convolutional VAE Improvements:**
```python
# Batch normalization for training stability
from tensorflow.keras.layers import BatchNormalization

encoder_hidden = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
encoder_hidden = BatchNormalization()(encoder_hidden)
```

### 10.3 Evaluation Metrics

```python
# Reconstruction quality
def calculate_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

# Latent space quality (if labels available)
def calculate_latent_separation(latent_codes, labels):
    # Measure how well different classes separate in latent space
    pass

# Generation quality (requires human evaluation or advanced metrics)
def calculate_fid_score(real_images, generated_images):
    # Fréchet Inception Distance
    pass
```

---

## 11. Real-World Applications

### Fashion and Design
- **Style transfer**: Apply style of one garment to another
- **Design exploration**: Generate variations of existing designs
- **Trend analysis**: Discover patterns in fashion data

### Medical Imaging
- **Anomaly detection**: Identify unusual patterns in medical scans
- **Data augmentation**: Generate synthetic training data
- **Compression**: Efficient storage of medical images

### Art and Creativity
- **Artistic generation**: Create new artworks in learned styles
- **Interactive tools**: Allow users to explore creative spaces
- **Style interpolation**: Blend different artistic styles

---

## Summary

This notebook demonstrates the power of Variational Autoencoders for generative modeling:

### Key Achievements
1. **Probabilistic latent space**: Learned structured representation enabling generation
2. **Reparameterization trick**: Enabled backpropagation through stochastic layers
3. **Balanced loss function**: Combined reconstruction quality with latent regularity
4. **Generation capability**: Created new Fashion-MNIST items not in training set
5. **Latent exploration**: Visualized and navigated the learned representation space

### Technical Insights
- **VAE loss balances** reconstruction fidelity with latent space structure
- **2D latent space** enables intuitive visualization but limits model capacity
- **Continuous latent space** allows smooth interpolation between data points
- **Regularization through KL divergence** ensures well-behaved latent distributions

The VAE successfully learns to generate new fashion items while maintaining a structured, explorable latent space, demonstrating the fundamental principles of modern generative modeling.