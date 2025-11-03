# Assignment - AutoEncoders (Solution) - Coding Guide

## Overview
This notebook demonstrates **supervised anomaly detection using AutoEncoders** on the KDD Cup 1999 network intrusion dataset. Instead of using reconstruction error for anomaly detection, it leverages the compressed features from the encoder for supervised classification.

## Key Learning Objectives
- Train AutoEncoders on relational (tabular) data
- Use compressed representations for downstream classification tasks
- Compare performance between original features vs encoded features
- Implement supervised anomaly detection with AutoEncoders

---

## 1. Library Imports and Setup

```python
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Model, models
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive
from sklearn.model_test_split import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import seaborn as sns
```

### Why These Libraries?
- **pandas**: Data manipulation and analysis for tabular data
- **numpy**: Numerical operations and array handling
- **MinMaxScaler**: Normalize continuous features to [0,1] range for neural networks
- **tensorflow.keras**: Deep learning framework for building AutoEncoders
- **OneHotEncoder**: Convert categorical features to numerical format
- **Adam optimizer**: Adaptive learning rate optimizer for neural network training
- **seaborn/matplotlib**: Data visualization for confusion matrices and plots

---

## 2. Data Loading and Preprocessing

### 2.1 Loading KDD Cup 1999 Dataset

```python
# Load the dataset (10% subset for computational efficiency)
data = pd.read_csv('/content/gdrive/MyDrive/kdd-data/kddcup.data_10_percent/kddcup.data_10_percent', header=None)

# Define column names (dataset has no headers)
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                # ... (41 total features)
                "label"]

data.columns = column_names
```

**Key Points:**
- **KDD Cup 1999**: Network intrusion detection dataset with ~5M samples
- **10% subset**: Used for computational efficiency in Colab environment
- **41 features**: Mix of continuous and categorical network traffic features
- **No headers**: Must manually assign column names

### 2.2 Label Processing

```python
# Convert multi-class labels to binary (normal vs anomaly)
data["label"] = data["label"].apply(lambda x: 0 if x == "normal." else 1)
```

**Why Binary Classification?**
- Original dataset has multiple attack types (dos, probe, r2l, u2r)
- Simplified to binary: 0 = normal traffic, 1 = any type of attack
- Makes the problem more manageable for demonstration

### 2.3 Train-Test Split

```python
# Separate features from labels
X = data.drop("label", axis=1)
y = data["label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Parameters Explained:**
- **test_size=0.2**: 80% training, 20% testing
- **random_state=42**: Ensures reproducible splits
- **stratify not used**: Could be added to maintain class balance

---

## 3. Feature Engineering

### 3.1 One-Hot Encoding for Categorical Features

```python
categorical_features = ["protocol_type", "service", "flag"]

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Fit on training data, transform both train and test
one_hot_data_train = encoder.fit_transform(X_train[categorical_features])
one_hot_data_test = encoder.transform(X_test[categorical_features])

# Create DataFrames and concatenate with original data
one_hot_df_train = pd.DataFrame(one_hot_data_train, 
                               columns=encoder.get_feature_names_out(categorical_features))
```

**Key Parameters:**
- **sparse=False**: Return dense arrays instead of sparse matrices
- **handle_unknown='ignore'**: Ignore unknown categories in test set
- **fit_transform vs transform**: Fit encoder on training data only to prevent data leakage

**Why One-Hot Encoding?**
- Neural networks require numerical inputs
- Categorical features like "tcp", "http", "SF" need conversion
- Creates binary columns for each category

### 3.2 Feature Scaling

```python
continuous_features = [x for x in column_names if x not in categorical_features and x != 'label']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale continuous features to [0,1] range
X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test[continuous_features] = scaler.transform(X_test[continuous_features])
```

**Why MinMaxScaler?**
- Neural networks perform better with normalized inputs
- Prevents features with large values from dominating
- [0,1] range works well with sigmoid activation functions
- **Important**: Fit scaler only on training data to prevent data leakage

---

## 4. AutoEncoder Architecture

### 4.1 Encoder Design

```python
input_dim = X_train.shape[1]  # Number of input features
encoding_dim = 4              # Compressed representation size

# Input layer
input_data = layers.Input(shape=(input_dim,))

# Encoder layers (compression)
encoded = layers.Dense(64, activation='relu')(input_data)
encoded = layers.Dense(32, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# Create encoder model
encoder = Model(input_data, encoded)
```

**Architecture Explanation:**
- **Input dimension**: Varies based on one-hot encoded features (~120 features)
- **Compression path**: input_dim → 64 → 32 → 4
- **Activation**: ReLU for non-linearity and avoiding vanishing gradients
- **Bottleneck**: 4-dimensional compressed representation

### 4.2 Decoder Design

```python
# Decoder input (compressed features)
encoded_input = layers.Input(shape=(encoding_dim,))

# Decoder layers (reconstruction)
decoded = layers.Dense(32, activation='relu')(encoded_input)
decoded = layers.Dense(64, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

# Create decoder model
decoder = Model(encoded_input, decoded)
```

**Architecture Explanation:**
- **Expansion path**: 4 → 32 → 64 → input_dim
- **Symmetric design**: Mirror of encoder architecture
- **Output activation**: Sigmoid to match [0,1] scaled input range
- **Reconstruction goal**: Recreate original input from compressed representation

### 4.3 Complete AutoEncoder

```python
# Combine encoder and decoder
autoencoder_encoded = encoder(input_data)
autoencoder_decoded = decoder(autoencoder_encoded)
autoencoder = Model(input_data, autoencoder_decoded)

# Compile with MSE loss
autoencoder.compile(optimizer=Adam(learning_rate=1e-5), loss='mse')
```

**Key Design Decisions:**
- **Loss function**: MSE (Mean Squared Error) for reconstruction
- **Optimizer**: Adam with low learning rate (1e-5) for stable training
- **End-to-end training**: Input → Encoder → Decoder → Reconstructed output

---

## 5. Training Process

### 5.1 AutoEncoder Training

```python
# Train autoencoder to reconstruct input
history = autoencoder.fit(X_train, X_train,  # Input = Target for reconstruction
                         epochs=5, 
                         batch_size=256, 
                         validation_data=(X_test, X_test))
```

**Training Parameters:**
- **Target = Input**: AutoEncoder learns to reconstruct its input
- **epochs=5**: Limited epochs for demonstration (usually need more)
- **batch_size=256**: Balance between memory usage and gradient stability
- **Validation data**: Monitor overfitting during training

### 5.2 Training Visualization

```python
# Plot training curves
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
```

**What to Look For:**
- **Decreasing loss**: Model is learning to reconstruct
- **Gap between train/val**: Indicates overfitting if too large
- **Convergence**: Loss should stabilize after sufficient epochs

---

## 6. Supervised Classification

### 6.1 Baseline Classifier (Original Features)

```python
def get_classifier(input_dimension, lr):
    classifier = models.Sequential()
    classifier.add(layers.Dense(64, activation='relu', input_dim=input_dimension))
    classifier.add(layers.Dense(32, activation='relu'))
    classifier.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    classifier.compile(optimizer=Adam(learning_rate=lr), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
    return classifier

# Train on original features
classifier = get_classifier(input_dim, 1e-5)
classifier.fit(X_train, y_train, epochs=5, batch_size=256, 
               validation_data=(X_test, y_test))
```

**Classifier Architecture:**
- **Hidden layers**: 64 → 32 neurons with ReLU activation
- **Output layer**: Single neuron with sigmoid for binary probability
- **Loss function**: Binary crossentropy for binary classification
- **Input**: All original features (after preprocessing)

### 6.2 AutoEncoder-Based Classifier

```python
# Extract compressed features using trained encoder
encoded_train_features = encoder.predict(X_train)
encoded_test_features = encoder.predict(X_test)

# Train classifier on compressed features
classifier = get_classifier(encoding_dim, 3e-5)  # Note: higher learning rate
classifier.fit(encoded_train_features, y_train, epochs=5, batch_size=256,
               validation_data=(encoded_test_features, y_test))
```

**Key Differences:**
- **Input dimension**: Only 4 compressed features instead of ~120 original features
- **Learning rate**: Slightly higher (3e-5) due to smaller input space
- **Feature extraction**: Uses pre-trained encoder to compress features
- **Hypothesis**: Compressed features capture essential information for classification

---

## 7. Model Evaluation

### 7.1 Performance Metrics

```python
# Make predictions
y_pred = classifier.predict(encoded_test_features)
y_pred = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to binary

# Evaluate performance
print(classification_report(y_test, y_pred))
```

**Evaluation Metrics:**
- **Precision**: TP / (TP + FP) - How many predicted anomalies are actually anomalies
- **Recall**: TP / (TP + FN) - How many actual anomalies were detected
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correct predictions

### 7.2 Confusion Matrix Visualization

```python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Normal", "Anomaly"], 
            yticklabels=["Normal", "Anomaly"])
```

**Confusion Matrix Interpretation:**
- **True Negatives (TN)**: Correctly identified normal traffic
- **False Positives (FP)**: Normal traffic incorrectly flagged as anomaly
- **False Negatives (FN)**: Missed anomalies (most critical error)
- **True Positives (TP)**: Correctly detected anomalies

---

## 8. Key Insights and Comparisons

### Performance Comparison
The notebook compares two approaches:

1. **Direct Classification**: Using all original features (~120 dimensions)
2. **AutoEncoder + Classification**: Using compressed features (4 dimensions)

### Expected Results
- **Dimensionality Reduction**: 4 features vs ~120 features (30x compression)
- **Performance**: Compressed features often achieve comparable accuracy
- **Efficiency**: Faster training and inference with fewer features
- **Interpretability**: Compressed features may capture essential patterns

### Why This Works
- **Feature Learning**: AutoEncoder learns meaningful representations
- **Noise Reduction**: Compression removes irrelevant variations
- **Regularization**: Bottleneck prevents overfitting
- **Transfer Learning**: Pre-trained encoder provides good features

---

## 9. Advanced Concepts

### 9.1 Hyperparameter Considerations

```python
# Key hyperparameters to tune:
encoding_dim = 4        # Size of compressed representation
learning_rate = 1e-5    # Training speed vs stability
batch_size = 256        # Memory vs gradient quality
epochs = 5              # Training duration
```

### 9.2 Architecture Variations

**Deeper Networks:**
```python
# More layers for complex patterns
encoded = layers.Dense(128, activation='relu')(input_data)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
```

**Regularization:**
```python
# Add dropout for regularization
encoded = layers.Dense(64, activation='relu')(input_data)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)
```

### 9.3 Alternative Approaches

**Variational AutoEncoder (VAE):**
- Probabilistic latent space
- Better for generation tasks
- More complex training procedure

**Denoising AutoEncoder:**
- Add noise to input during training
- Learn robust representations
- Better generalization

---

## 10. Practical Applications

### Network Security
- **Intrusion Detection**: Identify malicious network traffic
- **Anomaly Detection**: Find unusual patterns in network behavior
- **Feature Engineering**: Compress high-dimensional network features

### Other Domains
- **Fraud Detection**: Credit card transaction anomalies
- **Manufacturing**: Equipment failure prediction
- **Healthcare**: Medical image analysis and diagnosis

---

## 11. Common Pitfalls and Solutions

### Data Leakage
**Problem**: Fitting preprocessors on entire dataset
**Solution**: Fit only on training data, transform test data

### Overfitting
**Problem**: Model memorizes training data
**Solution**: Use validation data, early stopping, regularization

### Poor Reconstruction
**Problem**: High reconstruction error
**Solution**: Adjust architecture, learning rate, or training epochs

### Class Imbalance
**Problem**: Unequal distribution of normal vs anomaly samples
**Solution**: Use stratified sampling, class weights, or resampling techniques

---

## Summary

This notebook demonstrates a powerful approach to anomaly detection by combining:
1. **Unsupervised feature learning** through AutoEncoders
2. **Supervised classification** on compressed representations
3. **Practical preprocessing** for real-world tabular data

The key insight is that AutoEncoders can learn meaningful compressed representations that preserve classification-relevant information while reducing dimensionality and potentially improving generalization.