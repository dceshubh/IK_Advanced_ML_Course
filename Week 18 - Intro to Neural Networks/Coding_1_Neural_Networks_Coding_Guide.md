# Detailed Step-by-Step Guide: Neural Networks Coding Notebook - Week 18

This comprehensive guide explains every step, function, and concept used in the neural networks coding notebook, covering three different implementations: from scratch, TensorFlow, and PyTorch.

## Overview
The notebook demonstrates building neural networks for the Iris dataset classification problem using three different approaches:
1. **From Scratch Implementation** - Understanding the fundamentals
2. **TensorFlow/Keras Implementation** - High-level framework
3. **PyTorch Implementation** - Dynamic computation graphs

---

## Part 1: Data Preparation and Setup

### Step 1: Import Required Libraries
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
```

**Purpose**: Import essential libraries for data manipulation, machine learning utilities, and evaluation metrics.

**Functions Explained**:
- `numpy`: Numerical computing library for array operations
- `sklearn.datasets.load_iris`: Loads the famous Iris flower dataset
- `train_test_split`: Splits data into training and testing sets
- `OneHotEncoder`: Converts categorical labels to binary vectors
- `accuracy_score`: Calculates classification accuracy

### Step 2: Load and Prepare the Dataset
```python
# Load dataset
data = load_iris()
X, y = data.data, data.target
```

**Purpose**: Load the Iris dataset which contains 150 samples of iris flowers with 4 features each (sepal length, sepal width, petal length, petal width) and 3 classes (setosa, versicolor, virginica).

**Data Structure**:
- `X`: Feature matrix (150 × 4) - input features
- `y`: Target vector (150,) - class labels (0, 1, 2)

### Step 3: One-Hot Encode Target Labels
```python
# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))
```

**Purpose**: Convert integer class labels to one-hot encoded vectors for multi-class classification.

**Why One-Hot Encoding?**:
- Neural networks work better with one-hot encoded outputs
- Prevents the model from assuming ordinal relationships between classes
- Each class gets its own output neuron

**Transformation**:
- Original: `[0, 1, 2]` → One-hot: `[[1,0,0], [0,1,0], [0,0,1]]`

### Step 4: Train-Test Split
```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Purpose**: Split the dataset into training (80%) and testing (20%) sets.

**Parameters**:
- `test_size=0.2`: 20% for testing, 80% for training
- `random_state=42`: Ensures reproducible results

---

## Part 2: Neural Network From Scratch

### Step 5: Define the Neural Network Class
```python
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))
        self.learning_rate = learning_rate
```

**Purpose**: Initialize a simple feedforward neural network with one hidden layer.

**Architecture**:
- Input layer: 4 neurons (iris features)
- Hidden layer: 10 neurons (configurable)
- Output layer: 3 neurons (iris classes)

**Weight Initialization**:
- `np.random.randn()`: Random normal distribution (mean=0, std=1)
- `np.zeros()`: Initialize biases to zero
- Good practice to break symmetry and enable learning

### Step 6: Activation Functions
```python
def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(self, z):
    return z * (1 - z)
```

**Purpose**: Define the sigmoid activation function and its derivative.

**Sigmoid Function**:
- Formula: σ(z) = 1/(1 + e^(-z))
- Range: (0, 1)
- Smooth, differentiable
- Good for binary classification and hidden layers

**Derivative**:
- Used in backpropagation
- σ'(z) = σ(z) × (1 - σ(z))
- Efficient to compute from the sigmoid output

### Step 7: Forward Propagation
```python
def forward(self, X):
    self.z1 = np.dot(X, self.W1) + self.b1
    self.a1 = self.sigmoid(self.z1)
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    self.a2 = self.sigmoid(self.z2)
    return self.a2
```

**Purpose**: Compute the forward pass through the network.

**Step-by-Step Process**:
1. `z1 = X·W1 + b1`: Linear transformation (input to hidden)
2. `a1 = σ(z1)`: Apply sigmoid activation
3. `z2 = a1·W2 + b2`: Linear transformation (hidden to output)
4. `a2 = σ(z2)`: Apply sigmoid activation (final output)

**Mathematical Flow**:
- Input → Linear → Activation → Linear → Activation → Output

### Step 8: Backward Propagation
```python
def backward(self, X, y, output):
    error = output - y
    d_output = error * self.sigmoid_derivative(output)
    
    error_hidden = d_output.dot(self.W2.T)
    d_hidden = error_hidden * self.sigmoid_derivative(self.a1)
    
    self.W2 -= self.a1.T.dot(d_output) * self.learning_rate
    self.b2 -= np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
    self.W1 -= X.T.dot(d_hidden) * self.learning_rate
    self.b1 -= np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate
```

**Purpose**: Implement backpropagation to update weights and biases.

**Backpropagation Steps**:
1. **Output Layer Error**: `error = output - y`
2. **Output Layer Gradient**: `d_output = error × σ'(output)`
3. **Hidden Layer Error**: Propagate error backward through weights
4. **Hidden Layer Gradient**: `d_hidden = error_hidden × σ'(a1)`
5. **Update Weights**: Use gradients to update parameters

**Gradient Descent Updates**:
- `W2 -= learning_rate × gradient_W2`
- `b2 -= learning_rate × gradient_b2`
- Similar for W1 and b1

### Step 9: Training Loop
```python
def train(self, X, y, epochs=1000):
    for _ in range(epochs):
        output = self.forward(X)
        self.backward(X, y, output)
```

**Purpose**: Train the network by repeatedly applying forward and backward passes.

**Training Process**:
1. Forward pass: Compute predictions
2. Backward pass: Compute gradients and update weights
3. Repeat for specified number of epochs

### Step 10: Prediction Function
```python
def predict(self, X):
    output = self.forward(X)
    return np.argmax(output, axis=1)
```

**Purpose**: Make predictions on new data.

**Process**:
1. Forward pass to get output probabilities
2. `np.argmax()`: Return the class with highest probability
3. Convert from one-hot back to class indices

### Step 11: Train and Evaluate the From-Scratch Model
```python
# Initialize the neural network
input_dim = X_train.shape[1]  # 4 features
hidden_dim = 10               # 10 hidden neurons
output_dim = y_train.shape[1] # 3 classes
model = SimpleNN(input_dim, hidden_dim, output_dim, learning_rate=0.01)

# Train the model
model.train(X_train, y_train, epochs=10000)

# Make predictions
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy from Model: {accuracy * 100:.2f}%")
```

**Purpose**: Create, train, and evaluate the from-scratch neural network.

**Results**: Achieves 100% accuracy on the test set, demonstrating the effectiveness of the implementation.

---

## Part 3: TensorFlow/Keras Implementation

### Step 12: Import TensorFlow
```python
import tensorflow as tf
from tensorflow.keras import layers, models
```

**Purpose**: Import TensorFlow and Keras for high-level neural network construction.

### Step 13: Define the TensorFlow Model
```python
# Define the model
tf_model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(hidden_dim, activation='sigmoid'),
    layers.Dense(output_dim, activation='softmax')
])
```

**Purpose**: Create a sequential neural network model using Keras.

**Architecture Explanation**:
- `Sequential`: Linear stack of layers
- `Input`: Defines input shape (4 features)
- `Dense`: Fully connected layer
- `activation='sigmoid'`: Hidden layer activation
- `activation='softmax'`: Output layer activation for multi-class classification

**Softmax vs Sigmoid**:
- Softmax: Outputs sum to 1, good for multi-class classification
- Sigmoid: Each output independent, good for binary classification

### Step 14: Compile the Model
```python
# Compile the model
tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Purpose**: Configure the model for training.

**Parameters Explained**:
- `optimizer='adam'`: Adaptive learning rate optimization algorithm
- `loss='categorical_crossentropy'`: Loss function for multi-class classification
- `metrics=['accuracy']`: Track accuracy during training

**Adam Optimizer**:
- Combines benefits of AdaGrad and RMSprop
- Adaptive learning rates for each parameter
- Generally works well out-of-the-box

### Step 15: Set Up Model Checkpointing
```python
# Define a callback for saving the model
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('tf_model.h5.keras', save_best_only=True)
```

**Purpose**: Save the best model during training based on validation performance.

**Benefits**:
- Prevents overfitting by saving the best validation performance
- Allows recovery of best model even if training continues past optimal point

### Step 16: Train the TensorFlow Model
```python
# Train the model
tf_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[checkpoint_callback])
```

**Purpose**: Train the model with validation monitoring.

**Parameters**:
- `epochs=100`: Number of training iterations
- `validation_data`: Monitor performance on test set
- `callbacks`: List of callback functions (checkpointing)

**Training Output**: Shows progress with loss and accuracy for both training and validation sets.

### Step 17: Load and Evaluate Best Model
```python
# Load the best saved model
best_tf_model = tf.keras.models.load_model('tf_model.h5.keras')

# Evaluate the model
test_loss, test_acc = best_tf_model.evaluate(X_test, y_test, verbose=2)
print(f"Accuracy from TensorFlow Model: {test_acc * 100:.2f}%")
```

**Purpose**: Load the best saved model and evaluate its performance.

**Results**: Achieves 96.67% accuracy on the test set.

---

## Part 4: PyTorch Implementation

### Step 18: Import PyTorch Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

**Purpose**: Import PyTorch components for neural network construction and training.

### Step 19: Convert Data to PyTorch Tensors
```python
# Convert data to tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(np.argmax(y_train, axis=1)).long()
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(np.argmax(y_test, axis=1)).long()
```

**Purpose**: Convert NumPy arrays to PyTorch tensors.

**Key Points**:
- `torch.Tensor()`: Creates PyTorch tensor from NumPy array
- `np.argmax()`: Convert one-hot back to class indices
- `.long()`: Convert to long integer type (required for CrossEntropyLoss)

### Step 20: Create DataLoader
```python
# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

**Purpose**: Create a data loader for efficient batch processing.

**Benefits**:
- `TensorDataset`: Wraps tensors into a dataset
- `DataLoader`: Provides batching, shuffling, and parallel loading
- `batch_size=16`: Process 16 samples at a time
- `shuffle=True`: Randomize order each epoch

### Step 21: Define PyTorch Neural Network Class
```python
class PyTorchNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Purpose**: Define a neural network class inheriting from `nn.Module`.

**Architecture**:
- `nn.Linear`: Fully connected layer (equivalent to Dense in Keras)
- `nn.Sigmoid`: Sigmoid activation function
- `forward()`: Defines the forward pass computation

**PyTorch Design Pattern**:
- Inherit from `nn.Module`
- Define layers in `__init__()`
- Define forward pass in `forward()`

### Step 22: Initialize Model, Loss, and Optimizer
```python
# Instantiate the model
pt_model = PyTorchNN(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pt_model.parameters(), lr=0.01)
```

**Purpose**: Set up the model, loss function, and optimizer.

**Components**:
- `pt_model`: Instance of our neural network
- `nn.CrossEntropyLoss()`: Loss function for multi-class classification
- `optim.Adam()`: Adam optimizer with learning rate 0.01

### Step 23: Training Loop with Model Saving
```python
# Define a callback to save the model
best_acc = 0.0
for epoch in range(100):
    pt_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = pt_model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    # Evaluate on the test set
    pt_model.eval()
    with torch.no_grad():
        test_output = pt_model(X_test_tensor)
        _, y_pred_tensor = torch.max(test_output, 1)
        accuracy = (y_pred_tensor == y_test_tensor).sum().item() / len(y_test_tensor)

    # Save the best model
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(pt_model.state_dict(), 'pt_model.pth')

print(f"Accuracy from PyTorch Model: {best_acc * 100:.2f}%")
```

**Purpose**: Train the PyTorch model with evaluation and model saving.

**Training Steps**:
1. **Set Training Mode**: `pt_model.train()`
2. **Batch Processing**: Iterate through data loader
3. **Zero Gradients**: `optimizer.zero_grad()`
4. **Forward Pass**: Compute predictions
5. **Compute Loss**: Calculate error
6. **Backward Pass**: `loss.backward()`
7. **Update Parameters**: `optimizer.step()`

**Evaluation Steps**:
1. **Set Evaluation Mode**: `pt_model.eval()`
2. **Disable Gradients**: `torch.no_grad()`
3. **Make Predictions**: Forward pass on test data
4. **Calculate Accuracy**: Compare predictions with true labels

**Model Saving**:
- Save model state dictionary when accuracy improves
- `torch.save()`: Save model parameters to file

**Results**: Achieves 100% accuracy on the test set.

---

## Key Concepts and Functions Summary

### Mathematical Concepts
1. **Forward Propagation**: Computing outputs from inputs through layers
2. **Backpropagation**: Computing gradients and updating weights
3. **Gradient Descent**: Optimization algorithm for learning
4. **Activation Functions**: Non-linear transformations (sigmoid, softmax)
5. **Loss Functions**: Measuring prediction errors

### Important Functions
1. **Data Preprocessing**:
   - `load_iris()`: Load dataset
   - `train_test_split()`: Split data
   - `OneHotEncoder()`: Encode categorical labels

2. **NumPy Operations**:
   - `np.dot()`: Matrix multiplication
   - `np.argmax()`: Find maximum value indices
   - `np.random.randn()`: Random normal distribution

3. **TensorFlow/Keras**:
   - `Sequential()`: Linear model architecture
   - `Dense()`: Fully connected layer
   - `compile()`: Configure model for training
   - `fit()`: Train the model
   - `ModelCheckpoint()`: Save best model

4. **PyTorch**:
   - `nn.Module`: Base class for neural networks
   - `nn.Linear()`: Fully connected layer
   - `DataLoader()`: Batch data loading
   - `torch.no_grad()`: Disable gradient computation
   - `torch.save()`: Save model parameters

### Best Practices Demonstrated
1. **Data Splitting**: Separate training and testing data
2. **One-Hot Encoding**: Proper categorical encoding
3. **Model Checkpointing**: Save best performing models
4. **Evaluation Mode**: Proper train/eval mode switching
5. **Gradient Management**: Zero gradients and disable when needed

This comprehensive guide covers all the essential concepts and implementations for understanding neural networks from basic principles to modern frameworks.