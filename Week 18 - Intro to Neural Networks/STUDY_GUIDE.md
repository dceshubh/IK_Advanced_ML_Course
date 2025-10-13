# Week 18: Introduction to Neural Networks - Study Guide

## 📚 Overview
This week introduces the fundamentals of neural networks, the building blocks of deep learning. You'll learn how neural networks work from the ground up, including forward propagation, backpropagation, and implementation using both scratch code and modern frameworks (TensorFlow/Keras and PyTorch).

## 🎯 Learning Objectives
By the end of this week, you should be able to:
- Understand the biological inspiration behind neural networks
- Explain the architecture of feedforward neural networks
- Implement forward propagation and backpropagation from scratch
- Use activation functions and understand their purposes
- Build neural networks using TensorFlow/Keras and PyTorch
- Train and evaluate neural network models
- Understand key concepts: weights, biases, gradients, and loss functions

---

## 📖 Core Concepts

### 1. What Are Neural Networks?

#### 1.1 Biological Inspiration
**The Human Neuron:**
- Receives signals through dendrites
- Processes signals in the cell body
- Sends output through axon to other neurons
- Connections (synapses) have varying strengths

**Artificial Neuron (Perceptron):**
- Receives inputs (x₁, x₂, ..., xₙ)
- Multiplies by weights (w₁, w₂, ..., wₙ)
- Adds bias (b)
- Applies activation function
- Produces output

**Mathematical Representation:**
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
output = activation(z)
```

#### 1.2 Why Neural Networks?
**Advantages:**
- Learn complex non-linear patterns
- Automatic feature extraction
- Universal function approximators
- Scale with data and compute
- State-of-the-art in many domains

**Applications:**
- Image recognition (computer vision)
- Natural language processing
- Speech recognition
- Game playing (AlphaGo)
- Autonomous vehicles
- Medical diagnosis



### 2. Neural Network Architecture

#### 2.1 Layers
**Input Layer:**
- First layer, receives raw data
- Number of neurons = number of features
- No computation, just passes data forward

**Hidden Layer(s):**
- Between input and output layers
- Perform computations and feature extraction
- Can have multiple hidden layers (deep networks)
- Number of neurons is a hyperparameter

**Output Layer:**
- Final layer, produces predictions
- Number of neurons depends on task:
  - Binary classification: 1 neuron
  - Multi-class classification: n neurons (n classes)
  - Regression: 1 or more neurons

**Example Architecture:**
```
Input Layer (4 neurons) → Hidden Layer (10 neurons) → Output Layer (3 neurons)
```
For Iris dataset: 4 features → 10 hidden → 3 classes

#### 2.2 Connections
**Fully Connected (Dense) Layer:**
- Every neuron in layer i connects to every neuron in layer i+1
- Most common type of layer
- Parameters: weights and biases

**Weight Matrix:**
- For layer with m inputs and n outputs
- Weight matrix W has shape (m, n)
- Each element wᵢⱼ represents connection strength

**Bias Vector:**
- One bias per neuron in the layer
- Shape: (1, n) for n neurons
- Shifts the activation function

#### 2.3 Network Depth
**Shallow Network:**
- 1-2 hidden layers
- Good for simple problems
- Faster to train

**Deep Network:**
- 3+ hidden layers
- Can learn hierarchical features
- More powerful but harder to train
- Requires more data

---

### 3. Forward Propagation

#### 3.1 The Process
**Step-by-Step:**
1. Start with input data X
2. For each layer:
   - Compute weighted sum: z = Wx + b
   - Apply activation function: a = σ(z)
   - Use output as input to next layer
3. Final layer produces prediction

**Mathematical Notation:**
```
Layer 1:
z₁ = W₁X + b₁
a₁ = σ(z₁)

Layer 2:
z₂ = W₂a₁ + b₂
a₂ = σ(z₂)

Output = a₂
```

#### 3.2 Matrix Operations
**Why Matrices?**
- Efficient computation
- Vectorization (parallel processing)
- GPU acceleration

**Example:**
```python
# Input: (batch_size, input_dim)
# Weights: (input_dim, hidden_dim)
# Output: (batch_size, hidden_dim)

z = np.dot(X, W) + b  # Matrix multiplication
a = sigmoid(z)         # Element-wise activation
```

**Dimensions:**
- X: (m, n) - m samples, n features
- W: (n, h) - n inputs, h hidden neurons
- b: (1, h) - h biases
- z: (m, h) - m samples, h outputs

---

### 4. Activation Functions

#### 4.1 Why Activation Functions?
**Without Activation:**
- Network becomes linear: f(x) = Wx + b
- Multiple layers collapse to single layer
- Cannot learn complex patterns

**With Activation:**
- Introduces non-linearity
- Enables learning complex functions
- Each layer adds representational power

#### 4.2 Common Activation Functions

**Sigmoid:**
```
σ(z) = 1 / (1 + e^(-z))
```
- Output range: (0, 1)
- Smooth, differentiable
- **Pros:** Good for probabilities, smooth gradient
- **Cons:** Vanishing gradient problem, not zero-centered
- **Use:** Output layer for binary classification

**Tanh (Hyperbolic Tangent):**
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```
- Output range: (-1, 1)
- Zero-centered (better than sigmoid)
- **Pros:** Stronger gradients than sigmoid
- **Cons:** Still suffers from vanishing gradient
- **Use:** Hidden layers in shallow networks

**ReLU (Rectified Linear Unit):**
```
ReLU(z) = max(0, z)
```
- Output range: [0, ∞)
- Most popular activation
- **Pros:** Fast computation, no vanishing gradient, sparse activation
- **Cons:** Dying ReLU problem (neurons can die)
- **Use:** Hidden layers in deep networks

**Leaky ReLU:**
```
LeakyReLU(z) = max(0.01z, z)
```
- Fixes dying ReLU problem
- Small negative slope for z < 0
- **Pros:** Prevents dead neurons
- **Use:** Alternative to ReLU

**Softmax:**
```
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
```
- Output: probability distribution (sums to 1)
- **Use:** Output layer for multi-class classification

**Comparison Table:**
| Activation | Range | Pros | Cons | Best Use |
|------------|-------|------|------|----------|
| Sigmoid | (0, 1) | Smooth, probabilistic | Vanishing gradient | Binary output |
| Tanh | (-1, 1) | Zero-centered | Vanishing gradient | Hidden layers (shallow) |
| ReLU | [0, ∞) | Fast, no vanishing | Dying neurons | Hidden layers (deep) |
| Leaky ReLU | (-∞, ∞) | No dying neurons | Extra hyperparameter | Hidden layers |
| Softmax | (0, 1) | Probability distribution | Only for output | Multi-class output |

---

### 5. Loss Functions

#### 5.1 What is Loss?
**Definition:** Measures how wrong the model's predictions are
- Lower loss = better predictions
- Training goal: minimize loss
- Different tasks need different loss functions

#### 5.2 Common Loss Functions

**Mean Squared Error (MSE):**
```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```
- **Use:** Regression problems
- Penalizes large errors heavily
- Always positive

**Binary Cross-Entropy:**
```
BCE = -(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```
- **Use:** Binary classification
- Measures difference between two probability distributions
- Works with sigmoid output

**Categorical Cross-Entropy:**
```
CCE = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)
```
- **Use:** Multi-class classification
- Works with softmax output
- One-hot encoded labels

**Sparse Categorical Cross-Entropy:**
- Same as categorical cross-entropy
- **Use:** When labels are integers (not one-hot)
- More memory efficient

---

### 6. Backpropagation

#### 6.1 The Concept
**Goal:** Update weights to minimize loss
**Method:** Gradient descent using chain rule

**Intuition:**
1. Calculate loss (how wrong we are)
2. Calculate gradient (which direction to move)
3. Update weights (move in that direction)
4. Repeat until loss is minimized

#### 6.2 The Algorithm

**Step 1: Forward Pass**
- Compute predictions
- Calculate loss

**Step 2: Backward Pass**
- Start from output layer
- Calculate gradient of loss w.r.t. output
- Propagate gradient backward through layers
- Use chain rule at each layer

**Step 3: Update Weights**
```
W = W - learning_rate × gradient
b = b - learning_rate × gradient
```

#### 6.3 Mathematical Details

**Output Layer Gradient:**
```
δ_output = (ŷ - y) × σ'(z_output)
```
- ŷ: prediction
- y: true label
- σ': derivative of activation

**Hidden Layer Gradient:**
```
δ_hidden = (δ_next × W_next^T) × σ'(z_hidden)
```
- δ_next: gradient from next layer
- W_next: weights of next layer
- Chain rule in action!

**Weight Update:**
```
ΔW = learning_rate × (a_prev^T × δ)
Δb = learning_rate × Σ δ
```
- a_prev: activations from previous layer
- δ: gradient at current layer

#### 6.4 Gradient Descent Variants

**Batch Gradient Descent:**
- Use entire dataset for each update
- Slow but stable
- Guaranteed convergence (convex problems)

**Stochastic Gradient Descent (SGD):**
- Use one sample for each update
- Fast but noisy
- Can escape local minima

**Mini-Batch Gradient Descent:**
- Use small batch (e.g., 32, 64, 128 samples)
- Best of both worlds
- Most commonly used

---

### 7. Training Neural Networks

#### 7.1 Hyperparameters

**Learning Rate:**
- Controls step size in gradient descent
- Too high: overshooting, divergence
- Too low: slow convergence
- Typical values: 0.001, 0.01, 0.1

**Number of Epochs:**
- One epoch = one pass through entire dataset
- Too few: underfitting
- Too many: overfitting
- Use early stopping

**Batch Size:**
- Number of samples per gradient update
- Smaller: more updates, noisier
- Larger: fewer updates, more stable
- Typical values: 32, 64, 128, 256

**Number of Hidden Layers:**
- More layers = more capacity
- But harder to train
- Start simple, add complexity

**Number of Neurons per Layer:**
- More neurons = more capacity
- But more parameters to train
- Common: powers of 2 (64, 128, 256)

#### 7.2 Weight Initialization

**Why It Matters:**
- Bad initialization → slow training or failure
- All zeros → all neurons learn same thing
- Too large → exploding gradients
- Too small → vanishing gradients

**Common Methods:**

**Random Normal:**
```python
W = np.random.randn(n_in, n_out) * 0.01
```
- Small random values
- Simple but not optimal

**Xavier/Glorot Initialization:**
```python
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)
```
- Good for sigmoid/tanh
- Maintains variance across layers

**He Initialization:**
```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```
- Good for ReLU
- Accounts for ReLU's zero region

#### 7.3 Regularization

**Why Regularization:**
- Prevents overfitting
- Improves generalization
- Reduces model complexity

**L2 Regularization (Weight Decay):**
```
Loss = Original_Loss + λ × Σ(W²)
```
- Penalizes large weights
- Encourages smaller, distributed weights

**L1 Regularization:**
```
Loss = Original_Loss + λ × Σ|W|
```
- Encourages sparsity
- Some weights become exactly zero

**Dropout:**
- Randomly "drop" neurons during training
- Prevents co-adaptation
- Typical rate: 0.2-0.5

**Early Stopping:**
- Stop training when validation loss stops improving
- Prevents overfitting
- Simple and effective

---

### 8. Implementation Frameworks

#### 8.1 From Scratch (NumPy)

**Pros:**
- Deep understanding of internals
- Full control
- Educational value

**Cons:**
- Time-consuming
- Error-prone
- No GPU acceleration
- Not production-ready

**When to Use:**
- Learning fundamentals
- Understanding algorithms
- Research prototypes

#### 8.2 TensorFlow/Keras

**Keras API:**
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100, batch_size=32)
```

**Pros:**
- High-level, user-friendly API
- Fast prototyping
- Production-ready
- Excellent documentation
- Large community

**Cons:**
- Less flexibility than PyTorch
- Harder to debug
- Static computation graph (TF 1.x)

**When to Use:**
- Quick prototyping
- Standard architectures
- Production deployment
- Beginners

#### 8.3 PyTorch

**PyTorch API:**
```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
```

**Pros:**
- Dynamic computation graph
- Pythonic, intuitive
- Excellent for research
- Easy debugging
- Growing community

**Cons:**
- More verbose than Keras
- Steeper learning curve
- Less production tooling (improving)

**When to Use:**
- Research projects
- Custom architectures
- Need flexibility
- Dynamic models (RNNs, etc.)

**Framework Comparison:**
| Aspect | NumPy | Keras | PyTorch |
|--------|-------|-------|---------|
| Ease of Use | Hard | Easy | Medium |
| Flexibility | Full | Limited | High |
| Speed | Slow | Fast | Fast |
| GPU Support | No | Yes | Yes |
| Debugging | Easy | Hard | Easy |
| Production | No | Yes | Growing |
| Best For | Learning | Prototyping | Research |

---

### 9. Common Problems and Solutions

#### 9.1 Vanishing Gradients
**Problem:** Gradients become very small in deep networks
**Symptoms:** Early layers don't learn, training stalls
**Solutions:**
- Use ReLU instead of sigmoid/tanh
- Use batch normalization
- Use residual connections (ResNet)
- Better weight initialization

#### 9.2 Exploding Gradients
**Problem:** Gradients become very large
**Symptoms:** NaN values, unstable training
**Solutions:**
- Gradient clipping
- Lower learning rate
- Better weight initialization
- Batch normalization

#### 9.3 Overfitting
**Problem:** Model memorizes training data
**Symptoms:** High training accuracy, low test accuracy
**Solutions:**
- More training data
- Regularization (L1, L2, dropout)
- Simpler model (fewer layers/neurons)
- Early stopping
- Data augmentation

#### 9.4 Underfitting
**Problem:** Model too simple to capture patterns
**Symptoms:** Low training and test accuracy
**Solutions:**
- More complex model (more layers/neurons)
- Train longer
- Better features
- Reduce regularization

#### 9.5 Slow Training
**Problem:** Training takes too long
**Solutions:**
- Use GPU
- Larger batch size
- Better optimizer (Adam instead of SGD)
- Learning rate scheduling
- Batch normalization

---

## 💡 Practical Tips

### Model Building
1. **Start simple** - single hidden layer, few neurons
2. **Gradually increase complexity** - add layers/neurons as needed
3. **Use appropriate activation** - ReLU for hidden, softmax/sigmoid for output
4. **Initialize weights properly** - He for ReLU, Xavier for sigmoid/tanh
5. **Monitor training** - plot loss curves, check for over/underfitting

### Training
1. **Split data properly** - train/validation/test sets
2. **Normalize inputs** - zero mean, unit variance
3. **Use mini-batches** - typically 32-128 samples
4. **Start with higher learning rate** - reduce if unstable
5. **Use early stopping** - prevent overfitting

### Debugging
1. **Check data** - correct shapes, no NaN/Inf
2. **Verify forward pass** - output shapes correct
3. **Test on small dataset** - should overfit easily
4. **Monitor gradients** - not too large or small
5. **Visualize predictions** - sanity check

### Optimization
1. **Use Adam optimizer** - good default choice
2. **Try learning rate scheduling** - reduce over time
3. **Use batch normalization** - faster, more stable training
4. **Experiment with architecture** - depth vs width
5. **Ensemble models** - combine multiple models

---

## 🎓 Interview Questions

### Beginner Level

**Q1: What is a neural network?**
A: A neural network is a machine learning model inspired by the human brain. It consists of layers of interconnected neurons that process information. Each neuron receives inputs, applies weights and a bias, then passes the result through an activation function to produce an output.

**Q2: What is the purpose of activation functions?**
A: Activation functions introduce non-linearity into the network. Without them, multiple layers would collapse into a single linear transformation, limiting the network's ability to learn complex patterns. They enable neural networks to approximate any function.

**Q3: What is forward propagation?**
A: Forward propagation is the process of passing input data through the network layer by layer to generate predictions. At each layer, we compute weighted sums (z = Wx + b) and apply activation functions (a = σ(z)).

**Q4: What is backpropagation?**
A: Backpropagation is the algorithm for training neural networks. It calculates gradients of the loss function with respect to each weight by propagating errors backward through the network using the chain rule. These gradients are then used to update weights via gradient descent.

**Q5: What's the difference between batch, stochastic, and mini-batch gradient descent?**
A: Batch GD uses the entire dataset for each update (slow but stable). Stochastic GD uses one sample (fast but noisy). Mini-batch GD uses small batches (best balance, most commonly used).

### Intermediate Level

**Q6: Explain the vanishing gradient problem.**
A: In deep networks with sigmoid/tanh activations, gradients can become extremely small as they're backpropagated through layers. This causes early layers to learn very slowly or not at all. Solutions include using ReLU activations, batch normalization, and better weight initialization.

**Q7: Why is ReLU preferred over sigmoid in hidden layers?**
A: ReLU has several advantages: (1) No vanishing gradient for positive values, (2) Computationally efficient (simple max operation), (3) Sparse activation (some neurons output zero), (4) Faster convergence. However, it can suffer from dying ReLU problem.

**Q8: What is the purpose of the bias term?**
A: The bias term allows the activation function to be shifted left or right, providing more flexibility in fitting the data. Without bias, the activation function is forced to pass through the origin, limiting the model's expressiveness.

**Q9: How do you choose the number of hidden layers and neurons?**
A: Start simple (1-2 hidden layers) and increase complexity if needed. More layers help with hierarchical features but are harder to train. Number of neurons depends on problem complexity - start with powers of 2 (64, 128, 256). Use validation performance to guide decisions.

**Q10: What is one-hot encoding and why is it used for neural network outputs?**
A: One-hot encoding converts categorical labels into binary vectors (e.g., class 2 of 3 classes → [0, 1, 0]). It's used because: (1) Prevents ordinal assumptions, (2) Each class gets its own output neuron, (3) Works well with softmax and cross-entropy loss.

### Advanced Level

**Q11: Explain the mathematical intuition behind backpropagation.**
A: Backpropagation applies the chain rule of calculus to compute gradients. For a loss L and weight w in layer l: ∂L/∂w = (∂L/∂a) × (∂a/∂z) × (∂z/∂w), where a is activation and z is weighted sum. We compute these partial derivatives layer by layer, moving backward from output to input.

**Q12: What is the dying ReLU problem and how do you fix it?**
A: Dying ReLU occurs when neurons always output zero (when z < 0), causing zero gradients and no learning. This happens when large negative weights push inputs into the negative region. Solutions: (1) Leaky ReLU (small negative slope), (2) Parametric ReLU (learnable slope), (3) ELU (exponential linear unit), (4) Better weight initialization, (5) Lower learning rate.

**Q13: Compare Xavier and He initialization.**
A: Xavier (Glorot) initialization: W ~ N(0, 1/n_in), designed for sigmoid/tanh activations. Maintains variance across layers assuming linear activations. He initialization: W ~ N(0, 2/n_in), designed for ReLU. The factor of 2 accounts for ReLU zeroing out half the neurons. Use He for ReLU, Xavier for sigmoid/tanh.

**Q14: How does batch normalization help training?**
A: Batch normalization normalizes layer inputs to have zero mean and unit variance. Benefits: (1) Reduces internal covariate shift, (2) Allows higher learning rates, (3) Reduces sensitivity to initialization, (4) Acts as regularization, (5) Faster convergence. It adds learnable scale and shift parameters to maintain representational power.

**Q15: Explain the trade-off between model capacity and generalization.**
A: Model capacity refers to the complexity of functions a model can represent. Higher capacity (more layers/neurons) can fit training data better but risks overfitting. Lower capacity may underfit. The goal is finding the sweet spot where the model captures true patterns without memorizing noise. Use regularization, cross-validation, and appropriate architecture to balance this trade-off.

---

## 🛠️ Code Templates

### From Scratch Implementation
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y):
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer gradient
        delta = self.activations[-1] - y
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(self.activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            
            if verbose and epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-8))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Usage
nn = NeuralNetwork([4, 10, 3], learning_rate=0.01)
nn.train(X_train, y_train, epochs=1000)
predictions = nn.predict(X_test)
```

### TensorFlow/Keras Implementation
```python
import tensorflow as tf
from tensorflow import keras

# Sequential API (simple)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])

# Functional API (flexible)
inputs = keras.Input(shape=(4,))
x = keras.layers.Dense(10, activation='relu')(inputs)
outputs = keras.layers.Dense(3, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
```

### PyTorch Implementation
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize
model = NeuralNet(4, 10, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert data to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(np.argmax(y_train, axis=1))

# Train
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
```

---

## 📚 Additional Resources

### Documentation
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Neural Networks and Deep Learning (Free Book)](http://neuralnetworksanddeeplearning.com/)

### Courses
- Andrew Ng's Deep Learning Specialization (Coursera)
- Fast.ai Practical Deep Learning
- Stanford CS231n (Computer Vision)

### Papers
- "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
- "ImageNet Classification with Deep CNNs" (Krizhevsky et al., 2012)
- "Batch Normalization" (Ioffe & Szegedy, 2015)

### Tools
- TensorBoard (visualization)
- Weights & Biases (experiment tracking)
- Netron (model visualization)

---

## ✅ Key Takeaways

1. **Neural networks** are universal function approximators inspired by the brain
2. **Forward propagation** computes predictions layer by layer
3. **Backpropagation** calculates gradients using the chain rule
4. **Activation functions** introduce non-linearity (use ReLU for hidden layers)
5. **Loss functions** measure prediction error (cross-entropy for classification)
6. **Gradient descent** updates weights to minimize loss
7. **Proper initialization** is crucial for successful training
8. **Regularization** prevents overfitting (dropout, L2, early stopping)
9. **Frameworks** (Keras, PyTorch) make implementation easier
10. **Start simple** and increase complexity as needed

---

*This study guide covers the fundamentals of neural networks. Refer to the coding guides for hands-on implementation details with NumPy, TensorFlow/Keras, and PyTorch.*
