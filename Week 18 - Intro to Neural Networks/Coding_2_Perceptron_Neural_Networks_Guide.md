# Detailed Step-by-Step Guide: Perceptron and Neural Networks - Week 18 Coding Notebook-2

This comprehensive guide explains every step, function, and concept used in the second neural networks coding notebook, covering Perceptrons, logic gates, and neural network implementations.

## Overview
The notebook demonstrates:
1. **Perceptron Implementation** - Building a perceptron from scratch
2. **Logic Gate Classification** - AND, OR, and XOR operations
3. **Neural Network Models** - Using TensorFlow/Keras for regression
4. **Model Comparison** - Neural networks vs. gradient boosting

---

## Part 1: Perceptron Implementation

### Step 1: Import Required Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
```

**Purpose**: Import essential libraries for numerical computing, visualization, model persistence, and data manipulation.

**Libraries Explained**:
- `numpy`: Numerical operations and array handling
- `matplotlib.pyplot`: Data visualization and plotting
- `joblib`: Model serialization and persistence
- `pandas`: Data manipulation and analysis

### Step 2: Define the Perceptron Class
```python
class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4
    print(f"self.weights: {self.weights}")
    self.eta = eta
    self.epochs = epochs
```

**Purpose**: Initialize a perceptron with learning rate and training epochs.

**Parameters Explained**:
- `eta`: Learning rate (how much weights are adjusted each iteration)
- `epochs`: Number of training iterations
- `weights`: Random initialization with small values (3 weights for 2 inputs + bias)

**Why 3 weights?**:
- 2 weights for input features (x1, x2)
- 1 weight for bias term (constant offset)

### Step 3: Activation Function
```python
def activationFunction(self, inputs, weights): # binary activation function
  z = np.dot(inputs, weights)
  return np.where(z > 0 , 1, 0)
```

**Purpose**: Implement the step activation function for binary classification.

**Mathematical Process**:
1. `z = inputs · weights`: Compute weighted sum (dot product)
2. `np.where(z > 0, 1, 0)`: Apply step function (1 if z > 0, else 0)

**Step Function**:
- Output = 1 if weighted sum > 0
- Output = 0 if weighted sum ≤ 0
- Creates linear decision boundary

### Step 4: Training Method (fit)
```python
def fit(self, X, y):
  self.X = X
  self.y = y
  
  X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))] # concatenation
  print(f"X_with_bias: \n{X_with_bias}")
  
  for epoch in range(self.epochs):
    print(f"for epoch: {epoch}")
    y_hat = self.activationFunction(X_with_bias, self.weights)
    print(f"predicted value: \n{y_hat}")
    error = self.y - y_hat
    print(f"error: \n{error}")
    self.weights = self.weights + self.eta * np.dot(X_with_bias.T, error)
    print(f"updated weights: \n{self.weights}")
    print("#############\n")
```

**Purpose**: Train the perceptron using the perceptron learning algorithm.

**Step-by-Step Process**:
1. **Add Bias**: `np.c_[X, -np.ones()]` concatenates bias column (-1s)
2. **Forward Pass**: Compute predictions using current weights
3. **Error Calculation**: `error = actual - predicted`
4. **Weight Update**: `weights += learning_rate × X^T × error`

**Perceptron Learning Rule**:
- If prediction is correct: no weight change
- If prediction is wrong: adjust weights toward correct answer
- Guaranteed to converge for linearly separable data

### Step 5: Prediction Method
```python
def predict(self, X):
  X_with_bias = np.c_[X, -np.ones((len(self.X), 1))]
  return self.activationFunction(X_with_bias, self.weights)
```

**Purpose**: Make predictions on new data using trained weights.

---

## Part 2: Logic Gate Implementation

### Step 6: AND Gate Implementation
```python
data = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [0,0,0,1]}
AND = pd.DataFrame(data)
```

**Purpose**: Create truth table for AND logic gate.

**AND Gate Truth Table**:
- (0,0) → 0: False AND False = False
- (0,1) → 0: False AND True = False  
- (1,0) → 0: True AND False = False
- (1,1) → 1: True AND True = True

**Linear Separability**: AND gate is linearly separable - can be solved by perceptron.

### Step 7: Train Perceptron on AND Gate
```python
X = AND.drop("y", axis=1)
y = AND['y']
model = Perceptron(eta = 0.5, epochs=10)
model.fit(X,y)
```

**Purpose**: Train perceptron to learn AND gate logic.

**Training Process**:
1. Separate features (X) from target (y)
2. Initialize perceptron with learning rate 0.5
3. Train for 10 epochs
4. Weights adjust to create correct decision boundary

### Step 8: Visualize AND Gate
```python
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = y.values

plt.scatter(X_and[y_and == 1,0],X_and[y_and == 1,1], c = 'b', label = "1")
plt.scatter(X_and[y_and == 0,0],X_and[y_and == 0,1], c = 'r', label = "0")
plt.legend()
plt.show()
```

**Purpose**: Visualize the AND gate data points and decision boundary.

**Visualization Shows**:
- Blue points: Output = 1 (only point (1,1))
- Red points: Output = 0 (points (0,0), (0,1), (1,0))
- Linear boundary can separate these classes

### Step 9: OR Gate Implementation
```python
data = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [0,1,1,1]}
OR = pd.DataFrame(data)
```

**Purpose**: Create truth table for OR logic gate.

**OR Gate Truth Table**:
- (0,0) → 0: False OR False = False
- (0,1) → 1: False OR True = True
- (1,0) → 1: True OR False = True
- (1,1) → 1: True OR True = True

**Linear Separability**: OR gate is also linearly separable.

### Step 10: XOR Gate - The Challenge
```python
data = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [0,1,1,0]}
XOR = pd.DataFrame(data)
```

**Purpose**: Create truth table for XOR (exclusive OR) logic gate.

**XOR Gate Truth Table**:
- (0,0) → 0: False XOR False = False
- (0,1) → 1: False XOR True = True
- (1,0) → 1: True XOR False = True
- (1,1) → 0: True XOR True = False

**Non-Linear Problem**: XOR is NOT linearly separable - single perceptron cannot solve it.

### Step 11: Train Perceptron on XOR (Fails)
```python
model = Perceptron(eta = 0.5, epochs=50)
model.fit(X,y)
```

**Purpose**: Demonstrate that single perceptron cannot learn XOR.

**Why It Fails**:
- XOR requires non-linear decision boundary
- Single perceptron can only create linear boundaries
- No single line can separate XOR classes correctly

### Step 12: Multi-Layer Perceptron (MLP) Solution
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(25,),max_iter=10000)
mlp.fit(X_xor,y_xor)
pred = mlp.predict(X_xor)
```

**Purpose**: Show that MLP can solve XOR problem.

**MLP Architecture**:
- Input layer: 2 neurons (x1, x2)
- Hidden layer: 25 neurons with non-linear activation
- Output layer: 1 neuron for classification

**Why MLP Works**:
- Hidden layer creates non-linear transformations
- Multiple layers can approximate any function
- Can create complex decision boundaries

---

## Part 3: Neural Network for Regression

### Step 13: Load and Prepare Boston Housing Dataset
```python
data=pd.read_csv('/content/sample_data/Boston.csv')
data.head(7)
```

**Purpose**: Load Boston housing dataset for regression task.

**Dataset Features**:
- 13 input features (crime rate, rooms, age, etc.)
- 1 target variable (median home value)
- 506 samples total

### Step 14: Data Exploration
```python
data.isna().sum()  # Check for missing values
sns.pairplot(data)  # Visualize feature relationships
```

**Purpose**: Understand data quality and feature relationships.

**Data Quality Checks**:
- Missing values: None found
- Feature distributions: Various scales and ranges
- Target variable: Continuous values (regression problem)

### Step 15: Data Preprocessing
```python
x = data.iloc[:,:-1]  # Features (all columns except last)
y = data.iloc[:,-1]   # Target (last column)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,shuffle=True,random_state=21)
```

**Purpose**: Separate features from target and split into train/test sets.

**Split Details**:
- 75% training data (379 samples)
- 25% test data (127 samples)
- Random shuffle with fixed seed for reproducibility

### Step 16: Feature Scaling
```python
x_scale_object = MinMaxScaler((0,1))
x_train_scaled = x_scale_object.fit_transform(x_train)
x_test_scaled = x_scale_object.transform(x_test)
```

**Purpose**: Scale features to [0,1] range for better neural network training.

**Why Scaling is Important**:
- Neural networks sensitive to input scale
- Features have different units and ranges
- Scaling prevents dominant features from overwhelming others
- Improves convergence speed and stability

### Step 17: Simple Neural Network Architecture
```python
model = Sequential()
model.add(Input(shape=(12,)))
model.add(Dense(units=12,activation='relu',kernel_regularizer=regularizers.L2(l2=0.01)))
model.add(Dense(units=24,activation='relu',kernel_regularizer=regularizers.L2(l2=0.01)))
model.add(Dense(units='1',activation='linear',kernel_regularizer=regularizers.L2(l2=0.01)))
```

**Purpose**: Define a simple feedforward neural network for regression.

**Architecture Details**:
- Input layer: 12 features
- Hidden layer 1: 12 neurons, ReLU activation
- Hidden layer 2: 24 neurons, ReLU activation  
- Output layer: 1 neuron, linear activation (regression)

**Regularization**: L2 regularization (λ=0.01) prevents overfitting.

### Step 18: Model Compilation and Training
```python
optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss='mse')
history=model.fit(x_train_scaled,y_train,validation_split=0.3,epochs=100,verbose=0)
```

**Purpose**: Configure and train the neural network.

**Training Configuration**:
- Optimizer: Adam (adaptive learning rate)
- Loss function: Mean Squared Error (MSE)
- Validation split: 30% of training data
- Epochs: 100 training iterations

### Step 19: Learning Curve Visualization
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title('Learning Curve')
plt.legend(['Training Loss','Validation Loss'])
plt.show()
```

**Purpose**: Monitor training progress and detect overfitting.

**Learning Curve Analysis**:
- Training loss: Error on training data
- Validation loss: Error on validation data
- Gap between curves indicates overfitting
- Both should decrease over epochs

### Step 20: Model with Dropout Layers
```python
model2 = Sequential()
model2.add(Input(shape=(12,)))
model2.add(Dense(units=12,activation='relu',kernel_regularizer=regularizers.L2(l2=0.01)))
model2.add(Dropout(0.2))
model2.add(Dense(units=24,activation='relu',kernel_regularizer=regularizers.L2(l2=0.01)))
model2.add(Dropout(0.2))
model2.add(Dense(units='1',activation='linear',kernel_regularizer=regularizers.L2(l2=0.01)))
```

**Purpose**: Add dropout layers to reduce overfitting.

**Dropout Mechanism**:
- Randomly sets 20% of neurons to zero during training
- Prevents co-adaptation of neurons
- Improves generalization to unseen data
- Only active during training, not inference

### Step 21: Model with Batch Normalization
```python
model3.add(Dense(units=12,activation='relu',kernel_regularizer=regularizers.L2(l2=0.01)))
model3.add(BatchNormalization(momentum=0.99,epsilon=0.001))
model3.add(Dropout(0.2))
```

**Purpose**: Add batch normalization for training stability.

**Batch Normalization Benefits**:
- Normalizes inputs to each layer
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization technique

**Parameters**:
- `momentum=0.99`: Exponential moving average factor
- `epsilon=0.001`: Small constant for numerical stability

### Step 22: Model Comparison with XGBoost
```python
from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=25, max_depth=3, learning_rate=0.1)
xgb_model.fit(x_train_scaled, y_train)
```

**Purpose**: Compare neural network performance with gradient boosting.

**XGBoost Configuration**:
- `n_estimators=25`: Number of boosting rounds
- `max_depth=3`: Maximum tree depth
- `learning_rate=0.1`: Step size shrinkage

---

## Key Concepts and Functions Summary

### Mathematical Concepts
1. **Perceptron Learning Rule**: Weight updates based on prediction errors
2. **Linear Separability**: Whether classes can be separated by straight line
3. **Activation Functions**: ReLU, linear, step functions
4. **Backpropagation**: Gradient-based weight optimization
5. **Regularization**: L2 penalty and dropout for overfitting prevention

### Important Functions
1. **Data Preprocessing**:
   - `train_test_split()`: Split data into train/test sets
   - `MinMaxScaler()`: Scale features to [0,1] range
   - `pd.DataFrame()`: Create structured data tables

2. **Neural Network Components**:
   - `Sequential()`: Linear stack of layers
   - `Dense()`: Fully connected layer
   - `Dropout()`: Regularization layer
   - `BatchNormalization()`: Normalization layer

3. **Training and Evaluation**:
   - `compile()`: Configure model for training
   - `fit()`: Train the model
   - `predict()`: Make predictions
   - `evaluate()`: Assess model performance

4. **Visualization**:
   - `plt.scatter()`: Create scatter plots
   - `plt.plot()`: Plot learning curves
   - `sns.pairplot()`: Visualize feature relationships

### Key Insights Demonstrated
1. **Perceptron Limitations**: Cannot solve non-linearly separable problems (XOR)
2. **MLP Power**: Multi-layer networks can solve complex problems
3. **Regularization Importance**: Dropout and L2 prevent overfitting
4. **Feature Scaling**: Critical for neural network performance
5. **Model Comparison**: Different algorithms have different strengths

### Logic Gate Analysis
- **AND Gate**: Linearly separable, perceptron succeeds
- **OR Gate**: Linearly separable, perceptron succeeds  
- **XOR Gate**: Non-linearly separable, perceptron fails, MLP succeeds

### Training Concepts
- **Learning Rate**: Controls weight update magnitude
- **Epochs**: Number of complete passes through data
- **Validation Split**: Portion of data for monitoring overfitting
- **Early Stopping**: Prevent overfitting by stopping training early

This comprehensive guide covers the fundamental concepts of perceptrons, their limitations, and how multi-layer neural networks overcome these limitations to solve complex problems.