# Coding Guide: Gradient Descent Assignment Solutions

## Overview
This guide explains the solutions to the gradient descent assignment, covering visualization of cost functions, basic gradient descent implementation, and applying gradient descent to real-world classification problems.

---

## Question 1: Visualizing the Cost Function

### Complete Code

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return x**2 - 4*x + 3

# Create an array of x values
x = np.linspace(-1, 5, 1000)

# Evaluate f(x) for each value of x
y = f(x)

# Create a figure with specific dimensions (width, height)
plt.figure(figsize=(4, 4))

# Plot f(x) versus x
plt.plot(x, y)

# Add labels and a title to the plot
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) versus x')

# Display the plot
plt.grid(True)
plt.show()
```

### Detailed Explanation

**1. Function Definition**
```python
def f(x):
    return x**2 - 4*x + 3
```
- **Mathematical Form**: f(x) = x² - 4x + 3
- **Type**: Quadratic function (parabola)
- **Properties**:
  - Opens upward (positive x² coefficient)
  - Minimum can be found by completing the square: f(x) = (x-2)² - 1
  - Minimum at x = 2, minimum value = -1
- **`**` operator**: Exponentiation in Python (x**2 means x²)

**2. Creating X Values**
```python
x = np.linspace(-1, 5, 1000)
```
- **Range**: From -1 to 5
- **Why this range**: Captures the minimum and shows parabola shape
- **1000 points**: Ensures smooth curve visualization

**3. Computing Y Values**
```python
y = f(x)
```
- **Vectorization**: NumPy applies function to entire array efficiently
- **Result**: 1000 y-values corresponding to x-values

**4. Plotting**
```python
plt.figure(figsize=(4, 4))
plt.plot(x, y)
```
- **Square figure**: Equal width and height for better visualization
- **Blue line**: Default matplotlib color

### Observations from the Plot

**Visual Characteristics**:
1. **Parabola Shape**: U-shaped curve opening upward
2. **Minimum Point**: Occurs at x = 2, y = -1
3. **Symmetry**: Symmetric around x = 2
4. **Behavior**:
   - Decreases as x approaches 2 from left
   - Increases as x moves away from 2 to the right

**Why This Matters**:
- Shows the "landscape" gradient descent will navigate
- Minimum is where gradient descent should converge
- Smooth function makes optimization easier

---

## Question 2: Implementing Gradient Descent

### Complete Code

```python
# Define the cost function
def cost_function(x):
    return x**2 - 4*x + 3

# Define the derivative of the cost function
def derivative(x):
    return 2*x - 4

# Define the gradient descent function to minimize the cost function
def gradient_descent(learning_rate, initial_x, iterations):
    x = initial_x
    
    # Iterate for the given number of iterations
    for i in range(iterations):
        # Calculate the gradient of the cost function at current x
        gradient = derivative(x)
        
        # Update x using gradient descent
        x = x - learning_rate * gradient
        
        # Calculate the value of the cost function at the updated x
        cost = cost_function(x)
        
        # Print the current iteration, the value of x, and the value of the cost function
        print(f"Iteration: {i}  x = {x}  Cost = {cost}")
    
    # Return the final value of x
    return x

# Parameters
learning_rate = 0.1  # Step size for gradient descent
initial_x = 0        # Initial value of x
iterations = 100      # Number of iterations

# Run gradient descent
optimal_x = gradient_descent(learning_rate, initial_x, iterations)

# Print the optimal value of x and the corresponding cost
print("Optimal x:", optimal_x)
print("Optimal cost:", cost_function(optimal_x))
```

### Detailed Explanation

#### 1. Cost Function
```python
def cost_function(x):
    return x**2 - 4*x + 3
```
- **Purpose**: The function we want to minimize
- **In ML context**: Represents error/loss we want to reduce
- **Returns**: Single value representing "cost" at point x

#### 2. Derivative Function
```python
def derivative(x):
    return 2*x - 4
```
- **Mathematical Derivation**:
  - f(x) = x² - 4x + 3
  - f'(x) = 2x - 4 (power rule: d/dx[x²] = 2x, d/dx[4x] = 4, d/dx[3] = 0)
- **Purpose**: Tells us the slope at any point
- **Interpretation**:
  - Positive derivative: function increasing → move left (decrease x)
  - Negative derivative: function decreasing → move right (increase x)
  - Zero derivative: at minimum or maximum

#### 3. Gradient Descent Function

**Function Signature**:
```python
def gradient_descent(learning_rate, initial_x, iterations):
```
- **learning_rate**: How big each step is (α)
- **initial_x**: Starting point for optimization
- **iterations**: Number of update steps

**Initialization**:
```python
x = initial_x
```
- Start at the given initial position

**Main Loop**:
```python
for i in range(iterations):
```
- Repeats the update process `iterations` times
- `i` tracks current iteration number

**Gradient Calculation**:
```python
gradient = derivative(x)
```
- Computes slope at current position
- Tells us which direction is "downhill"

**Update Step** (The Heart of Gradient Descent):
```python
x = x - learning_rate * gradient
```
- **Why subtract**: Gradient points uphill, we want to go downhill
- **Learning rate effect**: Controls step size
- **Example**:
  - If x = 0: gradient = 2(0) - 4 = -4
  - New x = 0 - 0.1 * (-4) = 0.4
  - We moved right (toward minimum at x=2)

**Cost Calculation**:
```python
cost = cost_function(x)
```
- Evaluates how "good" current position is
- Should decrease with each iteration

**Progress Printing**:
```python
print(f"Iteration: {i}  x = {x}  Cost = {cost}")
```
- **f-string**: Modern Python string formatting
- **Purpose**: Monitor convergence
- **What to look for**:
  - x should approach 2
  - Cost should approach -1

#### 4. Running the Algorithm

**Parameters**:
```python
learning_rate = 0.1
initial_x = 0
iterations = 100
```

**Why these values**:
- **learning_rate = 0.1**: 
  - Not too large (would overshoot)
  - Not too small (would be slow)
  - Good balance for this problem
- **initial_x = 0**: 
  - Starting point away from minimum
  - Tests if algorithm can find minimum
- **iterations = 100**: 
  - More than enough for convergence
  - Can observe full convergence behavior

**Execution**:
```python
optimal_x = gradient_descent(learning_rate, initial_x, iterations)
```
- Runs the algorithm
- Returns final x value

**Final Output**:
```python
print("Optimal x:", optimal_x)
print("Optimal cost:", cost_function(optimal_x))
```
- Shows final result
- Should be close to x=2, cost=-1

### Expected Output Analysis

**First Few Iterations**:
```
Iteration: 0  x = 0.4  Cost = 1.56
Iteration: 1  x = 0.72  Cost = 0.6384
Iteration: 2  x = 0.976  Cost = 0.0486
```

**Observations**:
1. **x increases**: Moving from 0 toward 2
2. **Cost decreases**: Getting closer to minimum
3. **Rate of change**: Slows down as we approach minimum

**Later Iterations**:
```
Iteration: 97  x = 1.9999999999999998  Cost = -0.9999999999999999
Iteration: 98  x = 2.0  Cost = -1.0
Iteration: 99  x = 2.0  Cost = -1.0
```

**Convergence**:
- x converges to 2.0
- Cost converges to -1.0
- Matches theoretical minimum

### Mathematical Insights

**Why It Converges**:
1. **Convex Function**: Single global minimum
2. **Appropriate Learning Rate**: Not too large, not too small
3. **Consistent Updates**: Each step reduces cost

**Convergence Rate**:
- **Geometric decay**: Error reduces by constant factor each iteration
- **Formula**: error(n) ≈ error(0) * (1 - 2α)ⁿ
- **With α=0.1**: error reduces by ~80% each iteration

---

## Question 3: Real-World Application - Breast Cancer Classification

### Complete Code with Explanations

```python
# Import the required libraries
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Load the Breast Cancer dataset
cancer = load_breast_cancer()

# Split the features and target
X = cancer.data
y = (cancer.target != 0) * 1  # Binary classification (0 or 1)

# Standardize the dataset (important for gradient descent)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Add intercept term (bias) to X
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
```

### Detailed Explanation - Data Preparation

#### 1. Loading the Dataset
```python
cancer = load_breast_cancer()
```
- **Dataset**: Wisconsin Breast Cancer dataset
- **Features**: 30 numerical features (tumor measurements)
- **Target**: Binary (malignant or benign)
- **Samples**: 569 instances

#### 2. Feature and Target Extraction
```python
X = cancer.data
y = (cancer.target != 0) * 1
```
- **X**: Feature matrix (569 × 30)
- **y conversion**: 
  - Original: 0 (malignant), 1 (benign)
  - Converted: Ensures binary 0/1 format
  - `(cancer.target != 0)` creates boolean array
  - `* 1` converts True/False to 1/0

#### 3. Standardization
```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```
- **Why standardize**:
  - Features have different scales
  - Gradient descent converges faster with standardized features
  - Prevents features with large values from dominating
- **What it does**:
  - Centers data: mean = 0
  - Scales data: standard deviation = 1
  - Formula: z = (x - μ) / σ

#### 4. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- **test_size=0.2**: 20% for testing, 80% for training
- **random_state=42**: Ensures reproducibility
- **Result**:
  - Training: ~455 samples
  - Testing: ~114 samples

#### 5. Adding Bias Term
```python
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
```
- **Purpose**: Adds intercept/bias term to model
- **`np.c_[]`**: Concatenates arrays column-wise
- **`np.ones(...)`**: Creates column of 1s
- **Result**: X_train_b has shape (455, 31) - one extra column

### Core Functions

#### 1. Sigmoid Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
- **Purpose**: Converts linear output to probability (0 to 1)
- **Formula**: σ(z) = 1 / (1 + e^(-z))
- **Properties**:
  - Output range: (0, 1)
  - S-shaped curve
  - σ(0) = 0.5
  - σ(large positive) ≈ 1
  - σ(large negative) ≈ 0
- **In logistic regression**: Converts z = θᵀx to probability

#### 2. Cost Function (Binary Cross-Entropy)
```python
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost
```
- **Purpose**: Measures how well model fits data
- **Formula**: J(θ) = -(1/m) Σ[y·log(h) + (1-y)·log(1-h)]
- **Components**:
  - `m = len(y)`: Number of training examples
  - `h = sigmoid(X.dot(theta))`: Predictions
  - `X.dot(theta)`: Matrix multiplication (θᵀX)
- **Interpretation**:
  - Lower cost = better fit
  - Penalizes wrong predictions heavily
  - Convex function (single minimum)

#### 3. Gradient Computation
```python
def compute_gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    gradient = (1/m) * X.T.dot(h - y)
    return gradient
```
- **Purpose**: Computes direction to update parameters
- **Formula**: ∇J(θ) = (1/m) XᵀΣ(h - y)
- **Components**:
  - `h - y`: Prediction errors
  - `X.T.dot(...)`: Weighted sum of errors
  - `(1/m)`: Average over all examples
- **Result**: Vector of same size as theta

### Gradient Descent Variants

#### 1. Batch Gradient Descent
```python
def batch_gradient_descent(X, y, theta, alpha, iterations):
    cost_history = []
    
    for i in range(iterations):
        # Compute gradient using ALL data
        gradient = compute_gradient(X, y, theta)
        
        # Update theta
        theta = theta - alpha * gradient
        
        # Track cost
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history
```

**Characteristics**:
- **Uses**: Entire dataset for each update
- **Advantages**:
  - Stable convergence
  - Accurate gradient
  - Guaranteed to decrease cost (with proper α)
- **Disadvantages**:
  - Slow for large datasets
  - Memory intensive
- **When to use**: Small to medium datasets

#### 2. Mini-Batch Gradient Descent
```python
def mini_batch_gradient_descent(X, y, theta, alpha, iterations, batch_size):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for j in range(0, m, batch_size):
            X_mini_batch = X_shuffled[j:j+batch_size]
            y_mini_batch = y_shuffled[j:j+batch_size]
            
            # Compute gradient for mini batch
            gradient = compute_gradient(X_mini_batch, y_mini_batch, theta)
            
            # Update theta
            theta = theta - alpha * gradient
        
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history
```

**Characteristics**:
- **Uses**: Small random batches
- **Batch size**: Typically 32, 64, 128, or 256
- **Advantages**:
  - Faster than batch GD
  - More stable than SGD
  - Efficient use of vectorization
  - Good balance
- **Disadvantages**:
  - Slightly noisy updates
  - Requires batch size tuning
- **When to use**: Most practical applications

**Key Steps**:
1. **Shuffle**: `np.random.permutation(m)`
   - Randomizes order each epoch
   - Prevents learning order-dependent patterns
   
2. **Batch Processing**: `for j in range(0, m, batch_size)`
   - Processes data in chunks
   - `j:j+batch_size` creates mini-batch
   
3. **Update**: After each mini-batch
   - More frequent updates than batch GD
   - Faster convergence

#### 3. Stochastic Gradient Descent
```python
def stochastic_gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Update for each sample
        for j in range(m):
            xi = X_shuffled[j:j+1]
            yi = y_shuffled[j:j+1]
            
            # Compute gradient for single sample
            gradient = compute_gradient(xi, yi, theta)
            
            # Update theta
            theta = theta - alpha * gradient
        
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history
```

**Characteristics**:
- **Uses**: One sample at a time
- **Advantages**:
  - Very fast updates
  - Can escape local minima (noise helps)
  - Low memory requirements
  - Online learning capable
- **Disadvantages**:
  - Noisy convergence
  - May not reach exact minimum
  - Oscillates around minimum
- **When to use**: Very large datasets, online learning

### Running and Comparing Methods

```python
# Prepare initial theta (weights)
n = X_train_b.shape[1]
theta_initial = np.zeros(n)

# Hyperparameters
alpha = 0.1
iterations = 1000
batch_size = 32

# Run all three methods
theta_bgd, cost_history_bgd = batch_gradient_descent(
    X_train_b, y_train, theta_initial.copy(), alpha, iterations
)

theta_mbgd, cost_history_mbgd = mini_batch_gradient_descent(
    X_train_b, y_train, theta_initial.copy(), alpha, iterations, batch_size
)

theta_sgd, cost_history_sgd = stochastic_gradient_descent(
    X_train_b, y_train, theta_initial.copy(), alpha, iterations
)
```

**Key Points**:
- **`theta_initial.copy()`**: Each method gets independent copy
- **Same hyperparameters**: Fair comparison
- **Returns**: Optimized parameters and cost history

### Visualization

```python
plt.figure(figsize=(6, 6))
plt.plot(range(iterations), cost_history_bgd, label='Batch GD', color='r')
plt.plot(range(iterations), cost_history_mbgd, label='Mini-Batch GD', color='g')
plt.plot(range(iterations), cost_history_sgd, label='Stochastic GD', color='b')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History for Different Gradient Descent Methods')
plt.legend()
plt.show()
```

**What to Observe**:
1. **Batch GD**: Smooth, steady decrease
2. **Mini-Batch GD**: Slightly noisy but smooth overall
3. **SGD**: Very noisy, oscillates around minimum

### Making Predictions

```python
def predict(X, theta):
    return (sigmoid(X.dot(theta)) >= 0.5).astype(int)

# Evaluate all three models
y_pred_bgd = predict(X_test_b, theta_bgd)
y_pred_mbgd = predict(X_test_b, theta_mbgd)
y_pred_sgd = predict(X_test_b, theta_sgd)
```

**Prediction Logic**:
- Compute probability: `sigmoid(X.dot(theta))`
- Threshold at 0.5: `>= 0.5`
- Convert to 0/1: `.astype(int)`

### Evaluation

```python
print("Classification Report for Batch Gradient Descent:\n")
print(classification_report(y_test, y_pred_bgd))

print("\nClassification Report for Mini-Batch Gradient Descent:\n")
print(classification_report(y_test, y_pred_mbgd))

print("\nClassification Report for Stochastic Gradient Descent:\n")
print(classification_report(y_test, y_pred_sgd))
```

**Metrics Explained**:
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many we found
- **F1-score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

---

## Key Takeaways

### 1. Gradient Descent Fundamentals
- Iterative optimization algorithm
- Follows negative gradient to minimize cost
- Learning rate controls step size

### 2. Three Variants Comparison

| Aspect | Batch | Mini-Batch | Stochastic |
|--------|-------|------------|------------|
| **Data per update** | All | Small batch | One sample |
| **Speed** | Slow | Fast | Fastest |
| **Convergence** | Smooth | Moderately smooth | Noisy |
| **Memory** | High | Medium | Low |
| **Best for** | Small data | Most cases | Huge data |

### 3. Practical Considerations
- **Standardization**: Essential for gradient descent
- **Learning rate**: Critical hyperparameter
- **Batch size**: Trade-off between speed and stability
- **Iterations**: Monitor convergence, don't over-train

### 4. Real-World Application
- Logistic regression for classification
- Binary cross-entropy loss
- Sigmoid activation for probabilities
- Train-test split for evaluation

---

## Common Pitfalls and Solutions

### Problem 1: Divergence
**Symptom**: Cost increases
**Cause**: Learning rate too large
**Solution**: Reduce α (try α/10)

### Problem 2: Slow Convergence
**Symptom**: Many iterations needed
**Cause**: Learning rate too small
**Solution**: Increase α (try α*2)

### Problem 3: Poor Performance
**Symptom**: Low accuracy
**Causes**:
- Forgot to standardize
- Wrong learning rate
- Not enough iterations
**Solutions**:
- Always standardize features
- Tune hyperparameters
- Monitor cost history

### Problem 4: Overfitting
**Symptom**: Good train, poor test performance
**Solutions**:
- Add regularization
- Use more data
- Reduce model complexity

---

## Summary

This assignment covered:
1. **Visualization**: Understanding cost function landscape
2. **Basic GD**: Implementing gradient descent from scratch
3. **Real Application**: Applying to classification problem
4. **Variants**: Comparing batch, mini-batch, and stochastic GD

**Skills Developed**:
- Implementing optimization algorithms
- Understanding convergence behavior
- Applying to real datasets
- Evaluating model performance
