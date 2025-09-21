# Week 13: Calculus and Gradient Descent - Study Guide

## Part 1: Simple Explanations (For a 12-year-old)

### What is Calculus?
Imagine you're riding a bike and want to know how fast you're going at any exact moment. Calculus helps us figure that out! It's like having a super-smart speedometer that can tell you your speed at any tiny instant in time.

**The Speed Example:**
- If you travel 200 kilometers in 2 hours, your average speed is 100 km/hour
- But you don't go exactly 100 km/hour the whole time - sometimes faster, sometimes slower
- Calculus helps us find your exact speed at any moment by looking at really, really tiny time periods

### What is a Derivative?
A derivative is like asking "How fast is something changing right now?"
- Speed is the derivative of distance (how fast distance is changing)
- If you're on a hill, the steepness is the derivative of height (how fast height is changing)

**Simple Analogy:**
Think of a roller coaster. The derivative tells you:
- How steep the track is at any point
- Whether you're going up (positive) or down (negative)
- How fast you're climbing or dropping

### What is Gradient Descent?
Imagine you're blindfolded on a hill and want to get to the bottom (lowest point). Gradient descent is like:
1. Feel the ground around your feet
2. Figure out which direction goes downhill the most
3. Take a step in that direction
4. Repeat until you reach the bottom

**Why Do We Care?**
In machine learning, we want to find the "best" answer (like the bottom of the hill). Gradient descent helps our computer "walk downhill" to find the best solution automatically!

### Types of Gradient Descent (Simple Version)
1. **Full Batch**: Look at ALL your data before taking each step (slow but careful)
2. **Mini Batch**: Look at some of your data before each step (balanced approach)
3. **Stochastic**: Look at just ONE piece of data before each step (fast but jumpy)

---

## Part 2: Technical Deep Dive

### Mathematical Foundations

#### Derivatives and Limits
The derivative of a function f(x) is defined as:
```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

**Key Concepts:**
- **Rate of Change**: Derivative measures instantaneous rate of change
- **Slope**: Geometrically, derivative is the slope of the tangent line
- **Limit**: We approach infinitesimally small intervals (dt) but never reach zero

#### Partial Derivatives
For functions with multiple variables f(x,y):
```
∂f/∂x = partial derivative with respect to x (treating y as constant)
∂f/∂y = partial derivative with respect to y (treating x as constant)
```

#### Gradient Vector
The gradient ∇f is a vector containing all partial derivatives:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Properties:**
- Points in direction of steepest increase
- Magnitude indicates rate of increase
- Perpendicular to level curves/surfaces

### Gradient Descent Algorithm

#### Basic Algorithm
```python
# Initialize parameters
θ = initial_values
learning_rate = α

# Iterative optimization
for iteration in range(max_iterations):
    # Calculate gradient
    gradient = ∇J(θ)
    
    # Update parameters
    θ = θ - α * gradient
    
    # Check convergence
    if ||gradient|| < tolerance:
        break
```

#### Mathematical Formulation
For cost function J(θ):
```
θ(t+1) = θ(t) - α * ∇J(θ(t))
```

Where:
- θ: parameters to optimize
- α: learning rate (step size)
- ∇J(θ): gradient of cost function

### Variants of Gradient Descent

#### 1. Batch Gradient Descent (Full Batch)
```python
# Use entire dataset for each update
gradient = (1/m) * Σ(i=1 to m) ∇J(θ, x_i, y_i)
θ = θ - α * gradient
```

**Characteristics:**
- Stable convergence
- Computationally expensive for large datasets
- Guaranteed to converge to global minimum (for convex functions)

#### 2. Stochastic Gradient Descent (SGD)
```python
# Use single sample for each update
for each sample (x_i, y_i):
    gradient = ∇J(θ, x_i, y_i)
    θ = θ - α * gradient
```

**Characteristics:**
- Fast updates
- Noisy convergence path
- Can escape local minima due to noise
- May oscillate around minimum

#### 3. Mini-Batch Gradient Descent
```python
# Use small batch of samples
for each mini_batch:
    gradient = (1/batch_size) * Σ(samples in batch) ∇J(θ, x_i, y_i)
    θ = θ - α * gradient
```

**Characteristics:**
- Balance between stability and speed
- Efficient use of vectorized operations
- Most commonly used in practice

### Learning Rate Considerations

#### Effects of Learning Rate:
- **Too Large**: Overshooting, divergence, oscillation
- **Too Small**: Slow convergence, getting stuck
- **Adaptive**: Methods like Adam, RMSprop adjust learning rate automatically

#### Learning Rate Schedules:
```python
# Step decay
α(t) = α₀ * γ^(floor(t/step_size))

# Exponential decay
α(t) = α₀ * e^(-λt)

# Polynomial decay
α(t) = α₀ * (1 + γt)^(-power)
```

### Convergence Analysis

#### Convex Functions:
- Single global minimum
- Gradient descent guaranteed to converge
- Convergence rate depends on condition number

#### Non-Convex Functions:
- Multiple local minima
- May get trapped in local minimum
- Modern techniques: momentum, adaptive learning rates

---

## Part 3: Interview Questions and Detailed Answers

### Q1: Explain the mathematical intuition behind gradient descent. Why does it work?

**Answer:**
Gradient descent works based on the fundamental principle that the gradient of a function points in the direction of steepest increase. Here's the mathematical intuition:

1. **Taylor Series Expansion**: For a function f(x), we can approximate:
   ```
   f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx
   ```

2. **Minimization Goal**: To minimize f(x), we want f(x + Δx) < f(x), which means:
   ```
   ∇f(x)ᵀΔx < 0
   ```

3. **Optimal Direction**: The direction that maximizes the decrease is Δx = -α∇f(x), where α > 0.

4. **Convergence**: For convex functions, this process is guaranteed to reach the global minimum because:
   - The gradient becomes zero at the minimum
   - Each step reduces the function value
   - The algorithm cannot overshoot indefinitely with proper learning rate

**Code Example:**
```python
import numpy as np

def gradient_descent_1d(f, df, x0, learning_rate=0.01, max_iter=1000):
    """
    1D gradient descent example
    f: function to minimize
    df: derivative of f
    """
    x = x0
    history = [x]
    
    for i in range(max_iter):
        grad = df(x)
        x = x - learning_rate * grad
        history.append(x)
        
        if abs(grad) < 1e-6:  # Convergence check
            break
    
    return x, history

# Example: minimize f(x) = x^2
f = lambda x: x**2
df = lambda x: 2*x

minimum, path = gradient_descent_1d(f, df, x0=5.0)
print(f"Minimum found at x = {minimum}")
```

### Q2: Compare and contrast the three main variants of gradient descent. When would you use each?

**Answer:**

| Aspect | Batch GD | Mini-Batch GD | Stochastic GD |
|--------|----------|---------------|---------------|
| **Data Usage** | Entire dataset | Small batches | Single samples |
| **Memory** | High | Moderate | Low |
| **Convergence** | Smooth | Moderately smooth | Noisy |
| **Speed per epoch** | Slow | Fast | Fastest |
| **Parallelization** | Excellent | Good | Poor |
| **Generalization** | May overfit | Balanced | Better generalization |

**When to Use Each:**

1. **Batch Gradient Descent:**
   ```python
   # Use when:
   # - Small datasets (< 10K samples)
   # - Need stable convergence
   # - Convex optimization problems
   
   def batch_gradient_descent(X, y, theta, learning_rate, epochs):
       m = len(y)
       for epoch in range(epochs):
           # Compute gradient using ALL data
           predictions = X.dot(theta)
           gradient = (1/m) * X.T.dot(predictions - y)
           theta = theta - learning_rate * gradient
       return theta
   ```

2. **Mini-Batch Gradient Descent:**
   ```python
   # Use when:
   # - Medium to large datasets
   # - Want balance of speed and stability
   # - GPU acceleration available
   
   def mini_batch_gd(X, y, theta, learning_rate, epochs, batch_size=32):
       m = len(y)
       for epoch in range(epochs):
           # Shuffle data
           indices = np.random.permutation(m)
           X_shuffled = X[indices]
           y_shuffled = y[indices]
           
           # Process mini-batches
           for i in range(0, m, batch_size):
               X_batch = X_shuffled[i:i+batch_size]
               y_batch = y_shuffled[i:i+batch_size]
               
               predictions = X_batch.dot(theta)
               gradient = (1/len(y_batch)) * X_batch.T.dot(predictions - y_batch)
               theta = theta - learning_rate * gradient
       return theta
   ```

3. **Stochastic Gradient Descent:**
   ```python
   # Use when:
   # - Very large datasets
   # - Online learning scenarios
   # - Need to escape local minima
   
   def stochastic_gd(X, y, theta, learning_rate, epochs):
       m = len(y)
       for epoch in range(epochs):
           for i in range(m):
               # Use single sample
               xi = X[i:i+1]
               yi = y[i:i+1]
               
               prediction = xi.dot(theta)
               gradient = xi.T.dot(prediction - yi)
               theta = theta - learning_rate * gradient
       return theta
   ```

### Q3: What are the common problems with gradient descent and how do you solve them?

**Answer:**

#### Problem 1: Local Minima
**Issue**: Getting trapped in local minima in non-convex functions.

**Solutions:**
```python
# 1. Random Restarts
def gradient_descent_with_restarts(f, df, num_restarts=10):
    best_x = None
    best_value = float('inf')
    
    for _ in range(num_restarts):
        x0 = np.random.uniform(-10, 10)  # Random initialization
        x_final, _ = gradient_descent(f, df, x0)
        value = f(x_final)
        
        if value < best_value:
            best_value = value
            best_x = x_final
    
    return best_x

# 2. Momentum (helps escape local minima)
def gradient_descent_momentum(f, df, x0, lr=0.01, momentum=0.9, max_iter=1000):
    x = x0
    velocity = 0
    
    for i in range(max_iter):
        grad = df(x)
        velocity = momentum * velocity - lr * grad
        x = x + velocity
        
        if abs(grad) < 1e-6:
            break
    
    return x
```

#### Problem 2: Choosing Learning Rate
**Issue**: Too large causes divergence, too small causes slow convergence.

**Solutions:**
```python
# 1. Learning Rate Scheduling
def adaptive_learning_rate(epoch, initial_lr=0.01):
    # Step decay
    if epoch < 100:
        return initial_lr
    elif epoch < 200:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01

# 2. Line Search
def gradient_descent_line_search(f, df, x0, max_iter=1000):
    x = x0
    
    for i in range(max_iter):
        grad = df(x)
        
        # Find optimal step size using line search
        alpha = line_search(f, x, -grad)  # Search along negative gradient
        x = x - alpha * grad
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x

# 3. Adaptive Methods (Adam-like)
def adam_optimizer(f, df, x0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    x = x0
    m = 0  # First moment
    v = 0  # Second moment
    
    for t in range(1, 1001):
        grad = df(x)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        if abs(grad) < 1e-6:
            break
    
    return x
```

#### Problem 3: Saddle Points
**Issue**: Gradient is zero but point is not a minimum.

**Solutions:**
```python
# 1. Check Second Derivative (Hessian)
def check_critical_point(f, df, d2f, x):
    grad = df(x)
    hessian = d2f(x)
    
    if abs(grad) < 1e-6:
        if hessian > 0:
            return "Local minimum"
        elif hessian < 0:
            return "Local maximum"
        else:
            return "Saddle point"
    return "Not critical point"

# 2. Add Noise to Escape Saddle Points
def noisy_gradient_descent(f, df, x0, lr=0.01, noise_std=0.01):
    x = x0
    
    for i in range(1000):
        grad = df(x)
        noise = np.random.normal(0, noise_std)
        x = x - lr * grad + noise
        
        if abs(grad) < 1e-6:
            break
    
    return x
```

### Q4: How do you implement gradient descent for a real machine learning problem?

**Answer:**
Let's implement gradient descent for linear regression as a complete example:

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_history = []
        
    def add_intercept(self, X):
        """Add bias term (intercept) to features"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def cost_function(self, X, y, theta):
        """Mean Squared Error cost function"""
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost
    
    def gradient(self, X, y, theta):
        """Compute gradient of cost function"""
        m = len(y)
        predictions = X.dot(theta)
        gradient = (1/m) * X.T.dot(predictions - y)
        return gradient
    
    def fit(self, X, y, method='batch'):
        """Train the model using gradient descent"""
        # Add intercept term
        X_with_intercept = self.add_intercept(X)
        m, n = X_with_intercept.shape
        
        # Initialize parameters
        self.theta = np.random.normal(0, 0.01, n)
        
        if method == 'batch':
            self._batch_gradient_descent(X_with_intercept, y)
        elif method == 'stochastic':
            self._stochastic_gradient_descent(X_with_intercept, y)
        elif method == 'mini_batch':
            self._mini_batch_gradient_descent(X_with_intercept, y)
    
    def _batch_gradient_descent(self, X, y):
        """Full batch gradient descent"""
        for i in range(self.max_iterations):
            cost = self.cost_function(X, y, self.theta)
            self.cost_history.append(cost)
            
            grad = self.gradient(X, y, self.theta)
            self.theta = self.theta - self.learning_rate * grad
            
            # Check convergence
            if np.linalg.norm(grad) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
    
    def _stochastic_gradient_descent(self, X, y):
        """Stochastic gradient descent"""
        m = len(y)
        
        for epoch in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            for i in range(m):
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                
                # Compute cost and gradient for single sample
                cost = self.cost_function(xi, yi, self.theta)
                epoch_cost += cost
                
                grad = self.gradient(xi, yi, self.theta)
                self.theta = self.theta - self.learning_rate * grad
            
            self.cost_history.append(epoch_cost / m)
            
            # Check convergence (less strict for SGD)
            if epoch > 10 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                print(f"Converged after {epoch+1} epochs")
                break
    
    def _mini_batch_gradient_descent(self, X, y, batch_size=32):
        """Mini-batch gradient descent"""
        m = len(y)
        
        for epoch in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            num_batches = 0
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                cost = self.cost_function(X_batch, y_batch, self.theta)
                epoch_cost += cost
                num_batches += 1
                
                grad = self.gradient(X_batch, y_batch, self.theta)
                self.theta = self.theta - self.learning_rate * grad
            
            avg_cost = epoch_cost / num_batches
            self.cost_history.append(avg_cost)
            
            # Check convergence
            if epoch > 10 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                print(f"Converged after {epoch+1} epochs")
                break
    
    def predict(self, X):
        """Make predictions on new data"""
        X_with_intercept = self.add_intercept(X)
        return X_with_intercept.dot(self.theta)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    m = 1000  # Number of samples
    X = np.random.randn(m, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(m) * 0.5
    
    # Split data
    split_idx = int(0.8 * m)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train models with different methods
    methods = ['batch', 'stochastic', 'mini_batch']
    models = {}
    
    for method in methods:
        print(f"\nTraining with {method} gradient descent...")
        model = LinearRegressionGD(learning_rate=0.01, max_iterations=1000)
        model.fit(X_train, y_train, method=method)
        models[method] = model
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = np.mean((train_pred - y_train)**2)
        test_mse = np.mean((test_pred - y_test)**2)
        
        print(f"Final parameters: {model.theta}")
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
    
    # Plot cost histories
    plt.figure(figsize=(15, 5))
    for i, method in enumerate(methods):
        plt.subplot(1, 3, i+1)
        plt.plot(models[method].cost_history)
        plt.title(f'{method.title()} Gradient Descent')
        plt.xlabel('Iteration/Epoch')
        plt.ylabel('Cost')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

This implementation demonstrates:
1. **Complete workflow**: Data preparation, training, evaluation
2. **Multiple variants**: Batch, stochastic, and mini-batch
3. **Convergence checking**: Both gradient norm and cost change
4. **Practical considerations**: Data shuffling, parameter initialization
5. **Evaluation metrics**: Training and testing performance
6. **Visualization**: Cost function convergence

The key insights from this implementation:
- **Batch GD**: Smooth convergence, slower per iteration
- **Stochastic GD**: Faster updates, noisier convergence
- **Mini-batch GD**: Good balance, most practical for real applications

### Q5: How does gradient descent relate to other optimization algorithms used in modern deep learning?

**Answer:**

Gradient descent is the foundation for all modern optimization algorithms. Here's how it evolved:

#### 1. **Momentum-Based Methods**
```python
# Standard Momentum
def momentum_gd(params, grads, velocities, lr=0.01, momentum=0.9):
    for i in range(len(params)):
        velocities[i] = momentum * velocities[i] - lr * grads[i]
        params[i] += velocities[i]
    return params, velocities

# Nesterov Accelerated Gradient
def nesterov_gd(params, grads, velocities, lr=0.01, momentum=0.9):
    for i in range(len(params)):
        v_prev = velocities[i]
        velocities[i] = momentum * velocities[i] - lr * grads[i]
        params[i] += -momentum * v_prev + (1 + momentum) * velocities[i]
    return params, velocities
```

#### 2. **Adaptive Learning Rate Methods**
```python
# AdaGrad
def adagrad(params, grads, sum_squared_grads, lr=0.01, eps=1e-8):
    for i in range(len(params)):
        sum_squared_grads[i] += grads[i]**2
        params[i] -= lr * grads[i] / (np.sqrt(sum_squared_grads[i]) + eps)
    return params, sum_squared_grads

# RMSprop
def rmsprop(params, grads, avg_squared_grads, lr=0.01, decay=0.9, eps=1e-8):
    for i in range(len(params)):
        avg_squared_grads[i] = decay * avg_squared_grads[i] + (1-decay) * grads[i]**2
        params[i] -= lr * grads[i] / (np.sqrt(avg_squared_grads[i]) + eps)
    return params, avg_squared_grads

# Adam (combines momentum + adaptive learning rates)
def adam(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    for i in range(len(params)):
        # Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        
        # Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i]**2
        
        # Compute bias-corrected first moment estimate
        m_hat = m[i] / (1 - beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v[i] / (1 - beta2**t)
        
        # Update parameters
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return params, m, v
```

#### 3. **Modern Variants and Improvements**
```python
# AdamW (Adam with decoupled weight decay)
def adamw(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, 
          eps=1e-8, weight_decay=0.01):
    for i in range(len(params)):
        # Weight decay
        params[i] *= (1 - lr * weight_decay)
        
        # Adam updates
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i]**2
        
        m_hat = m[i] / (1 - beta1**t)
        v_hat = v[i] / (1 - beta2**t)
        
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return params, m, v

# Learning Rate Scheduling
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        
    def step(self, epoch):
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        self.optimizer.lr = lr
```

**Key Evolution Points:**

1. **Problem**: Vanilla GD oscillates in ravines
   **Solution**: Momentum smooths updates

2. **Problem**: Fixed learning rate doesn't work for all parameters
   **Solution**: Adaptive methods (AdaGrad, RMSprop, Adam)

3. **Problem**: Adam can fail to converge in some cases
   **Solution**: AdamW, better learning rate schedules

4. **Problem**: Need better generalization
   **Solution**: Weight decay, dropout, batch normalization

**Modern Best Practices:**
- Use Adam/AdamW as default
- Implement learning rate scheduling
- Add regularization techniques
- Use gradient clipping for stability
- Consider second-order methods for specific problems

This progression shows how gradient descent evolved from a simple optimization technique to the sophisticated optimizers powering modern deep learning systems.