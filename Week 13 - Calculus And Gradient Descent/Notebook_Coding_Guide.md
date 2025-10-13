# Coding Guide: Gradient Descent Notebook

## Overview
This notebook introduces gradient descent, a fundamental optimization algorithm used in machine learning to minimize cost functions. It demonstrates gradient descent with fixed learning rates and visualizes the optimization process.

---

## Section 1: Plotting the Cost Function

### Code Block 1: Import Libraries and Define Function

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return (x - 5) ** 2 + 20
```

**Purpose**: Set up the environment and define the cost function we want to minimize.

**Key Components**:

1. **`import numpy as np`**
   - **Why**: NumPy provides efficient numerical operations and array handling
   - **Usage**: Used for creating arrays and mathematical operations

2. **`import matplotlib.pyplot as plt`**
   - **Why**: Matplotlib is the standard plotting library in Python
   - **Usage**: Used to visualize the cost function and optimization process

3. **`def f(x):`**
   - **Purpose**: Defines the cost function f(x) = (x - 5)² + 20
   - **Parameters**: 
     - `x`: Input value (can be a single number or NumPy array)
   - **Returns**: The computed function value
   - **Mathematical Form**: This is a parabola with:
     - Minimum at x = 5
     - Minimum value = 20
     - Opens upward (positive coefficient)

### Code Block 2: Create Data Points and Plot

```python
# Create an array of x values
x = np.linspace(-10, 20, 1000)

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

**Key Components**:

1. **`np.linspace(-10, 20, 1000)`**
   - **Purpose**: Creates 1000 evenly spaced points between -10 and 20
   - **Why 1000 points**: Provides smooth curve visualization
   - **Arguments**:
     - `-10`: Start value
     - `20`: End value
     - `1000`: Number of points to generate

2. **`y = f(x)`**
   - **Purpose**: Computes function values for all x points
   - **How it works**: NumPy's vectorization allows applying the function to entire array at once
   - **Result**: Array of 1000 y-values corresponding to x-values

3. **`plt.figure(figsize=(4, 4))`**
   - **Purpose**: Creates a new figure with specified dimensions
   - **Arguments**: `figsize=(width, height)` in inches
   - **Why specify size**: Controls the aspect ratio and readability

4. **`plt.plot(x, y)`**
   - **Purpose**: Creates a line plot
   - **Arguments**:
     - `x`: X-axis values
     - `y`: Y-axis values
   - **Default behavior**: Connects points with lines

5. **`plt.xlabel()`, `plt.ylabel()`, `plt.title()`**
   - **Purpose**: Add descriptive labels to the plot
   - **Best Practice**: Always label axes for clarity

6. **`plt.grid(True)`**
   - **Purpose**: Adds a grid to the plot
   - **Why useful**: Helps read values from the graph

7. **`plt.show()`**
   - **Purpose**: Displays the plot
   - **Note**: In Jupyter notebooks, this is often optional

---

## Section 2: Gradient Descent Implementation

### Understanding Gradient Descent

**What is Gradient Descent?**
- An iterative optimization algorithm
- Finds the minimum of a function by following the negative gradient
- Like walking downhill to reach the lowest point

**Key Concepts**:
1. **Gradient**: The slope/derivative of the function at a point
2. **Learning Rate (α)**: Step size for each iteration
3. **Iteration**: One update step in the algorithm

**Update Rule**:
```
x_new = x_old - learning_rate * gradient
```

### Code Implementation (Expected in Full Notebook)

```python
def gradient_descent_fixed_lr(initial_x, learning_rate, iterations):
    """
    Performs gradient descent with a fixed learning rate
    
    Parameters:
    -----------
    initial_x : float
        Starting point for optimization
    learning_rate : float
        Step size for each iteration (α)
    iterations : int
        Number of optimization steps
        
    Returns:
    --------
    x_history : list
        History of x values at each iteration
    cost_history : list
        History of cost function values
    """
    x = initial_x
    x_history = [x]
    cost_history = [f(x)]
    
    for i in range(iterations):
        # Calculate gradient (derivative)
        gradient = 2 * (x - 5)  # Derivative of (x-5)² + 20
        
        # Update x using gradient descent rule
        x = x - learning_rate * gradient
        
        # Store history
        x_history.append(x)
        cost_history.append(f(x))
        
    return x_history, cost_history
```

**Key Components Explained**:

1. **Function Parameters**:
   - `initial_x`: Where to start the optimization
   - `learning_rate`: Controls how big each step is
   - `iterations`: How many steps to take

2. **Gradient Calculation**:
   ```python
   gradient = 2 * (x - 5)
   ```
   - **Why**: Derivative of f(x) = (x-5)² + 20 is f'(x) = 2(x-5)
   - **Interpretation**: 
     - If x < 5: gradient is negative → move right (increase x)
     - If x > 5: gradient is positive → move left (decrease x)

3. **Update Step**:
   ```python
   x = x - learning_rate * gradient
   ```
   - **Why subtract**: We want to go downhill (minimize)
   - **Learning rate effect**:
     - Too large: May overshoot minimum
     - Too small: Slow convergence

4. **History Tracking**:
   - Stores x and cost values at each step
   - Useful for visualization and analysis

### Visualization of Gradient Descent

```python
# Run gradient descent
initial_x = -5
learning_rate = 0.1
iterations = 50

x_history, cost_history = gradient_descent_fixed_lr(initial_x, learning_rate, iterations)

# Plot the function and optimization path
plt.figure(figsize=(12, 5))

# Left plot: Function with optimization path
plt.subplot(1, 2, 1)
x_range = np.linspace(-10, 20, 1000)
plt.plot(x_range, f(x_range), 'b-', label='f(x)')
plt.plot(x_history, cost_history, 'ro-', markersize=4, label='Optimization path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Path')
plt.legend()
plt.grid(True)

# Right plot: Cost vs iteration
plt.subplot(1, 2, 2)
plt.plot(cost_history, 'g-')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Over Iterations')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final x: {x_history[-1]:.6f}")
print(f"Final cost: {cost_history[-1]:.6f}")
print(f"True minimum at x=5, cost=20")
```

**Visualization Components**:

1. **`plt.subplot(1, 2, 1)`**
   - **Purpose**: Creates a grid of subplots
   - **Arguments**: (rows, columns, index)
   - **Usage**: Allows multiple plots in one figure

2. **Left Plot - Optimization Path**:
   - Shows the function curve
   - Overlays the path taken by gradient descent
   - Red dots show where algorithm visited

3. **Right Plot - Convergence**:
   - Shows how cost decreases over iterations
   - Helps diagnose convergence issues

4. **`plt.tight_layout()`**:
   - **Purpose**: Automatically adjusts subplot spacing
   - **Why**: Prevents overlapping labels

---

## Key Learning Points

### 1. Learning Rate Selection

**Too Large (e.g., α = 0.5)**:
```python
# May oscillate or diverge
x_history, cost_history = gradient_descent_fixed_lr(-5, 0.5, 50)
# Observe: Might overshoot and oscillate around minimum
```

**Too Small (e.g., α = 0.01)**:
```python
# Slow convergence
x_history, cost_history = gradient_descent_fixed_lr(-5, 0.01, 50)
# Observe: Takes many iterations to reach minimum
```

**Good Choice (e.g., α = 0.1)**:
```python
# Balanced convergence
x_history, cost_history = gradient_descent_fixed_lr(-5, 0.1, 50)
# Observe: Smooth, relatively fast convergence
```

### 2. Convergence Criteria

Instead of fixed iterations, we can stop when:
```python
def gradient_descent_with_tolerance(initial_x, learning_rate, tolerance=1e-6, max_iterations=1000):
    x = initial_x
    
    for i in range(max_iterations):
        gradient = 2 * (x - 5)
        
        # Check if gradient is small enough
        if abs(gradient) < tolerance:
            print(f"Converged after {i} iterations")
            break
            
        x = x - learning_rate * gradient
    
    return x
```

**Why tolerance**:
- Gradient near zero indicates we're at minimum
- Saves unnecessary iterations
- More efficient than fixed iteration count

### 3. Common Issues and Solutions

**Problem 1: Divergence**
- **Symptom**: Cost increases instead of decreasing
- **Cause**: Learning rate too large
- **Solution**: Reduce learning rate

**Problem 2: Slow Convergence**
- **Symptom**: Many iterations needed
- **Cause**: Learning rate too small
- **Solution**: Increase learning rate (carefully)

**Problem 3: Getting Stuck**
- **Symptom**: Stops before reaching minimum
- **Cause**: Numerical precision issues
- **Solution**: Adjust tolerance or use better precision

---

## Mathematical Insights

### Why Gradient Descent Works

1. **Taylor Series Approximation**:
   ```
   f(x + Δx) ≈ f(x) + f'(x)·Δx
   ```
   - To minimize f, we want f(x + Δx) < f(x)
   - This happens when f'(x)·Δx < 0
   - Choosing Δx = -α·f'(x) guarantees this (for small α)

2. **Convexity**:
   - Our function f(x) = (x-5)² + 20 is convex
   - Convex functions have single global minimum
   - Gradient descent guaranteed to find it

3. **Convergence Rate**:
   - For convex quadratic functions: linear convergence
   - Each step reduces error by constant factor
   - Number of iterations ∝ log(1/ε) for error ε

---

## Practice Exercises

### Exercise 1: Different Starting Points
Try gradient descent from different initial values:
```python
starting_points = [-10, -5, 0, 10, 15]
for start in starting_points:
    x_hist, cost_hist = gradient_descent_fixed_lr(start, 0.1, 50)
    print(f"Start: {start}, Final: {x_hist[-1]:.4f}, Iterations to converge: {len(x_hist)}")
```

### Exercise 2: Learning Rate Comparison
Compare different learning rates:
```python
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
for lr in learning_rates:
    x_hist, cost_hist = gradient_descent_fixed_lr(-5, lr, 50)
    print(f"LR: {lr}, Final cost: {cost_hist[-1]:.6f}")
```

### Exercise 3: Custom Function
Implement gradient descent for f(x) = x⁴ - 3x³ + 2:
```python
def f_new(x):
    return x**4 - 3*x**3 + 2

def gradient_new(x):
    return 4*x**3 - 9*x**2

# Implement gradient descent for this function
```

---

## Summary

**What We Learned**:
1. How to visualize cost functions
2. Basic gradient descent implementation
3. Impact of learning rate on convergence
4. How to track and visualize optimization progress

**Key Takeaways**:
- Gradient descent is an iterative optimization method
- Learning rate is crucial for convergence
- Visualization helps understand algorithm behavior
- Simple functions help build intuition for complex cases

**Next Steps**:
- Learn about variants (batch, mini-batch, stochastic)
- Explore adaptive learning rates
- Apply to real machine learning problems
