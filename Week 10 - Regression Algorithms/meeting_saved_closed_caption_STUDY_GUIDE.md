# Regression Algorithms Meeting Study Guide 📚
*Understanding Regression and Statistical Moments Like a Smart 12-Year-Old*

## 🎯 What This Guide Covers
This study guide covers regression algorithms, statistical moments, data distribution analysis, and exploratory data analysis concepts from the meeting transcript.

---

## 🌟 Part 1: Simple Explanations with Illustrations

### 1. What is Regression?
**Simple Explanation:**
Regression is like drawing the best line through scattered dots to predict where new dots might appear!

```
📈 House Price Prediction:
Size (sq ft) → Price ($)
1000 → $200k
1500 → $300k  
2000 → $400k
2500 → ?

Regression finds the pattern: Price = Size × $200 per sq ft
So 2500 sq ft house ≈ $500k
```

### 2. What are Statistical Moments?
**Simple Explanation:**
Statistical moments are like different ways to describe a group of people's heights!

```
👥 Class Height Analysis:

1st Moment (Mean): "Average height is 5'6""
- Tells us the CENTER of the group
- Formula: (h₁ + h₂ + ... + hₙ) ÷ n

2nd Moment (Variance): "Heights vary by ±3 inches"  
- Tells us the SPREAD of the group
- Formula: Average of (height - mean)²

3rd Moment (Skewness): "More tall people than short"
- Tells us if the group is SYMMETRIC
- Formula: Average of (height - mean)³

4th Moment (Kurtosis): "Very few extremely tall/short people"
- Tells us about EXTREME values
- Formula: Average of (height - mean)⁴
```

### 3. What is Normal Distribution?
**Simple Explanation:**
Normal distribution is like a perfect bell-shaped hill where most people are average height, and very few are extremely tall or short!

```
🔔 Bell Curve Properties:
- Peak at the center (mean = median)
- Symmetric on both sides
- 68% within 1 standard deviation
- 95% within 2 standard deviations
- 99.7% within 3 standard deviations

Real Example - Test Scores:
Most students: 70-80 (average)
Some students: 60-70, 80-90 (above/below average)
Few students: <60, >90 (very low/high)
```

---

## 🔬 Part 2: Technical Concepts

### 1. Statistical Moments Implementation
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_moments(data):
    """Calculate the four statistical moments"""
    
    # 1st Moment: Mean (central tendency)
    mean = np.mean(data)
    
    # 2nd Moment: Variance (spread)
    variance = np.var(data, ddof=1)
    std_dev = np.std(data, ddof=1)
    
    # 3rd Moment: Skewness (asymmetry)
    skewness = stats.skew(data)
    
    # 4th Moment: Kurtosis (tail heaviness)
    kurtosis = stats.kurtosis(data)
    
    print(f"Statistical Moments Analysis:")
    print(f"1st Moment (Mean): {mean:.3f}")
    print(f"2nd Moment (Variance): {variance:.3f}")
    print(f"2nd Moment (Std Dev): {std_dev:.3f}")
    print(f"3rd Moment (Skewness): {skewness:.3f}")
    print(f"4th Moment (Kurtosis): {kurtosis:.3f}")
    
    # Interpret skewness
    if abs(skewness) < 0.5:
        skew_interp = "Approximately symmetric"
    elif skewness > 0.5:
        skew_interp = "Right-skewed (positive)"
    else:
        skew_interp = "Left-skewed (negative)"
    
    # Interpret kurtosis
    if abs(kurtosis) < 0.5:
        kurt_interp = "Normal tail behavior"
    elif kurtosis > 0.5:
        kurt_interp = "Heavy tails (leptokurtic)"
    else:
        kurt_interp = "Light tails (platykurtic)"
    
    print(f"\nInterpretations:")
    print(f"Skewness: {skew_interp}")
    print(f"Kurtosis: {kurt_interp}")
    
    return mean, variance, skewness, kurtosis

# Example with different distributions
print("=== Normal Distribution ===")
normal_data = np.random.normal(50, 10, 1000)
calculate_moments(normal_data)

print("\n=== Right-Skewed Distribution ===")
skewed_data = np.random.exponential(2, 1000)
calculate_moments(skewed_data)
```

### 2. Linear Regression Implementation
```python
class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.r_squared = None
    
    def fit(self, X, y):
        """Fit linear regression using least squares"""
        n = len(X)
        
        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope and intercept
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        
        # Calculate R-squared
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Linear Regression Results:")
        print(f"Slope: {self.slope:.4f}")
        print(f"Intercept: {self.intercept:.4f}")
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Equation: y = {self.slope:.4f}x + {self.intercept:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        return self.slope * X + self.intercept

# Example usage
np.random.seed(42)
X = np.random.uniform(1, 10, 50)
y = 2.5 * X + 3 + np.random.normal(0, 2, 50)  # True relationship with noise

model = SimpleLinearRegression()
model.fit(X, y)
```

---

## 🎤 Part 3: Interview Questions & Detailed Answers

### Basic Level Questions

#### Q1: What's the difference between correlation and causation?

**Answer:**
**Correlation** means two variables move together, but **causation** means one variable directly causes changes in another.

```python
# Example: Ice cream sales vs drowning incidents
# Both increase in summer (correlation)
# But ice cream doesn't cause drowning (no causation)
# Hidden variable: hot weather causes both

import numpy as np
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(42)
temperature = np.random.uniform(60, 100, 100)
ice_cream_sales = 2 * temperature + np.random.normal(0, 10, 100)
drowning_incidents = 0.1 * temperature + np.random.normal(0, 2, 100)

correlation = np.corrcoef(ice_cream_sales, drowning_incidents)[0, 1]
print(f"Correlation between ice cream sales and drowning: {correlation:.3f}")
print("But ice cream doesn't cause drowning - temperature is the hidden cause!")
```

#### Q2: Explain the assumptions of linear regression.

**Answer:**
Linear regression has four key assumptions (LINE):

1. **L**inearity: Relationship between X and Y is linear
2. **I**ndependence: Observations are independent
3. **N**ormality: Residuals are normally distributed  
4. **E**qual variance: Constant variance of residuals (homoscedasticity)

```python
def check_regression_assumptions(X, y, model):
    """Check linear regression assumptions"""
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    print("Regression Assumptions Check:")
    print("=" * 40)
    
    # 1. Linearity (visual check)
    print("1. Linearity: Check scatter plot of X vs Y")
    
    # 2. Independence (Durbin-Watson test)
    from statsmodels.stats.diagnostic import durbin_watson
    dw_stat = durbin_watson(residuals)
    print(f"2. Independence: Durbin-Watson = {dw_stat:.3f}")
    print("   (Values near 2.0 indicate independence)")
    
    # 3. Normality of residuals
    _, p_normal = stats.shapiro(residuals)
    print(f"3. Normality: Shapiro-Wilk p-value = {p_normal:.4f}")
    print(f"   {'✓ Normal' if p_normal > 0.05 else '✗ Not normal'}")
    
    # 4. Equal variance (Breusch-Pagan test)
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, p_hetero, _, _ = het_breuschpagan(residuals, X.reshape(-1, 1))
    print(f"4. Equal Variance: Breusch-Pagan p-value = {p_hetero:.4f}")
    print(f"   {'✓ Homoscedastic' if p_hetero > 0.05 else '✗ Heteroscedastic'}")

# Example usage with our previous model
check_regression_assumptions(X, y, model)
```

### Intermediate Level Questions

#### Q3: How do you handle outliers in regression analysis?

**Answer:**
Several methods to detect and handle outliers:

```python
def detect_outliers(X, y, model):
    """Detect outliers using multiple methods"""
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Method 1: Z-score of residuals
    z_scores = np.abs(stats.zscore(residuals))
    z_outliers = np.where(z_scores > 3)[0]
    
    # Method 2: Interquartile Range (IQR)
    Q1 = np.percentile(residuals, 25)
    Q3 = np.percentile(residuals, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
    
    # Method 3: Cook's Distance
    def cooks_distance(X, residuals):
        n, p = len(X), 2  # n observations, p parameters
        mse = np.mean(residuals**2)
        
        # Leverage (hat values)
        X_design = np.column_stack([np.ones(len(X)), X])
        H = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
        leverage = np.diag(H)
        
        # Cook's distance
        cooks_d = (residuals**2 / (p * mse)) * (leverage / (1 - leverage)**2)
        return cooks_d
    
    cooks_d = cooks_distance(X, residuals)
    cook_outliers = np.where(cooks_d > 4/len(X))[0]  # Threshold: 4/n
    
    print("Outlier Detection Results:")
    print(f"Z-score outliers (|z| > 3): {len(z_outliers)} points")
    print(f"IQR outliers: {len(iqr_outliers)} points")
    print(f"Cook's distance outliers: {len(cook_outliers)} points")
    
    return z_outliers, iqr_outliers, cook_outliers

# Handling strategies
def handle_outliers(X, y, outlier_indices, method='remove'):
    """Handle outliers using different strategies"""
    
    if method == 'remove':
        # Remove outliers
        mask = np.ones(len(X), dtype=bool)
        mask[outlier_indices] = False
        return X[mask], y[mask]
    
    elif method == 'winsorize':
        # Cap extreme values
        from scipy.stats.mstats import winsorize
        y_winsorized = winsorize(y, limits=[0.05, 0.05])  # Cap at 5th and 95th percentiles
        return X, y_winsorized
    
    elif method == 'robust_regression':
        # Use robust regression (less sensitive to outliers)
        from sklearn.linear_model import HuberRegressor
        robust_model = HuberRegressor()
        robust_model.fit(X.reshape(-1, 1), y)
        return robust_model

# Example
outliers = detect_outliers(X, y, model)
```

### Advanced Level Questions

#### Q4: Implement and explain regularized regression (Ridge and Lasso).

**Answer:**
Regularized regression adds penalty terms to prevent overfitting:

```python
class RegularizedRegression:
    def __init__(self, alpha=1.0, method='ridge'):
        self.alpha = alpha  # Regularization strength
        self.method = method
        self.weights = None
        self.intercept = None
    
    def fit(self, X, y):
        """Fit regularized regression"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        if self.method == 'ridge':
            # Ridge: minimize ||y - Xw||² + α||w||²
            # Closed form solution: w = (X'X + αI)⁻¹X'y
            XTX = X_with_intercept.T @ X_with_intercept
            identity = np.eye(XTX.shape[0])
            identity[0, 0] = 0  # Don't regularize intercept
            
            weights = np.linalg.inv(XTX + self.alpha * identity) @ X_with_intercept.T @ y
            
        elif self.method == 'lasso':
            # Lasso: minimize ||y - Xw||² + α||w||₁
            # Use coordinate descent (simplified version)
            weights = self._coordinate_descent(X_with_intercept, y)
        
        self.intercept = weights[0]
        self.weights = weights[1:]
        
        print(f"{self.method.title()} Regression (α={self.alpha}):")
        print(f"Intercept: {self.intercept:.4f}")
        print(f"Weights: {self.weights}")
    
    def _coordinate_descent(self, X, y, max_iter=1000, tol=1e-6):
        """Simplified coordinate descent for Lasso"""
        n, p = X.shape
        weights = np.zeros(p)
        
        for iteration in range(max_iter):
            weights_old = weights.copy()
            
            for j in range(p):
                # Compute residual without j-th feature
                residual = y - X @ weights + weights[j] * X[:, j]
                rho = X[:, j] @ residual
                
                if j == 0:  # Don't regularize intercept
                    weights[j] = rho / (X[:, j] @ X[:, j])
                else:
                    # Soft thresholding
                    z = X[:, j] @ X[:, j]
                    if rho > self.alpha:
                        weights[j] = (rho - self.alpha) / z
                    elif rho < -self.alpha:
                        weights[j] = (rho + self.alpha) / z
                    else:
                        weights[j] = 0
            
            # Check convergence
            if np.sum(np.abs(weights - weights_old)) < tol:
                break
        
        return weights
    
    def predict(self, X):
        """Make predictions"""
        return self.intercept + X @ self.weights

# Compare different regularization methods
def compare_regularization():
    """Compare Ridge, Lasso, and OLS regression"""
    
    # Generate data with some irrelevant features
    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    
    # Only first 3 features are relevant
    true_weights = np.array([2, -1, 0.5] + [0] * 7)
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Fit different models
    models = {
        'OLS': SimpleLinearRegression(),
        'Ridge': RegularizedRegression(alpha=1.0, method='ridge'),
        'Lasso': RegularizedRegression(alpha=0.1, method='lasso')
    }
    
    print("Regularization Comparison:")
    print("True weights:", true_weights[:5])  # Show first 5
    print()
    
    for name, model in models.items():
        if name == 'OLS':
            # Use first feature only for simple regression
            model.fit(X[:, 0], y)
            print(f"{name}: weight = {model.slope:.4f}")
        else:
            model.fit(X, y)
            print(f"{name}: weights = {model.weights[:5]}")  # Show first 5
        print()

compare_regularization()
```

---

## 📚 Key Concepts from the Meeting

### 1. **Statistical Moments:**
- 1st moment: Mean (central tendency)
- 2nd moment: Variance (spread)
- 3rd moment: Skewness (asymmetry)
- 4th moment: Kurtosis (tail behavior)

### 2. **Data Distribution Analysis:**
- Normal distribution properties
- Outlier detection methods
- Visualization techniques (histograms, box plots)

### 3. **Regression Fundamentals:**
- Linear relationships
- Least squares estimation
- Model assumptions and diagnostics

---

*Remember: Regression interviews focus on understanding assumptions, handling violations, and knowing when to use different techniques. Practice both the mathematical concepts and practical implementation!* 🎯