# Coding Guide: Pearson Correlation Demo

## Overview
This notebook demonstrates how to calculate and interpret the Pearson Correlation Coefficient (PCC) using three different Python libraries. Correlation measures the strength and direction of the linear relationship between two variables.

---

## Objective

**Learn to calculate Pearson correlation coefficient using:**
1. Pandas
2. SciPy
3. NumPy

**Understand:**
- What correlation means
- How to interpret correlation values
- When to use correlation analysis
- Correlation vs causation

---

## What is Correlation?

### Simple Explanation

**Correlation** tells us if two things tend to change together:
- When one goes up, does the other go up? (Positive correlation)
- When one goes up, does the other go down? (Negative correlation)
- Do they not follow any pattern? (No correlation)

### Real-World Examples

**Positive Correlation** (+):
- Study hours ↑ → Test scores ↑
- Temperature ↑ → Ice cream sales ↑
- Exercise ↑ → Fitness ↑

**Negative Correlation** (-):
- Speed ↑ → Travel time ↓
- Price ↑ → Demand ↓
- Sleep deprivation ↑ → Performance ↓

**No Correlation** (0):
- Shoe size ↔ Intelligence
- Hair color ↔ Math ability
- Day of week ↔ Height

---

## Library Imports

```python
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
```

### Why Each Library?

1. **numpy (np)**:
   - Numerical computations
   - Array operations
   - Has `np.corrcoef()` function

2. **pandas (pd)**:
   - Data manipulation
   - DataFrame operations
   - Has `.corr()` method for correlation matrices

3. **scipy.stats**:
   - Statistical functions
   - `pearsonr()` function
   - Returns correlation and p-value

4. **seaborn (sns)**:
   - Statistical visualization
   - Built-in datasets
   - Scatter plots and heatmaps

---

## Loading the Dataset

```python
df = sns.load_dataset("car_crashes")
```

### About the Dataset

**What it contains**:
- Car crash statistics by US state
- Multiple features related to driving behavior
- Insurance premiums
- Demographic information

**Key Features**:
- `speeding`: Percentage of drivers speeding
- `alcohol`: Percentage of drivers involved in fatal collisions who were alcohol-impaired
- `ins_premium`: Car insurance premiums ($)
- And more...

### Viewing the Data

```python
df.head()
```

**What it shows**:
- First 5 rows of the dataset
- Column names and sample values
- Data types and structure

**Sample Output**:
```
   total  speeding  alcohol  ...  ins_premium  ins_losses
0   18.8      7.33     5.64  ...       784.55      145.08
1   18.1      7.79     4.52  ...      1053.48      133.93
...
```

---

## Visualizing Relationships

### Scatter Plot: Speeding vs Alcohol

```python
sns.scatterplot(df, x='speeding', y='alcohol')
```

### Understanding Scatter Plots

**Components**:
- **X-axis**: Independent variable (speeding)
- **Y-axis**: Dependent variable (alcohol)
- **Each point**: One state's data

**What to Look For**:
1. **Direction**: Do points trend upward or downward?
2. **Strength**: How close are points to a line?
3. **Outliers**: Any points far from the pattern?

**Interpreting the Pattern**:
```
Positive Correlation:
  ●
    ●
      ●
        ●  (Points go up-right)

Negative Correlation:
        ●
      ●
    ●
  ●  (Points go down-right)

No Correlation:
  ●   ●
    ●     ●
  ●   ●  (Random scatter)
```

---

## Method 1: Pandas Correlation

### Correlation Matrix

```python
df.corr()
```

### What is a Correlation Matrix?

**Definition**: A table showing correlations between all pairs of numerical columns.

**Example Output**:
```
              total  speeding  alcohol  ins_premium
total         1.000     0.830    0.199        0.624
speeding      0.830     1.000    0.286        0.466
alcohol       0.199     0.286    1.000        0.368
ins_premium   0.624     0.466    0.368        1.000
```

### Reading the Matrix

**Diagonal Values** (1.000):
- Each variable correlated with itself
- Always equals 1.0
- Not useful for analysis

**Off-Diagonal Values**:
- Correlation between different variables
- Range: -1 to +1
- Symmetric (same above and below diagonal)

**Example Interpretation**:
- `speeding` vs `total`: 0.830 (strong positive)
- `alcohol` vs `total`: 0.199 (weak positive)
- `speeding` vs `alcohol`: 0.286 (weak positive)

### Advantages of Pandas `.corr()`

✅ **Pros**:
- Calculates all correlations at once
- Easy to visualize with heatmap
- Works directly with DataFrames
- Handles missing values automatically

❌ **Cons**:
- Doesn't provide p-values
- Less control over calculation method
- Can be overwhelming with many columns

### Other Correlation Methods

```python
# Pearson (default)
df.corr(method='pearson')

# Spearman (rank-based)
df.corr(method='spearman')

# Kendall (ordinal)
df.corr(method='kendall')
```

---

## Method 2: SciPy Correlation

### Basic Usage

```python
scipy.stats.pearsonr(df['speeding'], df['alcohol'])[0]
```

### Understanding the Syntax

**Breaking it down**:
```python
result = scipy.stats.pearsonr(df['speeding'], df['alcohol'])
correlation = result[0]  # Correlation coefficient
p_value = result[1]      # Statistical significance
```

**What `[0]` does**:
- `pearsonr()` returns a tuple: (correlation, p-value)
- `[0]` extracts just the correlation coefficient
- `[1]` would extract the p-value

### Complete Example

```python
# Get both correlation and p-value
corr, p_value = scipy.stats.pearsonr(df['speeding'], df['alcohol'])

print(f"Correlation: {corr:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < 0.05:
    print("Statistically significant!")
else:
    print("Not statistically significant")
```

### Advantages of SciPy

✅ **Pros**:
- Provides p-value (statistical significance)
- More precise control
- Standard in scientific computing
- Well-documented

❌ **Cons**:
- Only calculates one pair at a time
- Requires extracting from tuple
- More verbose for multiple correlations

---

## Exploring Different Feature Pairs

### Example 2: Insurance Premium vs Alcohol

```python
sns.scatterplot(df, x='ins_premium', y='alcohol')
```

**What to observe**:
- Is there a visible pattern?
- Positive or negative trend?
- Strong or weak relationship?

### Calculate Correlation

```python
scipy.stats.pearsonr(df['ins_premium'], df['alcohol'])[0]
```

**Interpretation**:
- If positive: Higher premiums associated with more alcohol-related crashes
- If negative: Higher premiums associated with fewer alcohol-related crashes
- If near zero: No clear relationship

---

## Important Property: Symmetry

### Correlation is Symmetric

```python
# These give the SAME result:
scipy.stats.pearsonr(df['alcohol'], df['ins_premium'])[0]
scipy.stats.pearsonr(df['ins_premium'], df['alcohol'])[0]
```

**Why?**
- Correlation measures mutual relationship
- Order doesn't matter
- corr(X, Y) = corr(Y, X)

**Mathematical Proof**:
```
Correlation formula is symmetric:
r = Σ[(x-x̄)(y-ȳ)] / √[Σ(x-x̄)² × Σ(y-ȳ)²]

Swapping x and y gives same result!
```

---

## Method 3: NumPy Correlation

### Using `np.corrcoef()`

```python
# Calculate correlation matrix for two variables
corr_matrix = np.corrcoef(df['speeding'], df['alcohol'])
print(corr_matrix)
```

**Output**:
```
[[1.         0.28612345]
 [0.28612345 1.        ]]
```

**Reading the Output**:
- `[0, 0]`: speeding vs speeding = 1.0
- `[0, 1]`: speeding vs alcohol = 0.286
- `[1, 0]`: alcohol vs speeding = 0.286
- `[1, 1]`: alcohol vs alcohol = 1.0

### Extracting the Correlation

```python
correlation = np.corrcoef(df['speeding'], df['alcohol'])[0, 1]
print(f"Correlation: {correlation:.3f}")
```

### Advantages of NumPy

✅ **Pros**:
- Fast computation
- Works with arrays
- Part of core scientific stack
- No external dependencies

❌ **Cons**:
- No p-value
- Less intuitive output format
- Requires indexing to extract value

---

## Interpreting Correlation Values

### Correlation Coefficient Range

**Scale**: -1.0 to +1.0

```
-1.0 ←―――――――― 0 ―――――――→ +1.0
Perfect      No      Perfect
Negative  Correlation  Positive
```

### Strength Guidelines

| Value Range | Interpretation | Example |
|-------------|----------------|---------|
| 0.9 to 1.0 | Very Strong Positive | Height & Weight |
| 0.7 to 0.9 | Strong Positive | Study Time & Grades |
| 0.5 to 0.7 | Moderate Positive | Exercise & Health |
| 0.3 to 0.5 | Weak Positive | Coffee & Productivity |
| 0.0 to 0.3 | Very Weak/None | Shoe Size & IQ |
| -0.3 to 0.0 | Very Weak Negative | - |
| -0.5 to -0.3 | Weak Negative | Price & Demand |
| -0.7 to -0.5 | Moderate Negative | Speed & Safety |
| -0.9 to -0.7 | Strong Negative | Smoking & Lung Health |
| -1.0 to -0.9 | Very Strong Negative | Distance & Signal Strength |

### Visual Examples

**r = 0.9** (Strong Positive):
```
    ●
      ●
        ●
          ●
            ●
```

**r = 0.3** (Weak Positive):
```
  ●   ●
    ●     ●
  ●   ●
      ●
```

**r = -0.8** (Strong Negative):
```
          ●
        ●
      ●
    ●
  ●
```

---

## Correlation vs Causation

### Critical Distinction

**Correlation**: Two variables change together
**Causation**: One variable CAUSES the other to change

### Classic Examples

#### Example 1: Ice Cream and Drowning
- **Correlation**: Both increase in summer
- **Causation**: Ice cream doesn't cause drowning!
- **Real cause**: Hot weather (confounding variable)

#### Example 2: Shoe Size and Reading Ability
- **Correlation**: Positive in children
- **Causation**: Big feet don't make you smarter!
- **Real cause**: Age (confounding variable)

### The Golden Rule

```
┌─────────────────────────────────────┐
│  CORRELATION ≠ CAUSATION            │
│                                     │
│  Just because two things are        │
│  correlated doesn't mean one        │
│  causes the other!                  │
└─────────────────────────────────────┘
```

---

## Practical Applications

### When to Use Correlation

✅ **Good Uses**:
1. **Exploratory Data Analysis**: Find relationships
2. **Feature Selection**: Identify relevant variables
3. **Hypothesis Generation**: Suggest areas for research
4. **Data Validation**: Check for expected relationships

❌ **Bad Uses**:
1. **Proving Causation**: Need experiments for that
2. **Non-linear Relationships**: Pearson only measures linear
3. **Categorical Data**: Need different methods
4. **Small Samples**: Results may be unreliable

### Real-World Scenarios

**Finance**:
- Stock prices correlation
- Portfolio diversification
- Risk assessment

**Healthcare**:
- Symptom relationships
- Treatment effectiveness
- Risk factor identification

**Marketing**:
- Customer behavior patterns
- Product recommendations
- Campaign effectiveness

**Education**:
- Study habits vs performance
- Attendance vs grades
- Resource allocation

---

## Mermaid Diagram: Correlation Analysis Workflow

```mermaid
graph TD
    A[Start: Two Variables] --> B{Check Data Type}
    B -->|Numerical| C[Create Scatter Plot]
    B -->|Categorical| D[Use Different Method]
    
    C --> E{Visual Pattern?}
    E -->|Linear| F[Calculate Pearson Correlation]
    E -->|Non-linear| G[Consider Other Methods]
    E -->|No Pattern| H[Correlation ≈ 0]
    
    F --> I{Choose Library}
    I -->|Multiple Pairs| J[Use pandas.corr]
    I -->|Single Pair + P-value| K[Use scipy.pearsonr]
    I -->|Arrays Only| L[Use np.corrcoef]
    
    J --> M[Interpret Results]
    K --> M
    L --> M
    
    M --> N{|r| > 0.7?}
    N -->|Yes| O[Strong Correlation]
    N -->|No| P{|r| > 0.3?}
    P -->|Yes| Q[Moderate Correlation]
    P -->|No| R[Weak/No Correlation]
    
    O --> S[Check for Causation]
    Q --> S
    R --> T[No Relationship]
    
    S --> U[Design Experiment]
    
    style F fill:#ccffcc
    style M fill:#ffffcc
    style S fill:#ffcccc
```

---

## Common Mistakes and Solutions

### Mistake 1: Assuming Causation

❌ **Wrong**:
```python
corr = df['speeding'].corr(df['alcohol'])
print(f"Speeding causes {corr*100:.1f}% of alcohol-related crashes")
```

✅ **Correct**:
```python
corr = df['speeding'].corr(df['alcohol'])
print(f"Speeding and alcohol have a correlation of {corr:.3f}")
print("This suggests a relationship, but doesn't prove causation")
```

### Mistake 2: Ignoring Non-linearity

❌ **Wrong**: Using Pearson for curved relationships

✅ **Correct**: Check scatter plot first, consider Spearman for non-linear

### Mistake 3: Small Sample Size

❌ **Wrong**: Trusting correlation from 5 data points

✅ **Correct**: Need at least 30 points for reliable results

### Mistake 4: Outliers

❌ **Wrong**: Ignoring extreme values

✅ **Correct**: Check for and handle outliers appropriately

---

## Practice Exercises

### Exercise 1: Calculate Correlations
Using the car_crashes dataset:
```python
# 1. Find correlation between 'total' and 'speeding'
# 2. Find correlation between 'ins_premium' and 'ins_losses'
# 3. Which pair has stronger correlation?
```

### Exercise 2: Interpret Results
```python
corr1 = 0.85
corr2 = -0.65
corr3 = 0.15

# Describe the strength and direction of each
```

### Exercise 3: Create Correlation Heatmap
```python
# Use seaborn to create a heatmap of all correlations
# Hint: sns.heatmap(df.corr(), annot=True)
```

### Exercise 4: Real-World Application
Think of three pairs of variables:
1. One with positive correlation
2. One with negative correlation
3. One with no correlation

Explain why each relationship exists.

---

## Advanced Topics

### 1. Correlation Heatmap

```python
import matplotlib.pyplot as plt

# Create correlation matrix
corr_matrix = df.corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

**Benefits**:
- Visualize all correlations at once
- Color coding shows strength
- Easy to spot patterns

### 2. P-value Interpretation

```python
corr, p_value = scipy.stats.pearsonr(df['speeding'], df['alcohol'])

if p_value < 0.05:
    print(f"Correlation of {corr:.3f} is statistically significant")
else:
    print(f"Correlation of {corr:.3f} is NOT statistically significant")
```

**What is P-value?**
- Probability that correlation occurred by chance
- p < 0.05: Statistically significant (95% confidence)
- p < 0.01: Highly significant (99% confidence)

### 3. Partial Correlation

**Definition**: Correlation between two variables while controlling for others

**Use Case**: Remove confounding variable effects

### 4. Spearman vs Pearson

**Pearson**: Measures linear relationships
**Spearman**: Measures monotonic relationships (can be non-linear)

```python
# Pearson
pearson_corr = df['x'].corr(df['y'], method='pearson')

# Spearman
spearman_corr = df['x'].corr(df['y'], method='spearman')
```

---

## Summary

### Key Takeaways

1. **Correlation measures relationship strength and direction**
2. **Range: -1 (perfect negative) to +1 (perfect positive)**
3. **Three main libraries: Pandas, SciPy, NumPy**
4. **Always visualize with scatter plots first**
5. **Correlation ≠ Causation!**

### Quick Reference

```python
# Pandas - Multiple correlations
df.corr()

# SciPy - Single pair with p-value
corr, p_val = scipy.stats.pearsonr(df['x'], df['y'])

# NumPy - Array-based
corr = np.corrcoef(df['x'], df['y'])[0, 1]

# Visualization
sns.scatterplot(df, x='var1', y='var2')
sns.heatmap(df.corr(), annot=True)
```

### Decision Tree: Which Method?

```
Need p-value? → Use SciPy
Multiple pairs? → Use Pandas
Working with arrays? → Use NumPy
Just exploring? → Use Pandas + Heatmap
```

---

## Next Steps

1. **Practice**: Calculate correlations on different datasets
2. **Visualize**: Create scatter plots and heatmaps
3. **Interpret**: Understand what correlations mean
4. **Experiment**: Try different correlation methods
5. **Apply**: Use in real data analysis projects

---

## Additional Resources

- **Khan Academy**: Correlation and Regression
- **StatQuest**: Correlation Clearly Explained
- **Seaborn Gallery**: Correlation Plot Examples
- **SciPy Documentation**: Statistical Functions
- **Real Datasets**: Kaggle, UCI Machine Learning Repository

