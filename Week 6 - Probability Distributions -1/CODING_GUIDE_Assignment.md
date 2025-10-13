# Week 6: Probability Distributions 1 - Assignment Coding Guide

## 📓 Notebook: PD1_Assignment_Solution.ipynb

## 🎯 Objective
Apply discrete probability distributions to solve real-world business problems. Practice calculating probabilities, visualizing distributions, and interpreting results.

---

## 📊 Problem 1: Call Center Analysis

### Business Context
**Scenario:** Analyzing customer service calls at a call center
- Average: 5 calls per hour
- Distribution: Poisson (events over time)
- Goal: Calculate probability of exactly 3 calls in next hour

**Why Poisson?**
- Events occur independently
- Average rate is known (5 calls/hour)
- Counting events in fixed time interval
- Perfect fit for call center arrivals

### Solution Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameters
lambda_poisson = 5  # Average calls per hour

# Calculate probability of exactly 3 calls
prob_3_calls = poisson.pmf(3, lambda_poisson)
print(f"Probability of exactly 3 calls: {prob_3_calls:.4f}")
```

**Code Breakdown:**
- `poisson.pmf(k, lambda)`: Probability Mass Function
  - `k=3`: Specific value we want
  - `lambda=5`: Average rate
- Returns: Probability of exactly k events

**Expected Output:**
```
Probability of exactly 3 calls: 0.1404
```

**Interpretation:**
- 14.04% chance of exactly 3 calls
- Relatively low probability
- Most likely: 4-6 calls (around mean)

### Visualize PMF (Probability Mass Function)

```python
# Generate range of possible values
x = np.arange(0, 15)

# Calculate probabilities for each value
pmf_values = poisson.pmf(x, lambda_poisson)

# Plot PMF
plt.figure(figsize=(10, 6))
plt.bar(x, pmf_values, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=3, color='red', linestyle='--', label='k=3')
plt.title('Poisson PMF (λ=5)')
plt.xlabel('Number of Calls')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Visualization Features:**
- Bar chart showing all probabilities
- Red line at k=3 (our target)
- Peak at λ=5 (most likely value)
- Symmetric-ish distribution

**Business Insights:**
- Most hours: 3-7 calls
- Rare: 0-2 or >10 calls
- Staff for average + buffer
- Monitor if actual differs from expected

### Visualize CDF (Cumulative Distribution Function)

```python
# Calculate cumulative probabilities
cdf_values = poisson.cdf(x, lambda_poisson)

# Plot CDF
plt.figure(figsize=(10, 6))
plt.plot(x, cdf_values, marker='o', color='green')
plt.axhline(y=poisson.cdf(3, lambda_poisson), color='red', 
            linestyle='--', label=f'CDF at k=3: {poisson.cdf(3, lambda_poisson):.4f}')
plt.title('Poisson CDF (λ=5)')
plt.xlabel('Number of Calls')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**CDF Interpretation:**
- Shows P(X ≤ k)
- At k=3: ~26.5% chance of ≤3 calls
- At k=5: ~61.6% chance of ≤5 calls
- Useful for "at most" questions

**Key Difference:**
- **PMF:** P(X = k) - exactly k
- **CDF:** P(X ≤ k) - at most k

---

## 📊 Problem 2: Quality Control

### Business Context
**Scenario:** Manufacturing defects in production
- Batch size: 100 items
- Defect rate: 2% (p=0.02)
- Distribution: Binomial
- Goal: Probability of exactly 3 defects

**Why Binomial?**
- Fixed number of trials (n=100)
- Each item: defective or not (binary)
- Independent trials
- Constant probability (p=0.02)

### Solution Code

```python
from scipy.stats import binom

# Parameters
n = 100  # Batch size
p = 0.02  # Defect probability

# Calculate probability of exactly 3 defects
prob_3_defects = binom.pmf(3, n, p)
print(f"Probability of exactly 3 defects: {prob_3_defects:.4f}")
```

**Expected Output:**
```
Probability of exactly 3 defects: 0.1823
```

**Interpretation:**
- 18.23% chance of exactly 3 defects
- Expected defects: n × p = 100 × 0.02 = 2
- 3 is close to expected, so relatively likely

### Visualize Distribution

```python
# Generate range
x = np.arange(0, 10)

# Calculate PMF
pmf_values = binom.pmf(x, n, p)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(x, pmf_values, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(x=3, color='red', linestyle='--', label='k=3')
plt.title(f'Binomial PMF (n={n}, p={p})')
plt.xlabel('Number of Defects')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Business Insights:**
- Most batches: 0-5 defects
- Peak at 2 (expected value)
- If actual >> 2, investigate process
- Use for quality control thresholds

### Calculate Cumulative Probability

```python
# Probability of at most 3 defects
prob_at_most_3 = binom.cdf(3, n, p)
print(f"P(X ≤ 3) = {prob_at_most_3:.4f}")

# Probability of more than 3 defects
prob_more_than_3 = 1 - prob_at_most_3
print(f"P(X > 3) = {prob_more_than_3:.4f}")
```

**Use Cases:**
- P(X ≤ 3): Acceptable quality level
- P(X > 3): Reject batch threshold
- Set quality standards based on probabilities

---

## 📊 Problem 3: Customer Arrivals

### Business Context
**Scenario:** Store customer arrivals
- Time window: 30 minutes
- Average: 8 customers per 30 min
- Distribution: Poisson
- Goal: Probability of at least 10 customers

**Why Poisson?**
- Counting arrivals over time
- Independent events
- Known average rate
- No upper limit

### Solution Code

```python
# Parameters
lambda_arrivals = 8  # Average per 30 min

# Probability of at least 10 customers
# P(X ≥ 10) = 1 - P(X ≤ 9)
prob_at_least_10 = 1 - poisson.cdf(9, lambda_arrivals)
print(f"P(X ≥ 10) = {prob_at_least_10:.4f}")

# Alternative: sum of individual probabilities
prob_alt = sum(poisson.pmf(k, lambda_arrivals) for k in range(10, 25))
print(f"Alternative calculation: {prob_alt:.4f}")
```

**Expected Output:**
```
P(X ≥ 10) = 0.2834
Alternative calculation: 0.2834
```

**Interpretation:**
- 28.34% chance of ≥10 customers
- Fairly common occurrence
- Need capacity for surge

### Visualize with Threshold

```python
# Generate range
x = np.arange(0, 20)
pmf_values = poisson.pmf(x, lambda_arrivals)

# Plot with threshold
plt.figure(figsize=(10, 6))
colors = ['red' if k >= 10 else 'blue' for k in x]
plt.bar(x, pmf_values, alpha=0.7, color=colors, edgecolor='black')
plt.axvline(x=10, color='black', linestyle='--', label='Threshold: 10')
plt.title(f'Poisson PMF (λ={lambda_arrivals}) - At Least 10')
plt.xlabel('Number of Customers')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Business Insights:**
- Red bars: ≥10 customers (28.34%)
- Blue bars: <10 customers (71.66%)
- Staff for average + handle surges
- Monitor peak times

---

## 📊 Problem 4: Dice Rolling

### Business Context
**Scenario:** Fair six-sided die
- Outcomes: 1, 2, 3, 4, 5, 6
- Distribution: Discrete Uniform
- Goal: Probability of rolling 4 or higher

**Why Discrete Uniform?**
- All outcomes equally likely
- Finite number of outcomes
- Each has probability 1/6

### Solution Code

```python
from scipy.stats import randint

# Parameters (note: high is exclusive)
low = 1
high = 7  # 6+1 because exclusive

# Probability of rolling 4 or higher
# P(X ≥ 4) = P(X=4) + P(X=5) + P(X=6)
prob_4_or_higher = sum(randint.pmf(k, low, high) for k in range(4, 7))
print(f"P(X ≥ 4) = {prob_4_or_higher:.4f}")

# Alternative: using CDF
prob_alt = 1 - randint.cdf(3, low, high)
print(f"Alternative: {prob_alt:.4f}")
```

**Expected Output:**
```
P(X ≥ 4) = 0.5000
Alternative: 0.5000
```

**Interpretation:**
- 50% chance of rolling ≥4
- Makes sense: 3 favorable outcomes out of 6
- 3/6 = 0.5

### Visualize Uniform Distribution

```python
# Generate outcomes
x = np.arange(1, 7)
pmf_values = [randint.pmf(k, low, high) for k in x]

# Plot
plt.figure(figsize=(10, 6))
colors = ['green' if k >= 4 else 'gray' for k in x]
plt.bar(x, pmf_values, alpha=0.7, color=colors, edgecolor='black')
plt.axvline(x=3.5, color='red', linestyle='--', label='Threshold: 4')
plt.title('Discrete Uniform Distribution (Fair Die)')
plt.xlabel('Die Outcome')
plt.ylabel('Probability')
plt.ylim(0, 0.25)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Visualization Features:**
- All bars same height (1/6 ≈ 0.167)
- Green: favorable outcomes (≥4)
- Gray: unfavorable outcomes (<4)
- Perfectly flat distribution

---

## 💡 Key Concepts Summary

### Distribution Selection Guide

| Problem Type | Distribution | Key Indicator |
|--------------|--------------|---------------|
| Events over time | Poisson | "per hour", "per day" |
| Success/failure trials | Binomial | "out of n", "defect rate" |
| Equally likely outcomes | Discrete Uniform | "fair die", "random selection" |
| Single yes/no | Bernoulli | "will it happen?" |

### Probability Functions

**PMF (Probability Mass Function):**
```python
poisson.pmf(k, lambda)  # P(X = k)
binom.pmf(k, n, p)      # P(X = k)
randint.pmf(k, low, high)  # P(X = k)
```
- Returns probability of exactly k
- Use for "exactly" questions

**CDF (Cumulative Distribution Function):**
```python
poisson.cdf(k, lambda)  # P(X ≤ k)
binom.cdf(k, n, p)      # P(X ≤ k)
randint.cdf(k, low, high)  # P(X ≤ k)
```
- Returns probability of at most k
- Use for "at most" or "less than" questions

**Complement Rule:**
```python
# P(X > k) = 1 - P(X ≤ k)
prob_more_than_k = 1 - dist.cdf(k, params)

# P(X ≥ k) = 1 - P(X ≤ k-1)
prob_at_least_k = 1 - dist.cdf(k-1, params)
```

### Common Calculations

**Expected Value (Mean):**
- Poisson: E[X] = λ
- Binomial: E[X] = n × p
- Discrete Uniform: E[X] = (a + b) / 2

**Variance:**
- Poisson: Var(X) = λ
- Binomial: Var(X) = n × p × (1-p)
- Discrete Uniform: Var(X) = ((b-a+1)² - 1) / 12

---

## 🚨 Common Mistakes

1. **Wrong Distribution Choice**
   - Poisson for fixed trials → Use Binomial
   - Binomial for time-based events → Use Poisson

2. **Inclusive vs Exclusive**
   - `randint.rvs(1, 7)` → 1 to 6 (7 is exclusive)
   - Always add 1 to high value

3. **PMF vs CDF Confusion**
   - PMF: exactly k
   - CDF: at most k
   - For "at least k": use 1 - CDF(k-1)

4. **Forgetting Complement**
   - P(X > k) ≠ CDF(k)
   - P(X > k) = 1 - CDF(k)

5. **Wrong Parameters**
   - Poisson: only λ (rate)
   - Binomial: n (trials) and p (probability)
   - Check parameter order!

---

## 🎯 Practice Extensions

1. **Call Center:**
   - What if λ = 10? How does distribution change?
   - Calculate P(5 ≤ X ≤ 8)
   - Find probability of zero calls

2. **Quality Control:**
   - If p = 0.05, how many defects expected?
   - What batch size for <1% chance of >5 defects?
   - Compare n=50 vs n=200

3. **Customer Arrivals:**
   - Calculate P(X = 8) exactly
   - Find most likely number of arrivals
   - What if time window is 1 hour?

4. **Dice Rolling:**
   - Probability of rolling even number?
   - Two dice: probability sum ≥ 10?
   - Three dice: probability all different?

---

*This coding guide covers the assignment problems with detailed explanations. Practice with different parameters and scenarios to build intuition!*
