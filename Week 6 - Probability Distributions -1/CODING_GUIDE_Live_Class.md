# Week 6: Probability Distributions 1 - Live Class Coding Guide

## 📓 Notebook: Copy_of_PD1_Code.ipynb

## 🎯 Objective
Learn discrete probability distributions through a practical hospital operations case study. Understand how to model real-world scenarios using Bernoulli, Binomial, Poisson, and Discrete Uniform distributions.

---

## 📊 Case Study: Hospital Operations Optimization

**Business Context:**
A hospital wants to optimize operations and improve patient care using data-driven decisions. Different aspects of hospital operations can be modeled using discrete probability distributions.

**Key Questions:**
- Will patients show up for appointments?
- How many successful surgeries can we expect?
- How many emergency arrivals per hour?
- How to allocate resources efficiently?

---

## 🔧 Section 1: Setup and Imports

### Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, randint
```

**Libraries Explained:**
- `numpy`: Numerical operations and array handling
- `matplotlib.pyplot`: Data visualization
- `scipy.stats`: Statistical distributions and functions
  - `bernoulli`: Binary outcome distribution
  - `binom`: Binomial distribution
  - `poisson`: Poisson distribution
  - `randint`: Discrete uniform distribution

---

## 🔧 Section 2: Bernoulli Distribution

### 2.1 Concept
**What is Bernoulli Distribution?**
- Models a single trial with two possible outcomes
- Success (1) or Failure (0)
- Single parameter: p (probability of success)

**Real-World Example:**
- Coin flip: Heads (1) or Tails (0)
- Patient shows up (1) or doesn't (0)
- Surgery succeeds (1) or fails (0)

**Mathematical Notation:**
```
X ~ Bernoulli(p)
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p
Variance = p(1-p)
```

### 2.2 Hospital Scenario: Patient Appointments
**Problem:** Will patients show up for their appointments?

**Given:**
- Historical data shows 80% show-up rate
- Need to predict no-shows for scheduling

```python
# Parameters
p_bernoulli = 0.8  # Probability of showing up

# Generate 1000 random samples
bernoulli_trials = bernoulli.rvs(p_bernoulli, size=1000)

print(type(bernoulli_trials))  # numpy.ndarray
print(bernoulli_trials)  # Array of 0s and 1s
```

**Code Breakdown:**
- `bernoulli.rvs()`: Random Variates (samples)
  - `p_bernoulli`: Success probability (0.8)
  - `size=1000`: Generate 1000 samples
- Returns: NumPy array of 0s (no-show) and 1s (show-up)

**Expected Output:**
- Approximately 800 ones (show-ups)
- Approximately 200 zeros (no-shows)

### 2.3 Visualize Bernoulli Distribution
```python
plt.figure(figsize=(8, 6))
plt.hist(bernoulli_trials, bins=2, density=True, alpha=0.6, 
         color='g', edgecolor='black')
plt.title('Bernoulli Distribution (p=0.8)')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.xticks([0, 1])
plt.show()
```

**Visualization Parameters:**
- `bins=2`: Two bars (0 and 1)
- `density=True`: Show probabilities (not counts)
- `alpha=0.6`: Transparency
- `edgecolor='black'`: Bar borders

**Interpretation:**
- Bar at 0: ~20% height (no-shows)
- Bar at 1: ~80% height (show-ups)
- Confirms 80% show-up rate

### 2.4 Calculate Statistics
```python
print(f"Bernoulli Distribution: p={p_bernoulli}")
print(f"Mean of Bernoulli trials: {np.mean(bernoulli_trials)}")
```

**Expected Results:**
- Theoretical mean: 0.8
- Observed mean: ~0.765-0.800 (close to 0.8)
- Small deviation is normal with finite samples

**Business Insight:**
- Can predict ~800 out of 1000 patients will show up
- Schedule accordingly to minimize wasted slots
- Consider overbooking by ~20% to account for no-shows

---

## 🔧 Section 3: Binomial Distribution

### 3.1 Concept
**What is Binomial Distribution?**
- Models number of successes in n independent trials
- Each trial is a Bernoulli trial
- Two parameters: n (trials), p (success probability)

**Real-World Examples:**
- Number of heads in 10 coin flips
- Number of successful surgeries out of 20
- Number of patients who show up out of 50 appointments

**Mathematical Notation:**
```
X ~ Binomial(n, p)
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
Mean = n × p
Variance = n × p × (1-p)
```

**Relationship to Bernoulli:**
- Binomial = Sum of n Bernoulli trials
- If n=1, Binomial = Bernoulli

### 3.2 Hospital Scenario: Surgery Success Rate
**Problem:** Track successful surgeries out of total operations

**Given:**
- 20 surgeries performed
- 90% success rate historically

```python
# Parameters
n_binomial = 20  # Number of surgeries
p_binomial = 0.9  # Success probability

# Generate samples
binomial_trials = binom.rvs(n_binomial, p_binomial, size=1000)

print(binomial_trials)
```

**Code Breakdown:**
- `binom.rvs()`: Generate random samples
  - `n_binomial=20`: 20 trials (surgeries)
  - `p_binomial=0.9`: 90% success rate
  - `size=1000`: 1000 experiments
- Returns: Array of success counts (0 to 20)

**Expected Output:**
- Most values around 18 (20 × 0.9)
- Range: typically 15-20
- Rare to see < 15 or = 20

### 3.3 Visualize Binomial Distribution
```python
plt.figure(figsize=(10, 6))
plt.hist(binomial_trials, bins=range(0, n_binomial+2), density=True,
         alpha=0.6, color='b', edgecolor='black')
plt.title(f'Binomial Distribution (n={n_binomial}, p={p_binomial})')
plt.xlabel('Number of Successful Surgeries')
plt.ylabel('Probability')
plt.show()
```

**Visualization Features:**
- `bins=range(0, n_binomial+2)`: One bin per possible value
- Bell-shaped curve centered around 18
- Shows probability of each outcome

**Interpretation:**
- Peak at 18: Most likely outcome
- Symmetric around mean (approximately)
- Tails show rare outcomes

### 3.4 Calculate Statistics
```python
print(f"Binomial Distribution: n={n_binomial}, p={p_binomial}")
print(f"Mean of Binomial trials: {np.mean(binomial_trials)}")
print(f"Expected mean: {n_binomial * p_binomial}")
```

**Expected Results:**
- Theoretical mean: 20 × 0.9 = 18
- Observed mean: ~17.8-18.2
- Close match confirms correct modeling

**Business Insights:**
- Expect ~18 successful surgeries out of 20
- If actual < 16, investigate issues
- If actual > 19, celebrate excellence
- Use for resource planning and quality control

---

## 🔧 Section 4: Poisson Distribution

### 4.1 Concept
**What is Poisson Distribution?**
- Models number of events in fixed time/space interval
- Events occur independently
- Single parameter: λ (lambda) = average rate

**Real-World Examples:**
- Number of customers arriving per hour
- Number of emails received per day
- Number of emergency patients per hour
- Number of defects per product

**Mathematical Notation:**
```
X ~ Poisson(λ)
P(X = k) = (λ^k × e^(-λ)) / k!
Mean = λ
Variance = λ
```

**Key Properties:**
- Mean = Variance = λ
- Discrete values: 0, 1, 2, 3, ...
- No upper limit (theoretically)

### 4.2 Hospital Scenario: Emergency Arrivals
**Problem:** How many emergency patients arrive per hour?

**Given:**
- Average 5 emergency patients per hour
- Need to staff appropriately

```python
# Parameters
lambda_poisson = 5  # Average arrivals per hour

# Generate samples
poisson_trials = poisson.rvs(lambda_poisson, size=1000)

print(poisson_trials)
```

**Code Breakdown:**
- `poisson.rvs()`: Generate random samples
  - `lambda_poisson=5`: Average rate
  - `size=1000`: 1000 hours simulated
- Returns: Array of arrival counts

**Expected Output:**
- Most values around 5
- Range: typically 0-12
- Occasional higher values possible

### 4.3 Visualize Poisson Distribution
```python
plt.figure(figsize=(10, 6))
plt.hist(poisson_trials, bins=range(0, max(poisson_trials)+2),
         density=True, alpha=0.6, color='r', edgecolor='black')
plt.title(f'Poisson Distribution (λ={lambda_poisson})')
plt.xlabel('Number of Emergency Arrivals')
plt.ylabel('Probability')
plt.show()
```

**Visualization Features:**
- Right-skewed distribution
- Peak at or near λ=5
- Long right tail (rare high values)

**Interpretation:**
- Most hours: 3-7 arrivals
- Rare hours: 0-2 or >10 arrivals
- Helps plan staffing levels

### 4.4 Calculate Statistics
```python
print(f"Poisson Distribution: λ={lambda_poisson}")
print(f"Mean of Poisson trials: {np.mean(poisson_trials)}")
print(f"Variance of Poisson trials: {np.var(poisson_trials)}")
```

**Expected Results:**
- Theoretical mean: 5
- Observed mean: ~4.9-5.1
- Variance ≈ Mean (Poisson property)

**Business Insights:**
- Staff for 5 patients/hour on average
- Have surge capacity for 8-10 patients
- Monitor if actual rate changes
- Adjust staffing during peak hours

---

## 🔧 Section 5: Discrete Uniform Distribution

### 5.1 Concept
**What is Discrete Uniform Distribution?**
- All outcomes equally likely
- Defined by range: [a, b]
- Each value has probability 1/(b-a+1)

**Real-World Examples:**
- Rolling a fair die (1-6)
- Random room assignment (1-100)
- Random patient selection
- Lottery numbers

**Mathematical Notation:**
```
X ~ DiscreteUniform(a, b)
P(X = k) = 1/(b-a+1) for a ≤ k ≤ b
Mean = (a + b) / 2
Variance = ((b-a+1)² - 1) / 12
```

### 5.2 Hospital Scenario: Random Room Assignment
**Problem:** Assign patients to rooms randomly

**Given:**
- 50 rooms available (1-50)
- Each room equally likely

```python
# Parameters
low = 1
high = 50

# Generate samples
uniform_trials = randint.rvs(low, high+1, size=1000)

print(uniform_trials)
```

**Code Breakdown:**
- `randint.rvs()`: Generate random integers
  - `low=1`: Minimum value
  - `high+1=51`: Maximum (exclusive)
  - `size=1000`: 1000 assignments
- Returns: Array of room numbers (1-50)

**Important:** `high` parameter is exclusive, so use `high+1`

### 5.3 Visualize Discrete Uniform Distribution
```python
plt.figure(figsize=(12, 6))
plt.hist(uniform_trials, bins=range(low, high+2), density=True,
         alpha=0.6, color='purple', edgecolor='black')
plt.title(f'Discrete Uniform Distribution ({low} to {high})')
plt.xlabel('Room Number')
plt.ylabel('Probability')
plt.show()
```

**Visualization Features:**
- Flat distribution (all bars same height)
- Each room: ~2% probability (1/50)
- No peaks or valleys

**Interpretation:**
- All rooms equally likely
- Fair assignment process
- No bias toward any room

### 5.4 Calculate Statistics
```python
print(f"Discrete Uniform Distribution: [{low}, {high}]")
print(f"Mean of Uniform trials: {np.mean(uniform_trials)}")
print(f"Expected mean: {(low + high) / 2}")
```

**Expected Results:**
- Theoretical mean: (1+50)/2 = 25.5
- Observed mean: ~25-26
- Confirms uniform distribution

**Business Insights:**
- Fair room allocation
- No room overused or underused
- Can track actual vs expected usage
- Identify if certain rooms preferred/avoided

---

## 💡 Key Takeaways

### Distribution Comparison

| Distribution | Use Case | Parameters | Example |
|--------------|----------|------------|---------|
| **Bernoulli** | Single binary trial | p | Patient shows up? |
| **Binomial** | Multiple binary trials | n, p | Successful surgeries out of 20 |
| **Poisson** | Events in time/space | λ | Emergency arrivals per hour |
| **Discrete Uniform** | Equally likely outcomes | a, b | Random room assignment |

### When to Use Each Distribution

**Bernoulli:**
- Single yes/no question
- One trial only
- Example: Will this patient show up?

**Binomial:**
- Multiple independent trials
- Same success probability
- Example: How many out of n patients show up?

**Poisson:**
- Counting events over time/space
- Events occur independently
- Example: How many emergencies this hour?

**Discrete Uniform:**
- All outcomes equally likely
- Finite number of outcomes
- Example: Random selection from list

### Python Functions Summary

```python
# Generate samples
bernoulli.rvs(p, size=n)
binom.rvs(n, p, size=samples)
poisson.rvs(lambda, size=n)
randint.rvs(low, high, size=n)

# Calculate probabilities
bernoulli.pmf(k, p)
binom.pmf(k, n, p)
poisson.pmf(k, lambda)
randint.pmf(k, low, high)

# Calculate cumulative probabilities
bernoulli.cdf(k, p)
binom.cdf(k, n, p)
poisson.cdf(k, lambda)
randint.cdf(k, low, high)
```

---

## 🚨 Common Mistakes to Avoid

1. **Confusing Bernoulli and Binomial**
   - Bernoulli: Single trial
   - Binomial: Multiple trials

2. **Wrong Poisson Parameter**
   - λ is the RATE (average), not total count
   - Must be for specific time/space interval

3. **Discrete Uniform Range**
   - `randint.rvs(low, high)` → high is EXCLUSIVE
   - Use `high+1` to include high value

4. **Misinterpreting Histograms**
   - `density=True`: Shows probabilities
   - `density=False`: Shows counts
   - Choose based on what you want to see

5. **Sample Size Too Small**
   - Need large samples (>1000) for accurate distribution
   - Small samples show more variation

---

## 🎯 Practice Exercises

1. **Modify Bernoulli:**
   - Change p to 0.5, 0.3, 0.95
   - Observe how distribution changes

2. **Binomial Experiments:**
   - Try n=10, p=0.5 (coin flips)
   - Try n=100, p=0.01 (rare events)

3. **Poisson Scenarios:**
   - Model website visits per minute
   - Model defects per product batch

4. **Uniform Applications:**
   - Simulate dice rolls (1-6)
   - Random lottery number selection

5. **Real Data:**
   - Collect your own data
   - Fit appropriate distribution
   - Compare theoretical vs observed

---

*This coding guide explains the discrete probability distributions covered in the live class notebook. Practice with different parameters to build intuition!*
