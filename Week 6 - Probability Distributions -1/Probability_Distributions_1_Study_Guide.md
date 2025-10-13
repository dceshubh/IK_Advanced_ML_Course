# Probability Distributions - 1 Study Guide

## 🎯 Learning Objectives
Master discrete probability distributions and their applications:
- Bernoulli Distribution
- Binomial Distribution
- Poisson Distribution
- Uniform Distribution
- Real-world applications in data science

---

## 📚 Table of Contents
1. [Introduction to Probability Distributions](#intro)
2. [Bernoulli Distribution](#bernoulli)
3. [Binomial Distribution](#binomial)
4. [Poisson Distribution](#poisson)
5. [Uniform Distribution](#uniform)
6. [Applications](#applications)

---

## 🎲 Introduction {#intro}

### Simple Explanation (Like You're 12)
Imagine you're flipping a coin. Sometimes it's heads, sometimes tails. Probability distributions help us predict how often we'll get heads or tails if we flip many times. Different situations use different "rules" - these are different distributions!

### Technical Definition
A **probability distribution** describes how probabilities are distributed over the values of a random variable.

### Types of Distributions
- **Discrete**: Countable outcomes (coin flips, dice rolls)
- **Continuous**: Infinite possible values (height, weight, temperature)

---

## 🎯 Bernoulli Distribution {#bernoulli}

### Simple Explanation
Like flipping a coin once - you get either heads (success) or tails (failure). That's it, just one try!

### Technical Definition
Models a **single trial** with two possible outcomes: success (1) or failure (0).

### Parameters
- **p**: Probability of success (0 ≤ p ≤ 1)
- **q = 1-p**: Probability of failure

### Formula
```
P(X = 1) = p
P(X = 0) = 1 - p
```

### Real-World Examples
- Patient shows up for appointment (yes/no)
- Email is spam (yes/no)
- Product is defective (yes/no)
- Customer makes a purchase (yes/no)

### Python Implementation
```python
from scipy.stats import bernoulli

p = 0.8  # 80% probability of success
trials = bernoulli.rvs(p, size=1000)  # 1000 trials

# Results: array of 0s and 1s
# Expected: ~800 ones, ~200 zeros
```

### Key Properties
- **Mean (Expected Value)**: μ = p
- **Variance**: σ² = p(1-p)
- **Standard Deviation**: σ = √[p(1-p)]

---

## 🎲 Binomial Distribution {#binomial}

### Simple Explanation
Like flipping a coin 10 times and counting how many heads you get. It's multiple Bernoulli trials!

### Technical Definition
Models the **number of successes** in a **fixed number of independent trials**, each with the same probability of success.

### Parameters
- **n**: Number of trials
- **p**: Probability of success in each trial

### Formula
```
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

Where C(n,k) = n! / (k!(n-k)!)
```

### Real-World Examples
- Number of patients showing up out of 20 appointments
- Number of defective items in a batch of 100
- Number of successful sales calls out of 50 attempts
- Number of heads in 10 coin flips

### Python Implementation
```python
from scipy.stats import binom

n = 20  # Number of trials
p = 0.8  # Probability of success

# Generate samples
samples = binom.rvs(n, p, size=1000)

# Calculate probabilities
prob_15 = binom.pmf(15, n, p)  # P(X = 15)
prob_at_least_15 = 1 - binom.cdf(14, n, p)  # P(X ≥ 15)

# Mean and variance
mean = binom.mean(n, p)  # n * p
variance = binom.var(n, p)  # n * p * (1-p)
```

### Key Properties
- **Mean**: μ = np
- **Variance**: σ² = np(1-p)
- **Standard Deviation**: σ = √[np(1-p)]

### When to Use
- Fixed number of trials
- Each trial is independent
- Only two outcomes per trial
- Probability stays constant

---

## 📊 Poisson Distribution {#poisson}

### Simple Explanation
Counts how many times something happens in a fixed time period. Like counting how many customers enter a store per hour.

### Technical Definition
Models the **number of events** occurring in a **fixed interval** of time or space, given a known average rate.

### Parameters
- **λ (lambda)**: Average rate of occurrence (mean)

### Formula
```
P(X = k) = (λ^k × e^(-λ)) / k!

Where:
- k = number of events
- e ≈ 2.71828
```

### Real-World Examples
- Number of emergency room visits per hour
- Number of emails received per day
- Number of defects per square meter of fabric
- Number of calls to a call center per minute

### Python Implementation
```python
from scipy.stats import poisson

lambda_rate = 5  # Average 5 events per interval

# Generate samples
samples = poisson.rvs(lambda_rate, size=1000)

# Calculate probabilities
prob_3 = poisson.pmf(3, lambda_rate)  # P(X = 3)
prob_at_most_5 = poisson.cdf(5, lambda_rate)  # P(X ≤ 5)

# Mean and variance
mean = poisson.mean(lambda_rate)  # λ
variance = poisson.var(lambda_rate)  # λ
```

### Key Properties
- **Mean**: μ = λ
- **Variance**: σ² = λ
- **Standard Deviation**: σ = √λ

### When to Use
- Events occur independently
- Average rate is constant
- Two events cannot occur simultaneously
- Counting occurrences in an interval

---

## 🎰 Uniform Distribution {#uniform}

### Simple Explanation
Like rolling a fair die - each number (1-6) has an equal chance of appearing.

### Technical Definition
All outcomes in a range have **equal probability**.

### Parameters (Discrete)
- **a**: Minimum value
- **b**: Maximum value

### Formula
```
P(X = k) = 1 / (b - a + 1)

For k in {a, a+1, ..., b}
```

### Real-World Examples
- Rolling a fair die
- Random number generation
- Lottery number selection
- Random sampling

### Python Implementation
```python
from scipy.stats import randint

a = 1  # Minimum
b = 7  # Maximum (exclusive in randint)

# Generate samples
samples = randint.rvs(a, b, size=1000)

# Calculate probability
prob = randint.pmf(3, a, b)  # P(X = 3)

# Mean and variance
mean = randint.mean(a, b)  # (a + b - 1) / 2
variance = randint.var(a, b)
```

### Key Properties
- **Mean**: μ = (a + b) / 2
- **Variance**: σ² = [(b - a + 1)² - 1] / 12

---

## 🏥 Real-World Applications {#applications}

### Hospital Operations Case Study

**1. Appointment No-Shows (Bernoulli)**
- Model: Will patient show up?
- Use: Optimize scheduling, reduce wasted slots
- p = 0.8 means 80% show-up rate

**2. Multiple Appointments (Binomial)**
- Model: How many patients show up out of 20?
- Use: Staff allocation, resource planning
- n = 20 appointments, p = 0.8

**3. Emergency Arrivals (Poisson)**
- Model: Number of ER visits per hour
- Use: Staff scheduling, capacity planning
- λ = 5 patients per hour

**4. Random Assignment (Uniform)**
- Model: Random patient assignment to rooms
- Use: Fair distribution, load balancing

### Business Applications

**E-commerce**
- Bernoulli: Will customer buy?
- Binomial: Sales out of 100 visitors
- Poisson: Orders per hour
- Uniform: Random product recommendations

**Quality Control**
- Bernoulli: Is item defective?
- Binomial: Defects in batch
- Poisson: Defects per unit area
- Uniform: Random sampling

---

## 🔑 Key Takeaways

### Distribution Comparison

| Distribution | Trials | Outcomes | Use Case |
|--------------|--------|----------|----------|
| Bernoulli | 1 | 2 (0 or 1) | Single yes/no event |
| Binomial | n (fixed) | 0 to n | Count successes in n trials |
| Poisson | Unlimited | 0 to ∞ | Count events in interval |
| Uniform | 1 | a to b | Equal probability outcomes |

### When to Use Each

**Bernoulli**: Single trial, two outcomes
**Binomial**: Multiple trials, counting successes
**Poisson**: Counting rare events over time/space
**Uniform**: All outcomes equally likely

### Python Libraries
```python
from scipy.stats import bernoulli, binom, poisson, randint
import numpy as np
import matplotlib.pyplot as plt
```

---

## 💡 Common Mistakes

1. **Confusing Bernoulli and Binomial**: Bernoulli is ONE trial, Binomial is MULTIPLE
2. **Poisson vs Binomial**: Use Poisson when n is large and p is small
3. **Independence assumption**: Trials must be independent
4. **Constant probability**: p must stay the same across trials

---

## 📊 Visualization Tips

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

# Binomial distribution visualization
n, p = 20, 0.8
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

plt.bar(x, pmf)
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.show()
```

This guide covers all essential discrete probability distributions for data science applications!