# Statistics Basics Meeting Study Guide 📚
*Understanding Statistical Concepts Like a Smart 12-Year-Old*

## 🎯 What This Guide Covers
This study guide breaks down fundamental statistics concepts including descriptive statistics, inferential statistics, hypothesis testing, confidence intervals, and statistical distributions with easy-to-understand explanations, technical details, and interview preparation.

---

## 🌟 Part 1: Simple Explanations with Illustrations

### 1. What is Statistics?
**Simple Explanation:**
Statistics is like being a detective with numbers! It helps us understand what data is telling us and make smart decisions based on evidence.

```
🕵️ Statistics Detective Work:
📊 Collect clues (data)
🔍 Analyze patterns (descriptive statistics)
🎯 Make predictions (inferential statistics)
✅ Test theories (hypothesis testing)
📈 Measure confidence (confidence intervals)
```

**Two Main Types:**
```
📋 Descriptive Statistics: "What happened?"
- Summarize and describe data
- Mean, median, mode, standard deviation
- Like writing a summary of a book

🔮 Inferential Statistics: "What will happen?"
- Make predictions about populations from samples
- Hypothesis testing, confidence intervals
- Like predicting the ending of a book from the first few chapters
```

### 2. What are Measures of Central Tendency?
**Simple Explanation:**
These are different ways to find the "typical" or "average" value in your data - like finding the most representative student in your class!

```
🎯 The Three Averages:

📊 Mean (Arithmetic Average):
Class test scores: [80, 85, 90, 95, 100]
Mean = (80+85+90+95+100) ÷ 5 = 90
"Add everything up and divide by count"

🎯 Median (Middle Value):
Same scores arranged: [80, 85, 90, 95, 100]
Median = 90 (the middle number)
"The value in the exact middle"

👑 Mode (Most Common):
Favorite colors: [Red, Blue, Blue, Green, Blue]
Mode = Blue (appears most often)
"The most popular choice"
```

### 3. What is Standard Deviation?
**Simple Explanation:**
Standard deviation tells us how "spread out" or "scattered" our data is - like measuring how different students' heights are from the average height!

```
🎯 Low Standard Deviation (Everyone Similar):
Heights: [170cm, 171cm, 169cm, 170cm, 171cm]
Everyone is close to average → Small spread → Low std dev

🎯 High Standard Deviation (Very Different):
Heights: [150cm, 180cm, 160cm, 190cm, 140cm]
People vary a lot from average → Big spread → High std dev

📏 Visual Representation:
Low Std Dev:  ●●●●● (tight cluster)
High Std Dev: ●  ●  ●  ●  ● (spread out)
```

### 4. What is a Normal Distribution?
**Simple Explanation:**
The normal distribution is like a perfect bell-shaped hill where most people are "average" and fewer people are extremely tall or short!

```
🔔 Bell Curve Shape:
        ●
      ● ● ●
    ● ● ● ● ●
  ● ● ● ● ● ● ●
● ● ● ● ● ● ● ● ●

📊 The 68-95-99.7 Rule:
68% of data within 1 standard deviation
95% of data within 2 standard deviations  
99.7% of data within 3 standard deviations

🏃‍♂️ Example - Running Times:
Most people: Average time (middle of bell)
Few people: Very fast or very slow (edges of bell)
```

### 5. What is Hypothesis Testing?
**Simple Explanation:**
Hypothesis testing is like being a judge in court - you start by assuming someone is innocent (null hypothesis) and only change your mind if there's strong evidence they're guilty!

```
⚖️ Court Analogy:
Null Hypothesis (H₀): "Defendant is innocent"
Alternative Hypothesis (H₁): "Defendant is guilty"
Evidence: Data from the trial
Decision: Guilty only if evidence is very strong

🧪 Scientific Example:
H₀: "New medicine doesn't work"
H₁: "New medicine works"
Evidence: Test results from patients
Decision: Medicine works only if results are very convincing
```

---

## 🔬 Part 2: Technical Concepts

### 1. Descriptive Statistics Implementation

#### Measures of Central Tendency
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def calculate_central_tendency(data):
    """Calculate mean, median, and mode for a dataset"""
    
    # Convert to numpy array for easier calculation
    data = np.array(data)
    
    # Mean (arithmetic average)
    mean = np.mean(data)
    
    # Median (middle value)
    median = np.median(data)
    
    # Mode (most frequent value)
    mode_result = stats.mode(data, keepdims=True)
    mode = mode_result.mode[0]
    mode_count = mode_result.count[0]
    
    print(f"Dataset: {data}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode} (appears {mode_count} times)")
    
    return mean, median, mode

# Example with different types of distributions
print("=== Symmetric Distribution ===")
symmetric_data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
calculate_central_tendency(symmetric_data)

print("\n=== Right-Skewed Distribution ===")
right_skewed = [1, 1, 2, 2, 2, 3, 3, 4, 8, 10]
calculate_central_tendency(right_skewed)

print("\n=== Left-Skewed Distribution ===")
left_skewed = [1, 3, 7, 8, 8, 9, 9, 9, 10, 10]
calculate_central_tendency(left_skewed)
```

#### Measures of Variability
```python
def calculate_variability(data):
    """Calculate various measures of spread/variability"""
    
    data = np.array(data)
    
    # Range
    data_range = np.max(data) - np.min(data)
    
    # Variance (average squared deviation from mean)
    variance = np.var(data, ddof=1)  # ddof=1 for sample variance
    
    # Standard deviation (square root of variance)
    std_dev = np.std(data, ddof=1)
    
    # Interquartile Range (IQR)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Coefficient of Variation (relative variability)
    cv = (std_dev / np.mean(data)) * 100
    
    print(f"Dataset: {data}")
    print(f"Range: {data_range}")
    print(f"Variance: {variance:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"IQR (Q3 - Q1): {iqr:.2f}")
    print(f"Coefficient of Variation: {cv:.1f}%")
    
    return {
        'range': data_range,
        'variance': variance,
        'std_dev': std_dev,
        'iqr': iqr,
        'cv': cv
    }

# Compare variability in different datasets
print("=== Low Variability Dataset ===")
low_var_data = [48, 49, 50, 51, 52]
calculate_variability(low_var_data)

print("\n=== High Variability Dataset ===")
high_var_data = [10, 30, 50, 70, 90]
calculate_variability(high_var_data)
```

### 2. Probability Distributions

#### Normal Distribution
```python
def normal_distribution_analysis():
    """Analyze and visualize normal distribution properties"""
    
    # Generate normal distribution data
    mu, sigma = 100, 15  # mean=100, std=15 (like IQ scores)
    data = np.random.normal(mu, sigma, 10000)
    
    # Calculate empirical rule percentages
    within_1_std = np.sum((data >= mu - sigma) & (data <= mu + sigma)) / len(data)
    within_2_std = np.sum((data >= mu - 2*sigma) & (data <= mu + 2*sigma)) / len(data)
    within_3_std = np.sum((data >= mu - 3*sigma) & (data <= mu + 3*sigma)) / len(data)
    
    print("Normal Distribution Analysis (μ=100, σ=15):")
    print(f"Within 1 std dev (85-115): {within_1_std:.1%} (expected: 68%)")
    print(f"Within 2 std dev (70-130): {within_2_std:.1%} (expected: 95%)")
    print(f"Within 3 std dev (55-145): {within_3_std:.1%} (expected: 99.7%)")
    
    # Z-score calculations
    def calculate_z_score(x, mu, sigma):
        return (x - mu) / sigma
    
    # Example z-score calculations
    values = [85, 100, 115, 130]
    print(f"\nZ-score calculations:")
    for val in values:
        z = calculate_z_score(val, mu, sigma)
        percentile = stats.norm.cdf(z) * 100
        print(f"Value {val}: z-score = {z:.2f}, percentile = {percentile:.1f}%")
    
    return data

normal_data = normal_distribution_analysis()
```

#### Other Important Distributions
```python
def compare_distributions():
    """Compare different probability distributions"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Normal distribution
    normal = np.random.normal(50, 10, n_samples)
    
    # Uniform distribution
    uniform = np.random.uniform(30, 70, n_samples)
    
    # Exponential distribution
    exponential = np.random.exponential(10, n_samples)
    
    # Binomial distribution
    binomial = np.random.binomial(100, 0.3, n_samples)
    
    distributions = {
        'Normal': normal,
        'Uniform': uniform,
        'Exponential': exponential,
        'Binomial': binomial
    }
    
    print("Distribution Comparison:")
    print("-" * 50)
    
    for name, data in distributions.items():
        mean = np.mean(data)
        std = np.std(data)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        print(f"{name:12} | Mean: {mean:6.2f} | Std: {std:6.2f} | "
              f"Skew: {skewness:6.2f} | Kurt: {kurtosis:6.2f}")
    
    return distributions

distributions = compare_distributions()
```

### 3. Hypothesis Testing

#### One-Sample t-test
```python
def one_sample_t_test(sample_data, population_mean, alpha=0.05):
    """
    Perform one-sample t-test
    
    H₀: μ = population_mean
    H₁: μ ≠ population_mean
    """
    
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    n = len(sample_data)
    
    # Calculate t-statistic
    t_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
    
    # Calculate degrees of freedom
    df = n - 1
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Critical value
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Decision
    reject_null = p_value < alpha
    
    print("One-Sample t-Test Results:")
    print(f"Sample mean: {sample_mean:.3f}")
    print(f"Population mean (H₀): {population_mean}")
    print(f"Sample size: {n}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"Degrees of freedom: {df}")
    print(f"p-value: {p_value:.4f}")
    print(f"Critical value (α={alpha}): ±{t_critical:.3f}")
    print(f"Decision: {'Reject H₀' if reject_null else 'Fail to reject H₀'}")
    
    if reject_null:
        print(f"Conclusion: There is significant evidence that the population mean ≠ {population_mean}")
    else:
        print(f"Conclusion: There is insufficient evidence that the population mean ≠ {population_mean}")
    
    return t_stat, p_value, reject_null

# Example: Test if average height is different from 170cm
heights = [168, 172, 165, 175, 170, 169, 173, 167, 171, 174]
one_sample_t_test(heights, population_mean=170, alpha=0.05)
```

#### Two-Sample t-test
```python
def two_sample_t_test(group1, group2, alpha=0.05):
    """
    Perform independent two-sample t-test
    
    H₀: μ₁ = μ₂
    H₁: μ₁ ≠ μ₂
    """
    
    # Calculate sample statistics
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    
    # Standard error
    se = pooled_std * np.sqrt(1/n1 + 1/n2)
    
    # t-statistic
    t_stat = (mean1 - mean2) / se
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    # p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Critical value
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Decision
    reject_null = p_value < alpha
    
    print("Two-Sample t-Test Results:")
    print(f"Group 1: n={n1}, mean={mean1:.3f}, std={std1:.3f}")
    print(f"Group 2: n={n2}, mean={mean2:.3f}, std={std2:.3f}")
    print(f"Difference in means: {mean1 - mean2:.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"Degrees of freedom: {df}")
    print(f"p-value: {p_value:.4f}")
    print(f"Critical value (α={alpha}): ±{t_critical:.3f}")
    print(f"Decision: {'Reject H₀' if reject_null else 'Fail to reject H₀'}")
    
    return t_stat, p_value, reject_null

# Example: Compare test scores between two teaching methods
method_a = [85, 87, 82, 90, 88, 86, 84, 89, 91, 83]
method_b = [78, 80, 75, 82, 79, 77, 81, 76, 84, 78]
two_sample_t_test(method_a, method_b, alpha=0.05)
```

### 4. Confidence Intervals

#### Confidence Interval for Mean
```python
def confidence_interval_mean(data, confidence_level=0.95):
    """
    Calculate confidence interval for population mean
    """
    
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    # Calculate alpha and degrees of freedom
    alpha = 1 - confidence_level
    df = n - 1
    
    # Critical value from t-distribution
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Margin of error
    margin_error = t_critical * se
    
    # Confidence interval
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    print(f"Confidence Interval for Population Mean:")
    print(f"Sample size: {n}")
    print(f"Sample mean: {mean:.3f}")
    print(f"Sample std: {std:.3f}")
    print(f"Standard error: {se:.3f}")
    print(f"Confidence level: {confidence_level:.0%}")
    print(f"Degrees of freedom: {df}")
    print(f"t-critical: {t_critical:.3f}")
    print(f"Margin of error: {margin_error:.3f}")
    print(f"{confidence_level:.0%} CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    print(f"\nInterpretation:")
    print(f"We are {confidence_level:.0%} confident that the true population mean")
    print(f"is between {ci_lower:.3f} and {ci_upper:.3f}")
    
    return ci_lower, ci_upper

# Example: Confidence interval for average reaction time
reaction_times = [0.25, 0.28, 0.22, 0.30, 0.26, 0.24, 0.29, 0.23, 0.27, 0.25]
confidence_interval_mean(reaction_times, confidence_level=0.95)
```

### 5. Effect Size and Power Analysis

#### Cohen's d (Effect Size)
```python
def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size for two groups
    
    Cohen's d interpretation:
    - Small effect: d = 0.2
    - Medium effect: d = 0.5  
    - Large effect: d = 0.8
    """
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    # Interpret effect size
    if abs(d) < 0.2:
        interpretation = "Negligible"
    elif abs(d) < 0.5:
        interpretation = "Small"
    elif abs(d) < 0.8:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    
    print(f"Effect Size Analysis (Cohen's d):")
    print(f"Group 1: mean={mean1:.3f}, std={std1:.3f}, n={n1}")
    print(f"Group 2: mean={mean2:.3f}, std={std2:.3f}, n={n2}")
    print(f"Pooled standard deviation: {pooled_std:.3f}")
    print(f"Cohen's d: {d:.3f}")
    print(f"Effect size interpretation: {interpretation}")
    
    return d

# Example: Effect size for two treatment groups
treatment_a = [7.2, 8.1, 6.8, 7.9, 8.3, 7.5, 8.0, 7.7, 8.2, 7.4]
treatment_b = [6.1, 6.8, 5.9, 6.5, 6.9, 6.3, 6.7, 6.2, 6.6, 6.4]
cohens_d(treatment_a, treatment_b)
```

---

## 🎤 Part 3: Interview Questions & Detailed Answers

### Basic Level Questions

#### Q1: What's the difference between population and sample?

**Answer:**

| Aspect | Population | Sample |
|--------|------------|--------|
| **Definition** | Complete group of interest | Subset of the population |
| **Size** | Usually very large/infinite | Smaller, manageable size |
| **Parameters** | μ (mu), σ (sigma) | x̄ (x-bar), s |
| **Purpose** | What we want to know about | What we actually study |
| **Example** | All voters in a country | 1,000 voters surveyed |

**Code Example:**
```python
# Population vs Sample demonstration
import numpy as np

# Simulate a population (all students' test scores)
np.random.seed(42)
population = np.random.normal(75, 12, 100000)  # 100,000 students
population_mean = np.mean(population)
population_std = np.std(population)

print(f"Population (N=100,000):")
print(f"μ (population mean) = {population_mean:.2f}")
print(f"σ (population std) = {population_std:.2f}")

# Take a sample
sample_size = 100
sample = np.random.choice(population, sample_size, replace=False)
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)  # ddof=1 for sample std

print(f"\nSample (n=100):")
print(f"x̄ (sample mean) = {sample_mean:.2f}")
print(f"s (sample std) = {sample_std:.2f}")

print(f"\nSample estimates population parameters:")
print(f"Difference in means: {abs(population_mean - sample_mean):.2f}")
print(f"Difference in stds: {abs(population_std - sample_std):.2f}")
```

#### Q2: Explain the difference between Type I and Type II errors.

**Answer:**

**Type I Error (α):** Rejecting a true null hypothesis (False Positive)
**Type II Error (β):** Failing to reject a false null hypothesis (False Negative)

| Reality | H₀ True | H₀ False |
|---------|---------|----------|
| **Reject H₀** | Type I Error (α) | Correct Decision |
| **Fail to Reject H₀** | Correct Decision | Type II Error (β) |

**Medical Test Analogy:**
```python
def medical_test_errors():
    """
    Demonstrate Type I and Type II errors with medical testing
    """
    
    print("Medical Test Error Types:")
    print("H₀: Patient is healthy")
    print("H₁: Patient is sick")
    print()
    
    scenarios = [
        ("Type I Error (α)", "Patient is healthy", "Test says sick", "False Positive"),
        ("Type II Error (β)", "Patient is sick", "Test says healthy", "False Negative"),
        ("Correct Decision", "Patient is healthy", "Test says healthy", "True Negative"),
        ("Correct Decision", "Patient is sick", "Test says sick", "True Positive")
    ]
    
    for error_type, reality, test_result, consequence in scenarios:
        print(f"{error_type:15} | Reality: {reality:15} | Test: {test_result:15} | {consequence}")
    
    print(f"\nConsequences:")
    print(f"Type I Error: Unnecessary treatment, anxiety, cost")
    print(f"Type II Error: Missed diagnosis, delayed treatment, health risk")
    
    # Relationship between α and β
    print(f"\nTrade-off between Type I and Type II errors:")
    print(f"- Lower α (stricter criteria) → Higher β (more missed cases)")
    print(f"- Higher α (lenient criteria) → Lower β (fewer missed cases)")

medical_test_errors()
```

#### Q3: What is the Central Limit Theorem and why is it important?

**Answer:**

**Central Limit Theorem (CLT):** Regardless of the population distribution shape, the sampling distribution of sample means approaches a normal distribution as sample size increases (n ≥ 30).

**Key Points:**
1. **Sample means are normally distributed** (even if population isn't)
2. **Mean of sample means = population mean** (μₓ̄ = μ)
3. **Standard error decreases with sample size** (σₓ̄ = σ/√n)

```python
def demonstrate_central_limit_theorem():
    """
    Demonstrate CLT with different population distributions
    """
    
    import matplotlib.pyplot as plt
    
    # Create different population distributions
    np.random.seed(42)
    
    # Highly skewed population (exponential)
    population = np.random.exponential(2, 100000)
    
    sample_sizes = [5, 10, 30, 100]
    sample_means_collections = []
    
    print("Central Limit Theorem Demonstration:")
    print(f"Population: Exponential distribution (highly skewed)")
    print(f"Population mean: {np.mean(population):.3f}")
    print(f"Population std: {np.std(population):.3f}")
    print()
    
    for n in sample_sizes:
        # Take many samples of size n
        sample_means = []
        for _ in range(1000):
            sample = np.random.choice(population, n, replace=False)
            sample_means.append(np.mean(sample))
        
        sample_means = np.array(sample_means)
        sample_means_collections.append(sample_means)
        
        # Calculate statistics
        mean_of_means = np.mean(sample_means)
        std_of_means = np.std(sample_means)
        theoretical_se = np.std(population) / np.sqrt(n)
        
        print(f"Sample size n={n}:")
        print(f"  Mean of sample means: {mean_of_means:.3f}")
        print(f"  Std of sample means: {std_of_means:.3f}")
        print(f"  Theoretical SE (σ/√n): {theoretical_se:.3f}")
        print(f"  Normality test p-value: {stats.shapiro(sample_means[:100])[1]:.4f}")
        print()
    
    return sample_means_collections

demonstrate_central_limit_theorem()
```

### Intermediate Level Questions

#### Q4: How do you choose between a t-test and z-test?

**Answer:**

**Decision Framework:**

| Condition | Test Choice | Reason |
|-----------|-------------|--------|
| **σ known, any n** | Z-test | Population std known |
| **σ unknown, n ≥ 30** | Z-test or t-test | CLT applies, both valid |
| **σ unknown, n < 30** | t-test | Account for uncertainty in σ |
| **Non-normal population, n < 30** | Non-parametric | Assumptions violated |

```python
def test_selection_guide(sample_data, population_std=None, population_mean=100):
    """
    Guide for selecting appropriate statistical test
    """
    
    n = len(sample_data)
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    
    print("Statistical Test Selection Guide:")
    print(f"Sample size: {n}")
    print(f"Sample mean: {sample_mean:.3f}")
    print(f"Sample std: {sample_std:.3f}")
    print(f"Population std known: {'Yes' if population_std else 'No'}")
    print()
    
    # Test for normality
    if n >= 8:  # Shapiro-Wilk requires n >= 8
        _, p_normal = stats.shapiro(sample_data)
        is_normal = p_normal > 0.05
        print(f"Normality test p-value: {p_normal:.4f}")
        print(f"Data appears normal: {'Yes' if is_normal else 'No'}")
    else:
        is_normal = True  # Assume normal for small samples
        print("Sample too small for normality test")
    
    print()
    
    # Decision logic
    if population_std is not None:
        print("Recommendation: Z-test")
        print("Reason: Population standard deviation is known")
        
        # Perform z-test
        z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(n))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        print(f"Z-statistic: {z_stat:.3f}")
        print(f"P-value: {p_value:.4f}")
        
    elif n >= 30:
        print("Recommendation: t-test (or z-test)")
        print("Reason: Large sample size (n ≥ 30), CLT applies")
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)
        
        print(f"t-statistic: {t_stat:.3f}")
        print(f"P-value: {p_value:.4f}")
        
    elif n < 30 and is_normal:
        print("Recommendation: t-test")
        print("Reason: Small sample, population std unknown, data is normal")
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)
        
        print(f"t-statistic: {t_stat:.3f}")
        print(f"P-value: {p_value:.4f}")
        
    else:
        print("Recommendation: Non-parametric test (e.g., Wilcoxon signed-rank)")
        print("Reason: Small sample, data not normal")
        
        # Perform Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(sample_data - population_mean)
        
        print(f"Wilcoxon statistic: {stat:.3f}")
        print(f"P-value: {p_value:.4f}")

# Examples
print("=== Example 1: Large sample ===")
large_sample = np.random.normal(102, 8, 50)
test_selection_guide(large_sample)

print("\n=== Example 2: Small sample, normal ===")
small_normal = np.random.normal(98, 6, 15)
test_selection_guide(small_normal)

print("\n=== Example 3: Known population std ===")
known_std_sample = np.random.normal(105, 10, 20)
test_selection_guide(known_std_sample, population_std=10)
```

#### Q5: Explain the concept of statistical power and how to calculate it.

**Answer:**

**Statistical Power** is the probability of correctly rejecting a false null hypothesis (1 - β). It's the ability to detect an effect when it truly exists.

**Factors affecting power:**
1. **Effect size** (larger effect → higher power)
2. **Sample size** (larger n → higher power)  
3. **Significance level α** (larger α → higher power)
4. **Population variance** (smaller σ → higher power)

```python
def power_analysis_demonstration():
    """
    Demonstrate statistical power calculation and factors affecting it
    """
    
    from scipy.stats import norm, t
    
    def calculate_power(effect_size, sample_size, alpha=0.05, sigma=1):
        """
        Calculate statistical power for one-sample t-test
        
        Args:
            effect_size: (μ₁ - μ₀) / σ (Cohen's d)
            sample_size: n
            alpha: significance level
            sigma: population standard deviation
        """
        
        # Critical value for two-tailed test
        df = sample_size - 1
        t_critical = t.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size)
        
        # Power calculation using non-central t-distribution
        # This is an approximation using normal distribution
        se = sigma / np.sqrt(sample_size)
        critical_value = t_critical * se
        
        # Power = P(reject H₀ | H₁ is true)
        power = 1 - norm.cdf(critical_value - effect_size, 0, se) + norm.cdf(-critical_value - effect_size, 0, se)
        
        return power
    
    # Demonstrate effect of different factors on power
    print("Statistical Power Analysis:")
    print("=" * 50)
    
    # 1. Effect of sample size
    print("\n1. Effect of Sample Size (effect size = 0.5, α = 0.05):")
    sample_sizes = [10, 20, 30, 50, 100]
    for n in sample_sizes:
        power = calculate_power(effect_size=0.5, sample_size=n)
        print(f"n = {n:3d}: Power = {power:.3f}")
    
    # 2. Effect of effect size
    print("\n2. Effect of Effect Size (n = 30, α = 0.05):")
    effect_sizes = [0.2, 0.5, 0.8, 1.0, 1.5]
    for es in effect_sizes:
        power = calculate_power(effect_size=es, sample_size=30)
        print(f"Effect size = {es:.1f}: Power = {power:.3f}")
    
    # 3. Effect of significance level
    print("\n3. Effect of Significance Level (n = 30, effect size = 0.5):")
    alphas = [0.01, 0.05, 0.10, 0.20]
    for alpha in alphas:
        power = calculate_power(effect_size=0.5, sample_size=30, alpha=alpha)
        print(f"α = {alpha:.2f}: Power = {power:.3f}")
    
    # Sample size calculation for desired power
    def calculate_sample_size_for_power(effect_size, desired_power=0.8, alpha=0.05):
        """Calculate required sample size for desired power"""
        
        for n in range(5, 1000):
            power = calculate_power(effect_size, n, alpha)
            if power >= desired_power:
                return n
        return None
    
    print(f"\n4. Sample Size for 80% Power:")
    effect_sizes = [0.2, 0.5, 0.8]
    for es in effect_sizes:
        n_required = calculate_sample_size_for_power(es, desired_power=0.8)
        print(f"Effect size {es:.1f}: n = {n_required}")
    
    return calculate_power

power_calc = power_analysis_demonstration()
```

### Advanced Level Questions

#### Q6: Implement and explain the bootstrap method for confidence intervals.

**Answer:**

**Bootstrap Method:** A resampling technique that estimates the sampling distribution by repeatedly sampling with replacement from the original sample.

```python
def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=10000, confidence_level=0.95):
    """
    Calculate bootstrap confidence interval for any statistic
    
    Args:
        data: Original sample data
        statistic_func: Function to calculate statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Bootstrap confidence interval
    """
    
    n = len(data)
    bootstrap_statistics = []
    
    # Generate bootstrap samples
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        # Calculate statistic for this bootstrap sample
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_statistics.append(bootstrap_stat)
    
    bootstrap_statistics = np.array(bootstrap_statistics)
    
    # Calculate confidence interval using percentile method
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
    ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
    
    # Original statistic
    original_stat = statistic_func(data)
    
    # Bootstrap standard error
    bootstrap_se = np.std(bootstrap_statistics)
    
    print(f"Bootstrap Confidence Interval Analysis:")
    print(f"Original sample size: {n}")
    print(f"Number of bootstrap samples: {n_bootstrap}")
    print(f"Confidence level: {confidence_level:.0%}")
    print(f"Original statistic: {original_stat:.4f}")
    print(f"Bootstrap mean: {np.mean(bootstrap_statistics):.4f}")
    print(f"Bootstrap standard error: {bootstrap_se:.4f}")
    print(f"{confidence_level:.0%} CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return ci_lower, ci_upper, bootstrap_statistics

def compare_bootstrap_methods():
    """Compare bootstrap with traditional methods"""
    
    # Generate sample data
    np.random.seed(42)
    sample_data = np.random.exponential(2, 30)  # Skewed distribution
    
    print("Comparison: Bootstrap vs Traditional Methods")
    print("=" * 50)
    print(f"Sample size: {len(sample_data)}")
    print(f"Sample mean: {np.mean(sample_data):.4f}")
    print(f"Sample median: {np.median(sample_data):.4f}")
    print()
    
    # 1. Bootstrap CI for mean
    print("1. Confidence Interval for Mean:")
    boot_ci_lower, boot_ci_upper, boot_stats = bootstrap_confidence_interval(
        sample_data, np.mean, confidence_level=0.95
    )
    
    # Traditional t-based CI for mean
    mean = np.mean(sample_data)
    se = stats.sem(sample_data)
    df = len(sample_data) - 1
    t_critical = stats.t.ppf(0.975, df)
    traditional_ci_lower = mean - t_critical * se
    traditional_ci_upper = mean + t_critical * se
    
    print(f"Traditional 95% CI: [{traditional_ci_lower:.4f}, {traditional_ci_upper:.4f}]")
    print()
    
    # 2. Bootstrap CI for median (no traditional equivalent)
    print("2. Confidence Interval for Median:")
    bootstrap_confidence_interval(sample_data, np.median, confidence_level=0.95)
    print("Note: No simple traditional method for median CI")
    print()
    
    # 3. Bootstrap CI for custom statistic
    def coefficient_of_variation(x):
        return np.std(x, ddof=1) / np.mean(x)
    
    print("3. Confidence Interval for Coefficient of Variation:")
    bootstrap_confidence_interval(sample_data, coefficient_of_variation, confidence_level=0.95)
    print("Note: Bootstrap easily handles complex statistics")
    
    return boot_stats

bootstrap_stats = compare_bootstrap_methods()
```

#### Q7: Explain and implement multiple testing correction methods.

**Answer:**

**Multiple Testing Problem:** When performing multiple hypothesis tests, the probability of making at least one Type I error increases. Need to adjust p-values or significance levels.

```python
def multiple_testing_correction():
    """
    Demonstrate multiple testing problem and correction methods
    """
    
    from scipy.stats import false_discovery_control
    
    # Simulate multiple hypothesis tests
    np.random.seed(42)
    n_tests = 20
    
    # Generate p-values: 15 from null hypothesis (should be uniform)
    # and 5 from alternative hypothesis (should be small)
    null_p_values = np.random.uniform(0, 1, 15)
    alternative_p_values = np.random.beta(1, 10, 5)  # Skewed toward 0
    
    p_values = np.concatenate([null_p_values, alternative_p_values])
    np.random.shuffle(p_values)
    
    alpha = 0.05
    
    print("Multiple Testing Correction Methods:")
    print("=" * 50)
    print(f"Number of tests: {n_tests}")
    print(f"Original α level: {alpha}")
    print(f"P-values: {p_values}")
    print()
    
    # 1. No correction (naive approach)
    naive_significant = p_values < alpha
    naive_discoveries = np.sum(naive_significant)
    
    print(f"1. No Correction:")
    print(f"   Significant tests: {naive_discoveries}")
    print(f"   Expected false positives: {n_tests * alpha:.1f}")
    print(f"   Family-wise error rate: {1 - (1-alpha)**n_tests:.3f}")
    print()
    
    # 2. Bonferroni correction
    bonferroni_alpha = alpha / n_tests
    bonferroni_significant = p_values < bonferroni_alpha
    bonferroni_discoveries = np.sum(bonferroni_significant)
    
    print(f"2. Bonferroni Correction:")
    print(f"   Adjusted α: {bonferroni_alpha:.4f}")
    print(f"   Significant tests: {bonferroni_discoveries}")
    print(f"   Controls: Family-wise error rate ≤ {alpha}")
    print()
    
    # 3. Holm-Bonferroni correction
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    holm_significant = np.zeros(n_tests, dtype=bool)
    for i, p in enumerate(sorted_p_values):
        adjusted_alpha = alpha / (n_tests - i)
        if p < adjusted_alpha:
            holm_significant[sorted_indices[i]] = True
        else:
            break  # Stop at first non-significant test
    
    holm_discoveries = np.sum(holm_significant)
    
    print(f"3. Holm-Bonferroni Correction:")
    print(f"   Significant tests: {holm_discoveries}")
    print(f"   Controls: Family-wise error rate ≤ {alpha}")
    print(f"   More powerful than Bonferroni")
    print()
    
    # 4. Benjamini-Hochberg (FDR control)
    def benjamini_hochberg(p_values, fdr_level=0.05):
        """Benjamini-Hochberg procedure for FDR control"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Find largest k such that P(k) <= (k/n) * α
        significant = np.zeros(n, dtype=bool)
        for i in range(n-1, -1, -1):
            if sorted_p[i] <= (i+1) / n * fdr_level:
                # Reject all hypotheses up to and including i
                for j in range(i+1):
                    significant[sorted_indices[j]] = True
                break
        
        return significant
    
    bh_significant = benjamini_hochberg(p_values, fdr_level=alpha)
    bh_discoveries = np.sum(bh_significant)
    
    print(f"4. Benjamini-Hochberg (FDR Control):")
    print(f"   Significant tests: {bh_discoveries}")
    print(f"   Controls: False Discovery Rate ≤ {alpha}")
    print(f"   More powerful than FWER methods")
    print()
    
    # Summary comparison
    print("Summary Comparison:")
    print(f"{'Method':<20} {'Discoveries':<12} {'Type of Control'}")
    print("-" * 50)
    print(f"{'No correction':<20} {naive_discoveries:<12} {'None'}")
    print(f"{'Bonferroni':<20} {bonferroni_discoveries:<12} {'FWER'}")
    print(f"{'Holm-Bonferroni':<20} {holm_discoveries:<12} {'FWER'}")
    print(f"{'Benjamini-Hochberg':<20} {bh_discoveries:<12} {'FDR'}")
    
    print(f"\nDefinitions:")
    print(f"FWER: Family-Wise Error Rate (probability of ≥1 false positive)")
    print(f"FDR: False Discovery Rate (expected proportion of false positives)")
    
    return p_values, {
        'naive': naive_significant,
        'bonferroni': bonferroni_significant,
        'holm': holm_significant,
        'bh': bh_significant
    }

p_vals, corrections = multiple_testing_correction()
```

---

## 🚀 Practical Tips for Interviews

### 1. **Know When to Use Each Test**
```python
# Quick reference for test selection
test_selection = {
    "One sample, σ known": "Z-test",
    "One sample, σ unknown, n≥30": "t-test",
    "One sample, σ unknown, n<30, normal": "t-test", 
    "One sample, non-normal, small n": "Wilcoxon signed-rank",
    "Two samples, independent": "Two-sample t-test",
    "Two samples, paired": "Paired t-test",
    "Multiple groups": "ANOVA",
    "Categorical data": "Chi-square test"
}
```

### 2. **Understand Assumptions**
Always check and state assumptions:
- **Normality**: Data follows normal distribution
- **Independence**: Observations are independent
- **Homoscedasticity**: Equal variances across groups
- **Random sampling**: Sample represents population

### 3. **Interpret Results Correctly**
```python
def interpret_p_value(p_value, alpha=0.05):
    """Proper interpretation of p-values"""
    if p_value < alpha:
        return f"p = {p_value:.4f} < α = {alpha}, reject H₀"
    else:
        return f"p = {p_value:.4f} ≥ α = {alpha}, fail to reject H₀"
```

### 4. **Know Effect Sizes**
Don't just report significance - report practical significance:
- **Cohen's d**: 0.2 (small), 0.5 (medium), 0.8 (large)
- **R-squared**: Proportion of variance explained
- **Confidence intervals**: Range of plausible values

---

## 📚 Key Concepts from the Meeting

### 1. **Descriptive Statistics:**
- Measures of central tendency (mean, median, mode)
- Measures of variability (range, variance, standard deviation)
- Distribution shapes and properties

### 2. **Inferential Statistics:**
- Sampling distributions and Central Limit Theorem
- Hypothesis testing framework
- Confidence intervals and interpretation

### 3. **Statistical Tests:**
- t-tests (one-sample, two-sample, paired)
- Z-tests and when to use them
- Non-parametric alternatives

### 4. **Advanced Topics:**
- Statistical power and effect size
- Multiple testing corrections
- Bootstrap methods for robust inference

---

## 📊 Additional Resources

### Essential Statistical Concepts:
1. **Descriptive Statistics**: Summarizing and describing data
2. **Probability Distributions**: Normal, t, chi-square, F distributions
3. **Hypothesis Testing**: Framework for statistical inference
4. **Confidence Intervals**: Estimating population parameters

### Practical Applications:
- A/B testing in business and tech
- Quality control in manufacturing
- Clinical trials in medicine
- Survey research and polling

### Interview Preparation:
- Practice interpreting statistical output
- Understand when to use different tests
- Know how to check assumptions
- Be able to explain results to non-technical audiences

---

*Remember: Statistics interviews test both theoretical understanding and practical application. Focus on understanding concepts deeply rather than memorizing formulas, and always consider the real-world context of statistical problems!* 🎯