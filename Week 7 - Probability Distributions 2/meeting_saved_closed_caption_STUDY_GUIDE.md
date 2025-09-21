# Probability Distributions 2 Meeting Study Guide 📚
*Understanding Normal Distribution and Central Limit Theorem Like a Smart 12-Year-Old*

## 🎯 What This Guide Covers
This study guide covers advanced probability distributions, focusing on the normal distribution, Central Limit Theorem, and their applications in data science and statistics.

---

## 🌟 Part 1: Simple Explanations with Illustrations

### 1. What is the Normal Distribution?
**Simple Explanation:**
The normal distribution is like a perfect bell-shaped hill where most people are "average" and fewer people are at the extremes!

```
🔔 Bell Curve Shape:
        ●
      ● ● ●
    ● ● ● ● ●
  ● ● ● ● ● ● ●
● ● ● ● ● ● ● ● ●

🏃‍♂️ Real Example - Heights:
Short people: Few (left tail)
Average height: Most people (center peak)
Very tall people: Few (right tail)

📊 Age Distribution Example:
Babies (0-5): Few people
Adults (25-35): Most people (peak)
Elderly (80+): Few people
```

### 2. Why is Normal Distribution So Popular?
**Simple Explanation:**
Normal distribution is like the "Swiss Army knife" of statistics - it works for almost everything!

```
🌟 Why Everyone Loves Normal Distribution:

1. 📏 Symmetric: Perfect balance on both sides
2. 🎯 Predictable: 68-95-99.7 rule always works
3. 🔄 Stable: Adding normal distributions gives normal distribution
4. 📈 Universal: Many real-world things follow this pattern
5. 🧮 Easy Math: Simple formulas and calculations

Real Examples:
- Test scores in a large class
- Heights of people
- Measurement errors
- Blood pressure readings
- IQ scores
```

### 3. What is the Central Limit Theorem?
**Simple Explanation:**
The Central Limit Theorem is like magic - no matter what shape your data starts with, if you take enough samples and average them, you'll always get a bell curve!

```
🎩 The Magic Trick:

Step 1: Start with ANY distribution (even weird shapes)
🔷🔶🔸 (could be skewed, uniform, or bumpy)

Step 2: Take many samples (groups of 30+ data points)
Sample 1: [🔷🔶🔸🔷🔶] → Average = 5.2
Sample 2: [🔸🔷🔶🔷🔸] → Average = 4.8
Sample 3: [🔶🔸🔷🔶🔷] → Average = 5.1
... (do this 1000+ times)

Step 3: Plot all the sample averages
Result: Perfect bell curve! 🔔

🎯 Key Point: Sample size ≥ 30 is the magic number!
```

### 4. What are the Properties of Normal Distribution?
**Simple Explanation:**
Normal distribution has special "superpowers" that make it incredibly useful!

```
🦸‍♂️ Normal Distribution Superpowers:

1. 🎯 Mean = Median = Mode (all at the center)
2. 📏 Perfectly symmetric (mirror image on both sides)
3. 📊 68-95-99.7 Rule:
   - 68% within 1 standard deviation
   - 95% within 2 standard deviations
   - 99.7% within 3 standard deviations
4. 🔄 Stable under addition (Normal + Normal = Normal)
5. 📐 Defined by just 2 parameters: μ (mean) and σ (standard deviation)

🎲 Example - IQ Scores:
Mean (μ) = 100, Standard Deviation (σ) = 15
- 68% of people have IQ between 85-115
- 95% of people have IQ between 70-130
- 99.7% of people have IQ between 55-145
```

---

## 🔬 Part 2: Technical Concepts

### 1. Normal Distribution Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class NormalDistribution:
    def __init__(self, mu=0, sigma=1):
        """
        Initialize normal distribution
        mu: mean
        sigma: standard deviation
        """
        self.mu = mu
        self.sigma = sigma
        self.distribution = stats.norm(mu, sigma)
    
    def pdf(self, x):
        """Probability Density Function"""
        return self.distribution.pdf(x)
    
    def cdf(self, x):
        """Cumulative Distribution Function"""
        return self.distribution.cdf(x)
    
    def sample(self, size=1000):
        """Generate random samples"""
        return self.distribution.rvs(size=size)
    
    def empirical_rule_check(self, data=None):
        """Verify the 68-95-99.7 rule"""
        if data is None:
            data = self.sample(10000)
        
        # Calculate percentages within 1, 2, 3 standard deviations
        within_1_std = np.sum((data >= self.mu - self.sigma) & 
                             (data <= self.mu + self.sigma)) / len(data)
        within_2_std = np.sum((data >= self.mu - 2*self.sigma) & 
                             (data <= self.mu + 2*self.sigma)) / len(data)
        within_3_std = np.sum((data >= self.mu - 3*self.sigma) & 
                             (data <= self.mu + 3*self.sigma)) / len(data)
        
        print(f"Empirical Rule Verification (μ={self.mu}, σ={self.sigma}):")
        print(f"Within 1σ: {within_1_std:.1%} (expected: 68%)")
        print(f"Within 2σ: {within_2_std:.1%} (expected: 95%)")
        print(f"Within 3σ: {within_3_std:.1%} (expected: 99.7%)")
        
        return within_1_std, within_2_std, within_3_std

# Example usage
normal_dist = NormalDistribution(mu=100, sigma=15)  # IQ scores
normal_dist.empirical_rule_check()
```

### 2. Central Limit Theorem Demonstration
```python
def demonstrate_central_limit_theorem():
    """
    Demonstrate CLT with different population distributions
    """
    
    # Create different population distributions
    np.random.seed(42)
    
    distributions = {
        'Uniform': np.random.uniform(0, 10, 100000),
        'Exponential': np.random.exponential(2, 100000),
        'Bimodal': np.concatenate([
            np.random.normal(2, 1, 50000),
            np.random.normal(8, 1, 50000)
        ])
    }
    
    sample_sizes = [5, 10, 30, 100]
    n_samples = 1000
    
    results = {}
    
    for dist_name, population in distributions.items():
        print(f"\n{dist_name} Distribution:")
        print(f"Population mean: {np.mean(population):.3f}")
        print(f"Population std: {np.std(population):.3f}")
        
        results[dist_name] = {}
        
        for n in sample_sizes:
            # Take many samples of size n and calculate their means
            sample_means = []
            for _ in range(n_samples):
                sample = np.random.choice(population, n, replace=False)
                sample_means.append(np.mean(sample))
            
            sample_means = np.array(sample_means)
            
            # Test normality of sample means
            _, p_value = stats.shapiro(sample_means[:100])  # Shapiro-Wilk test
            
            results[dist_name][n] = {
                'means': sample_means,
                'mean_of_means': np.mean(sample_means),
                'std_of_means': np.std(sample_means),
                'theoretical_se': np.std(population) / np.sqrt(n),
                'normality_p': p_value
            }
            
            print(f"  n={n}: mean={np.mean(sample_means):.3f}, "
                  f"std={np.std(sample_means):.3f}, "
                  f"theoretical_SE={np.std(population)/np.sqrt(n):.3f}, "
                  f"normal_p={p_value:.4f}")
    
    return results

clt_results = demonstrate_central_limit_theorem()
```

### 3. Standard Normal Distribution (Z-scores)
```python
def z_score_analysis():
    """
    Demonstrate Z-score calculations and interpretations
    """
    
    def calculate_z_score(x, mu, sigma):
        """Calculate Z-score: (x - μ) / σ"""
        return (x - mu) / sigma
    
    def interpret_z_score(z):
        """Interpret Z-score magnitude"""
        abs_z = abs(z)
        if abs_z < 1:
            return "Within 1 standard deviation (common)"
        elif abs_z < 2:
            return "Within 2 standard deviations (somewhat unusual)"
        elif abs_z < 3:
            return "Within 3 standard deviations (unusual)"
        else:
            return "Beyond 3 standard deviations (very rare)"
    
    # Example: SAT scores (μ=1500, σ=300)
    mu, sigma = 1500, 300
    
    test_scores = [1200, 1500, 1650, 1800, 2100]
    
    print("Z-Score Analysis for SAT Scores (μ=1500, σ=300):")
    print("-" * 60)
    
    for score in test_scores:
        z = calculate_z_score(score, mu, sigma)
        percentile = stats.norm.cdf(z) * 100
        interpretation = interpret_z_score(z)
        
        print(f"Score: {score:4d} | Z-score: {z:6.2f} | "
              f"Percentile: {percentile:5.1f}% | {interpretation}")
    
    # Probability calculations
    print(f"\nProbability Calculations:")
    print(f"P(Score > 1800) = {1 - stats.norm.cdf(calculate_z_score(1800, mu, sigma)):.4f}")
    print(f"P(1200 < Score < 1800) = {stats.norm.cdf(calculate_z_score(1800, mu, sigma)) - stats.norm.cdf(calculate_z_score(1200, mu, sigma)):.4f}")

z_score_analysis()
```

### 4. Comparing Different Distributions
```python
def compare_distributions():
    """
    Compare normal distribution with other distributions
    """
    
    # Generate data from different distributions
    np.random.seed(42)
    n_samples = 10000
    
    distributions = {
        'Normal': np.random.normal(50, 10, n_samples),
        'Uniform': np.random.uniform(30, 70, n_samples),
        'Exponential': np.random.exponential(10, n_samples),
        'Beta (α=2, β=5)': np.random.beta(2, 5, n_samples) * 100,
        'Gamma': np.random.gamma(2, 5, n_samples)
    }
    
    print("Distribution Comparison:")
    print("-" * 80)
    print(f"{'Distribution':<15} {'Mean':<8} {'Std':<8} {'Skewness':<10} {'Kurtosis':<10} {'Symmetric?'}")
    print("-" * 80)
    
    for name, data in distributions.items():
        mean = np.mean(data)
        std = np.std(data)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        is_symmetric = "Yes" if abs(skewness) < 0.5 else "No"
        
        print(f"{name:<15} {mean:<8.2f} {std:<8.2f} {skewness:<10.3f} "
              f"{kurtosis:<10.3f} {is_symmetric}")
    
    # Test for normality
    print(f"\nNormality Tests (Shapiro-Wilk p-values):")
    for name, data in distributions.items():
        _, p_value = stats.shapiro(data[:1000])  # Use subset for speed
        is_normal = "Normal" if p_value > 0.05 else "Not Normal"
        print(f"{name:<15}: p = {p_value:.6f} ({is_normal})")

compare_distributions()
```

### 5. Applications in Quality Control
```python
def quality_control_example():
    """
    Demonstrate normal distribution in quality control
    """
    
    # Manufacturing process: bolt lengths should be 10.0 ± 0.1 cm
    target_length = 10.0
    tolerance = 0.1
    process_std = 0.03  # Process standard deviation
    
    # Generate production data
    np.random.seed(42)
    production_data = np.random.normal(target_length, process_std, 10000)
    
    # Calculate process capability
    upper_spec = target_length + tolerance
    lower_spec = target_length - tolerance
    
    # Cp (Process Capability): measures process spread vs specification spread
    cp = (upper_spec - lower_spec) / (6 * process_std)
    
    # Cpk (Process Capability Index): accounts for process centering
    cpu = (upper_spec - np.mean(production_data)) / (3 * process_std)
    cpl = (np.mean(production_data) - lower_spec) / (3 * process_std)
    cpk = min(cpu, cpl)
    
    # Defect rate calculation
    defect_rate = (np.sum(production_data < lower_spec) + 
                   np.sum(production_data > upper_spec)) / len(production_data)
    
    # Theoretical defect rate using normal distribution
    theoretical_defect_rate = (stats.norm.cdf(lower_spec, target_length, process_std) + 
                              (1 - stats.norm.cdf(upper_spec, target_length, process_std)))
    
    print("Quality Control Analysis:")
    print(f"Target length: {target_length} cm")
    print(f"Tolerance: ±{tolerance} cm")
    print(f"Specification limits: {lower_spec} - {upper_spec} cm")
    print(f"Process mean: {np.mean(production_data):.4f} cm")
    print(f"Process std: {np.std(production_data):.4f} cm")
    print(f"\nProcess Capability:")
    print(f"Cp = {cp:.3f} ({'Good' if cp >= 1.33 else 'Needs improvement'})")
    print(f"Cpk = {cpk:.3f} ({'Good' if cpk >= 1.33 else 'Needs improvement'})")
    print(f"\nDefect Analysis:")
    print(f"Observed defect rate: {defect_rate:.4%}")
    print(f"Theoretical defect rate: {theoretical_defect_rate:.4%}")
    
    return production_data, cp, cpk

production_data, cp, cpk = quality_control_example()
```

---

## 🎤 Part 3: Interview Questions & Detailed Answers

### Basic Level Questions

#### Q1: What makes the normal distribution special compared to other distributions?

**Answer:**

The normal distribution has several unique properties that make it fundamental in statistics:

**Key Properties:**
1. **Symmetry**: Perfectly symmetric around the mean
2. **Central tendency**: Mean = Median = Mode
3. **Empirical rule**: 68-95-99.7 rule always applies
4. **Stability**: Sum of normal distributions is normal
5. **Ubiquity**: Many natural phenomena follow normal distribution

```python
def demonstrate_normal_properties():
    """Demonstrate key properties of normal distribution"""
    
    # Generate normal data
    np.random.seed(42)
    data = np.random.normal(100, 15, 10000)
    
    # Property 1: Central tendencies
    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data, keepdims=True).mode[0]
    
    print("Normal Distribution Properties:")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode:.2f}")
    print(f"Difference (Mean-Median): {abs(mean-median):.4f}")
    
    # Property 2: Symmetry (skewness ≈ 0)
    skewness = stats.skew(data)
    print(f"Skewness: {skewness:.4f} (close to 0 = symmetric)")
    
    # Property 3: Empirical rule
    within_1std = np.sum(np.abs(data - mean) <= 15) / len(data)
    within_2std = np.sum(np.abs(data - mean) <= 30) / len(data)
    within_3std = np.sum(np.abs(data - mean) <= 45) / len(data)
    
    print(f"Within 1σ: {within_1std:.1%} (expected 68%)")
    print(f"Within 2σ: {within_2std:.1%} (expected 95%)")
    print(f"Within 3σ: {within_3std:.1%} (expected 99.7%)")

demonstrate_normal_properties()
```

#### Q2: Explain the Central Limit Theorem and its importance.

**Answer:**

**Central Limit Theorem (CLT):** Regardless of the population distribution shape, the sampling distribution of sample means approaches normal distribution as sample size increases (n ≥ 30).

**Key Components:**
- **Population**: Can have any distribution shape
- **Sample size**: n ≥ 30 (rule of thumb)
- **Sampling distribution**: Distribution of sample means
- **Result**: Always approaches normal distribution

```python
def clt_importance_demo():
    """Demonstrate why CLT is important"""
    
    # Start with highly skewed population (exponential)
    np.random.seed(42)
    population = np.random.exponential(2, 100000)
    
    print("Central Limit Theorem Importance:")
    print(f"Population distribution: Exponential (highly skewed)")
    print(f"Population mean: {np.mean(population):.3f}")
    print(f"Population skewness: {stats.skew(population):.3f}")
    
    # Take samples of different sizes
    sample_sizes = [5, 15, 30, 100]
    
    for n in sample_sizes:
        # Generate 1000 sample means
        sample_means = []
        for _ in range(1000):
            sample = np.random.choice(population, n)
            sample_means.append(np.mean(sample))
        
        sample_means = np.array(sample_means)
        
        # Test normality
        _, p_value = stats.shapiro(sample_means[:100])
        
        print(f"\nSample size n={n}:")
        print(f"  Mean of sample means: {np.mean(sample_means):.3f}")
        print(f"  Std of sample means: {np.std(sample_means):.3f}")
        print(f"  Skewness: {stats.skew(sample_means):.3f}")
        print(f"  Normality test p-value: {p_value:.4f}")
        print(f"  Appears normal: {'Yes' if p_value > 0.05 else 'No'}")

clt_importance_demo()
```

**Why CLT is Important:**
1. **Enables inference**: Can use normal distribution properties even with non-normal data
2. **Confidence intervals**: Can construct CIs for any population
3. **Hypothesis testing**: Can perform t-tests and z-tests
4. **Quality control**: Enables statistical process control
5. **Sampling theory**: Foundation for survey sampling

#### Q3: How do you test if data follows a normal distribution?

**Answer:**

Several methods exist to test normality, each with different strengths:

```python
def test_normality(data, alpha=0.05):
    """
    Comprehensive normality testing
    """
    
    print("Normality Testing Results:")
    print("=" * 50)
    
    # 1. Visual methods
    print("1. Visual Assessment:")
    
    # Skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    print(f"   Skewness: {skewness:.4f} (normal ≈ 0)")
    print(f"   Kurtosis: {kurtosis:.4f} (normal ≈ 0)")
    
    # 2. Statistical tests
    print("\n2. Statistical Tests:")
    
    # Shapiro-Wilk test (best for n < 5000)
    if len(data) <= 5000:
        stat_sw, p_sw = stats.shapiro(data)
        print(f"   Shapiro-Wilk: p = {p_sw:.6f} ({'Normal' if p_sw > alpha else 'Not Normal'})")
    
    # Anderson-Darling test
    stat_ad, critical_values, significance_levels = stats.anderson(data, dist='norm')
    print(f"   Anderson-Darling: statistic = {stat_ad:.4f}")
    
    # Kolmogorov-Smirnov test
    # First standardize the data
    standardized = (data - np.mean(data)) / np.std(data)
    stat_ks, p_ks = stats.kstest(standardized, 'norm')
    print(f"   Kolmogorov-Smirnov: p = {p_ks:.6f} ({'Normal' if p_ks > alpha else 'Not Normal'})")
    
    # D'Agostino's normality test
    stat_da, p_da = stats.normaltest(data)
    print(f"   D'Agostino: p = {p_da:.6f} ({'Normal' if p_da > alpha else 'Not Normal'})")
    
    # 3. Rule of thumb checks
    print("\n3. Rule of Thumb:")
    
    # Check if mean ≈ median
    mean_median_diff = abs(np.mean(data) - np.median(data))
    std_dev = np.std(data)
    print(f"   |Mean - Median|/Std: {mean_median_diff/std_dev:.4f} (normal < 0.1)")
    
    # Check empirical rule
    within_1std = np.sum(np.abs(data - np.mean(data)) <= std_dev) / len(data)
    print(f"   Within 1σ: {within_1std:.1%} (normal ≈ 68%)")
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_p': p_sw if len(data) <= 5000 else None,
        'ks_p': p_ks,
        'dagostino_p': p_da
    }

# Test with different data types
print("=== Testing Normal Data ===")
normal_data = np.random.normal(50, 10, 1000)
test_normality(normal_data)

print("\n=== Testing Skewed Data ===")
skewed_data = np.random.exponential(2, 1000)
test_normality(skewed_data)
```

### Intermediate Level Questions

#### Q4: How do you handle non-normal data in statistical analysis?

**Answer:**

When data doesn't follow normal distribution, several strategies are available:

```python
def handle_non_normal_data():
    """
    Demonstrate methods for handling non-normal data
    """
    
    # Generate skewed data
    np.random.seed(42)
    skewed_data = np.random.exponential(2, 1000)
    
    print("Handling Non-Normal Data:")
    print("=" * 40)
    print(f"Original data skewness: {stats.skew(skewed_data):.3f}")
    
    # Method 1: Transformations
    print("\n1. Data Transformations:")
    
    # Log transformation
    log_transformed = np.log(skewed_data)
    print(f"   Log transform skewness: {stats.skew(log_transformed):.3f}")
    
    # Square root transformation
    sqrt_transformed = np.sqrt(skewed_data)
    print(f"   Sqrt transform skewness: {stats.skew(sqrt_transformed):.3f}")
    
    # Box-Cox transformation
    from scipy.stats import boxcox
    boxcox_transformed, lambda_param = boxcox(skewed_data)
    print(f"   Box-Cox transform skewness: {stats.skew(boxcox_transformed):.3f}")
    print(f"   Optimal λ parameter: {lambda_param:.3f}")
    
    # Method 2: Non-parametric tests
    print("\n2. Non-Parametric Alternatives:")
    
    # Instead of t-test, use Wilcoxon signed-rank test
    # Instead of ANOVA, use Kruskal-Wallis test
    # Instead of Pearson correlation, use Spearman correlation
    
    # Example: Compare two groups
    group1 = np.random.exponential(2, 100)
    group2 = np.random.exponential(2.5, 100)
    
    # Parametric test (assumes normality)
    t_stat, t_p = stats.ttest_ind(group1, group2)
    
    # Non-parametric alternative
    u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    print(f"   t-test p-value: {t_p:.4f}")
    print(f"   Mann-Whitney U p-value: {u_p:.4f}")
    
    # Method 3: Robust statistics
    print("\n3. Robust Statistics:")
    
    # Use median instead of mean
    # Use IQR instead of standard deviation
    # Use trimmed mean
    
    from scipy.stats import trim_mean
    
    print(f"   Mean: {np.mean(skewed_data):.3f}")
    print(f"   Median: {np.median(skewed_data):.3f}")
    print(f"   Trimmed mean (10%): {trim_mean(skewed_data, 0.1):.3f}")
    print(f"   Standard deviation: {np.std(skewed_data):.3f}")
    print(f"   IQR: {np.percentile(skewed_data, 75) - np.percentile(skewed_data, 25):.3f}")
    
    return {
        'original': skewed_data,
        'log_transformed': log_transformed,
        'boxcox_transformed': boxcox_transformed
    }

transformations = handle_non_normal_data()
```

### Advanced Level Questions

#### Q5: Implement and explain the Box-Cox transformation for normalizing data.

**Answer:**

**Box-Cox Transformation** is a power transformation that can make non-normal data more normal-like:

```python
class BoxCoxTransformer:
    def __init__(self):
        self.lambda_param = None
        self.shift_param = 0
    
    def fit(self, data):
        """
        Find optimal lambda parameter for Box-Cox transformation
        
        Box-Cox formula:
        - If λ ≠ 0: y(λ) = (x^λ - 1) / λ
        - If λ = 0: y(λ) = ln(x)
        """
        
        # Ensure all values are positive
        if np.min(data) <= 0:
            self.shift_param = abs(np.min(data)) + 1
            data_shifted = data + self.shift_param
        else:
            data_shifted = data
        
        # Find optimal lambda using maximum likelihood
        from scipy.stats import boxcox
        self.transformed_data, self.lambda_param = boxcox(data_shifted)
        
        print(f"Box-Cox Transformation Results:")
        print(f"Shift parameter: {self.shift_param}")
        print(f"Optimal λ: {self.lambda_param:.4f}")
        
        # Interpret lambda
        if abs(self.lambda_param) < 0.1:
            interpretation = "≈ Log transformation"
        elif abs(self.lambda_param - 0.5) < 0.1:
            interpretation = "≈ Square root transformation"
        elif abs(self.lambda_param - 1) < 0.1:
            interpretation = "≈ No transformation needed"
        elif abs(self.lambda_param - 2) < 0.1:
            interpretation = "≈ Square transformation"
        else:
            interpretation = f"Power transformation (λ={self.lambda_param:.3f})"
        
        print(f"Interpretation: {interpretation}")
        
        return self
    
    def transform(self, data):
        """Apply Box-Cox transformation"""
        if self.lambda_param is None:
            raise ValueError("Must fit transformer first")
        
        # Apply shift if needed
        data_shifted = data + self.shift_param
        
        # Apply Box-Cox transformation
        if abs(self.lambda_param) < 1e-6:  # λ ≈ 0
            return np.log(data_shifted)
        else:
            return (np.power(data_shifted, self.lambda_param) - 1) / self.lambda_param
    
    def inverse_transform(self, transformed_data):
        """Reverse Box-Cox transformation"""
        if self.lambda_param is None:
            raise ValueError("Must fit transformer first")
        
        # Reverse Box-Cox
        if abs(self.lambda_param) < 1e-6:  # λ ≈ 0
            original = np.exp(transformed_data)
        else:
            original = np.power(self.lambda_param * transformed_data + 1, 1/self.lambda_param)
        
        # Remove shift
        return original - self.shift_param
    
    def evaluate_transformation(self, original_data, transformed_data):
        """Evaluate effectiveness of transformation"""
        
        # Calculate normality metrics before and after
        orig_skew = stats.skew(original_data)
        trans_skew = stats.skew(transformed_data)
        
        orig_kurt = stats.kurtosis(original_data)
        trans_kurt = stats.kurtosis(transformed_data)
        
        # Normality tests
        _, orig_p = stats.shapiro(original_data[:1000])  # Limit for speed
        _, trans_p = stats.shapiro(transformed_data[:1000])
        
        print(f"\nTransformation Evaluation:")
        print(f"{'Metric':<15} {'Original':<10} {'Transformed':<12} {'Improvement'}")
        print("-" * 50)
        print(f"{'Skewness':<15} {orig_skew:<10.4f} {trans_skew:<12.4f} {abs(orig_skew) > abs(trans_skew)}")
        print(f"{'Kurtosis':<15} {orig_kurt:<10.4f} {trans_kurt:<12.4f} {abs(orig_kurt) > abs(trans_kurt)}")
        print(f"{'Shapiro p-val':<15} {orig_p:<10.6f} {trans_p:<12.6f} {trans_p > orig_p}")
        
        return {
            'original_skewness': orig_skew,
            'transformed_skewness': trans_skew,
            'original_shapiro_p': orig_p,
            'transformed_shapiro_p': trans_p
        }

# Example usage
def demonstrate_boxcox():
    """Demonstrate Box-Cox transformation on different data types"""
    
    np.random.seed(42)
    
    # Test on different distributions
    datasets = {
        'Exponential': np.random.exponential(2, 1000),
        'Lognormal': np.random.lognormal(1, 0.5, 1000),
        'Gamma': np.random.gamma(2, 2, 1000)
    }
    
    for name, data in datasets.items():
        print(f"\n{'='*20} {name} Data {'='*20}")
        
        transformer = BoxCoxTransformer()
        transformer.fit(data)
        
        transformed = transformer.transform(data)
        transformer.evaluate_transformation(data, transformed)
        
        # Test inverse transformation
        reconstructed = transformer.inverse_transform(transformed)
        reconstruction_error = np.mean(np.abs(data - reconstructed))
        print(f"Reconstruction error: {reconstruction_error:.6f}")

demonstrate_boxcox()
```

---

## 🚀 Practical Tips for Interviews

### 1. **Know the Key Properties**
```python
# Quick reference for normal distribution
normal_properties = {
    "Shape": "Bell-shaped, symmetric",
    "Parameters": "μ (mean), σ (standard deviation)",
    "Empirical Rule": "68-95-99.7 within 1-2-3 σ",
    "Central Tendencies": "Mean = Median = Mode",
    "Standardization": "Z = (X - μ) / σ"
}
```

### 2. **Understand When to Use Normal Distribution**
- Large sample sizes (CLT applies)
- Naturally occurring measurements (height, weight, IQ)
- Errors and residuals in regression
- Quality control and manufacturing
- Financial returns (approximately)

### 3. **Know the Limitations**
- Real data is rarely perfectly normal
- Sensitive to outliers
- Assumes continuous data
- May not work for bounded data
- Skewed data needs transformation

### 4. **Master the Applications**
```python
# Common applications in interviews
applications = {
    "Hypothesis Testing": "t-tests, z-tests assume normality",
    "Confidence Intervals": "Based on normal distribution",
    "Quality Control": "Control charts use normal distribution",
    "Risk Management": "VaR calculations in finance",
    "A/B Testing": "Statistical significance testing"
}
```

---

## 📚 Key Concepts from the Meeting

### 1. **Normal Distribution Properties:**
- Bell-shaped and symmetric
- Defined by mean (μ) and standard deviation (σ)
- Empirical rule: 68-95-99.7
- Most widely used distribution in statistics

### 2. **Central Limit Theorem:**
- Sample means approach normal distribution
- Works regardless of population distribution
- Sample size ≥ 30 rule of thumb
- Foundation for statistical inference

### 3. **Practical Applications:**
- Quality control in manufacturing
- Hypothesis testing and confidence intervals
- Risk assessment and modeling
- Data transformation techniques

---

*Remember: Normal distribution interviews test both theoretical understanding and practical applications. Focus on understanding when and why to use normal distribution, and be prepared to discuss alternatives for non-normal data!* 🎯