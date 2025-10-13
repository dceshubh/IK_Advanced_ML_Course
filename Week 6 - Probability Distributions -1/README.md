# Week 6 - Probability Distributions 1

## 📋 Summary

This folder contains comprehensive learning materials for **Probability Distributions 1**, covering discrete probability distributions and their applications in data science and machine learning.

---

## 📁 Files Overview

### 📓 Coding Notebooks

1. **Copy_of_PD1_Code.ipynb** - Live Class Notebook
   - Hospital operations case study
   - Hands-on implementation of discrete distributions
   - Visualization examples
   - Real-world scenarios

2. **PD1_Assignment_Solution.ipynb** - Assignment Solutions
   - Call center analysis (Poisson)
   - Quality control (Binomial)
   - Customer arrivals (Poisson)
   - Dice rolling (Uniform)

### 📚 Study Materials

3. **Probability_Distributions_1_Study_Guide.md** - Comprehensive Study Guide
   - Concepts explained for beginners
   - Technical definitions
   - Real-world applications
   - Key takeaways and comparisons
   - ✅ **Status: UP TO DATE**

4. **CODING_GUIDE_Live_Class.md** - Live Class Coding Guide
   - Detailed code explanations
   - Step-by-step breakdowns
   - Function arguments and syntax
   - Business insights
   - ✅ **Status: UP TO DATE**

5. **CODING_GUIDE_Assignment.md** - Assignment Coding Guide
   - Problem-by-problem solutions
   - Code explanations
   - Visualization techniques
   - Common mistakes to avoid
   - ✅ **Status: UP TO DATE**

---

## 🎯 Topics Covered

### 1. Bernoulli Distribution
- Single trial with two outcomes
- Patient appointment attendance
- Success/failure scenarios

### 2. Binomial Distribution
- Multiple independent trials
- Surgery success rates
- Quality control applications

### 3. Poisson Distribution
- Events over time/space
- Emergency room arrivals
- Call center analysis

### 4. Discrete Uniform Distribution
- Equally likely outcomes
- Random room assignments
- Fair dice rolling

---

## 🔧 Key Python Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, randint
```

---

## 📊 Distribution Quick Reference

| Distribution | Use Case | Parameters | Example |
|--------------|----------|------------|---------|
| **Bernoulli** | Single yes/no | p | Will patient show up? |
| **Binomial** | Count successes | n, p | Successful surgeries out of 20 |
| **Poisson** | Events in interval | λ | Emergency arrivals per hour |
| **Uniform** | Equal probability | a, b | Random room assignment |

---

## ✅ Verification Status

All materials have been reviewed and verified:

- ✅ **Live Class Coding Guide**: Comprehensive and up to date
- ✅ **Assignment Coding Guide**: Complete with all problems solved
- ✅ **Study Guide**: Thorough coverage of all concepts
- ✅ **No meeting transcripts found**: Study materials created from notebooks

---

## 🎓 Learning Path

1. **Start with**: `Probability_Distributions_1_Study_Guide.md`
   - Understand concepts from basics to advanced
   - Learn when to use each distribution

2. **Practice with**: `Copy_of_PD1_Code.ipynb` + `CODING_GUIDE_Live_Class.md`
   - Follow along with hospital case study
   - Understand code implementation

3. **Apply knowledge**: `PD1_Assignment_Solution.ipynb` + `CODING_GUIDE_Assignment.md`
   - Solve real-world problems
   - Practice calculations and visualizations

---

## 💡 Key Learning Outcomes

After completing this week, you should be able to:

- ✅ Identify which distribution to use for different scenarios
- ✅ Calculate probabilities using PMF and CDF
- ✅ Implement distributions in Python using scipy.stats
- ✅ Visualize probability distributions
- ✅ Apply distributions to real-world business problems
- ✅ Interpret results and make data-driven decisions

---

## 🚀 Next Steps

- Practice with different parameters
- Try creating your own scenarios
- Explore continuous distributions in Week 7
- Apply these concepts to your own datasets

---

**Note**: All coding guides are designed for users who know Python basics but are new to probability distributions and data science applications.
