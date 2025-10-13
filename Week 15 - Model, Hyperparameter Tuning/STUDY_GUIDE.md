# Week 15: Model Evaluation & Hyperparameter Tuning - Study Guide

## 📚 Overview
This week covers essential techniques for evaluating machine learning models and optimizing their performance through hyperparameter tuning. You'll learn how to assess model quality beyond simple accuracy, handle imbalanced datasets, and interpret model predictions.

## 🎯 Learning Objectives
By the end of this week, you should be able to:
- Evaluate models using multiple metrics appropriate for imbalanced datasets
- Apply techniques to handle class imbalance (SMOTE, class weights)
- Perform hyperparameter tuning using Grid Search and Random Search
- Interpret model predictions using LIME and SHAP
- Understand overfitting, underfitting, and model generalization
- Use cross-validation for robust model evaluation

---

## 📖 Core Concepts

### 1. Model Evaluation Metrics

#### 1.1 Beyond Accuracy
**Why Accuracy Isn't Enough:**
- In imbalanced datasets (e.g., 95% class A, 5% class B), a model predicting all samples as class A achieves 95% accuracy but is useless
- Need metrics that capture performance on minority class

**Key Metrics:**

**Confusion Matrix:**
```
                Predicted
                Pos    Neg
Actual  Pos     TP     FN
        Neg     FP     TN
```

**Precision:** TP / (TP + FP)
- "Of all positive predictions, how many were correct?"
- Important when false positives are costly (e.g., spam detection)

**Recall (Sensitivity):** TP / (TP + FN)
- "Of all actual positives, how many did we catch?"
- Important when false negatives are costly (e.g., disease detection)

**F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Good for imbalanced datasets

**Specificity:** TN / (TN + FP)
- "Of all actual negatives, how many did we correctly identify?"

**ROC-AUC Score:**
- Area Under the Receiver Operating Characteristic curve
- Measures model's ability to distinguish between classes
- Range: 0.5 (random) to 1.0 (perfect)

#### 1.2 Classification Report
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```
Provides precision, recall, F1-score for each class plus averages.

---

### 2. Handling Imbalanced Data

#### 2.1 Why Imbalance Matters
- Models tend to bias toward majority class
- Minority class often more important (fraud, disease, churn)
- Standard metrics misleading

#### 2.2 Techniques

**A. SMOTE (Synthetic Minority Over-sampling Technique)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**How SMOTE Works:**
1. For each minority sample, find k nearest neighbors (default k=5)
2. Randomly select one neighbor
3. Create synthetic sample along line between sample and neighbor
4. Repeat until classes balanced

**Advantages:**
- Creates synthetic samples, not duplicates
- Reduces overfitting compared to random oversampling
- Improves minority class representation

**Disadvantages:**
- Can create noisy samples in overlapping regions
- Computationally expensive for large datasets
- May not work well with high-dimensional data

**B. Class Weights**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
```
- Penalizes misclassification of minority class more heavily
- No data modification needed
- Works with most sklearn classifiers

**C. Other Techniques:**
- **Random Undersampling:** Remove majority class samples
- **Random Oversampling:** Duplicate minority class samples
- **ADASYN:** Adaptive Synthetic Sampling
- **Tomek Links:** Remove borderline majority samples

---

### 3. Hyperparameter Tuning

#### 3.1 What Are Hyperparameters?
**Parameters:** Learned from data (e.g., weights in neural networks)
**Hyperparameters:** Set before training (e.g., learning rate, tree depth)

**Common Hyperparameters by Algorithm:**

**Logistic Regression:**
- `C`: Inverse regularization strength
- `penalty`: L1 or L2 regularization
- `solver`: Optimization algorithm

**Random Forest:**
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split node
- `max_features`: Features to consider for split

**XGBoost:**
- `learning_rate`: Step size shrinkage
- `max_depth`: Maximum tree depth
- `n_estimators`: Number of boosting rounds
- `subsample`: Fraction of samples for training
- `colsample_bytree`: Fraction of features for training

#### 3.2 Grid Search
**Exhaustive search over specified parameter grid:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**Advantages:**
- Guaranteed to find best combination in grid
- Comprehensive evaluation

**Disadvantages:**
- Computationally expensive (tries all combinations)
- Doesn't scale well with many parameters
- May miss optimal values between grid points

#### 3.3 Random Search
**Randomly samples from parameter distributions:**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42
)

random_search.fit(X_train, y_train)
```

**Advantages:**
- More efficient than grid search
- Can explore wider parameter space
- Good for continuous parameters

**Disadvantages:**
- No guarantee of finding optimal combination
- Results vary with random seed

#### 3.4 Best Practices
1. **Start with Random Search** for broad exploration
2. **Refine with Grid Search** around promising regions
3. **Use Cross-Validation** to avoid overfitting
4. **Choose appropriate scoring metric** for your problem
5. **Consider computational budget** (time, resources)

---

### 4. Cross-Validation

#### 4.1 Why Cross-Validation?
- Single train-test split may not be representative
- Results can vary significantly with different splits
- CV provides more robust performance estimate

#### 4.2 K-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='f1'
)

print(f"Mean F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Process:**
1. Split data into K folds
2. For each fold:
   - Train on K-1 folds
   - Validate on remaining fold
3. Average results across all folds

**Common K values:**
- K=5: Good balance of bias-variance
- K=10: More robust but computationally expensive
- K=n (LOOCV): Maximum data usage but very expensive

#### 4.3 Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
- Maintains class distribution in each fold
- Essential for imbalanced datasets

---

### 5. Model Interpretability

#### 5.1 Why Interpretability Matters
- Build trust in model predictions
- Identify potential biases
- Debug model behavior
- Meet regulatory requirements
- Gain domain insights

#### 5.2 LIME (Local Interpretable Model-agnostic Explanations)

**Concept:**
- Explains individual predictions
- Creates local linear approximation around prediction
- Shows which features contributed most

```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['No', 'Yes'],
    mode='classification'
)

# Explain a single prediction
exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba
)

exp.show_in_notebook()
```

**How LIME Works:**
1. Perturb input sample (create variations)
2. Get model predictions for perturbed samples
3. Weight samples by proximity to original
4. Fit simple linear model to weighted samples
5. Use linear model coefficients as explanations

**Advantages:**
- Model-agnostic (works with any model)
- Intuitive explanations
- Identifies important features for specific prediction

**Limitations:**
- Only local explanations (one prediction at a time)
- Can be unstable (different runs give different results)
- Computationally expensive

#### 5.3 SHAP (SHapley Additive exPlanations)

**Concept:**
- Based on game theory (Shapley values)
- Shows contribution of each feature to prediction
- Provides both local and global explanations

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Visualize single prediction
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test.iloc[0]
)

# Summary plot (global importance)
shap.summary_plot(shap_values[1], X_test)
```

**SHAP Visualizations:**

**Force Plot:** Shows how features push prediction from base value
**Summary Plot:** Feature importance across all samples
**Dependence Plot:** Relationship between feature and SHAP value
**Waterfall Plot:** Cumulative feature contributions

**Advantages:**
- Theoretically sound (based on Shapley values)
- Consistent and locally accurate
- Both local and global explanations
- Multiple visualization options

**Disadvantages:**
- Computationally expensive
- Can be slow for large datasets
- Requires understanding of Shapley values

**LIME vs SHAP:**
| Aspect | LIME | SHAP |
|--------|------|------|
| Theory | Local linear approximation | Game theory (Shapley values) |
| Speed | Faster | Slower |
| Consistency | Less consistent | More consistent |
| Scope | Local only | Local + Global |
| Interpretability | Very intuitive | Requires more understanding |

---

### 6. Overfitting and Underfitting

#### 6.1 Definitions

**Underfitting:**
- Model too simple to capture data patterns
- High training error, high test error
- High bias, low variance

**Overfitting:**
- Model too complex, memorizes training data
- Low training error, high test error
- Low bias, high variance

**Good Fit:**
- Captures true patterns without memorizing noise
- Similar training and test error
- Balanced bias-variance tradeoff

#### 6.2 Detecting Over/Underfitting

**Learning Curves:**
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.legend()
```

**Interpretation:**
- **Underfitting:** Both curves low, converging
- **Overfitting:** Large gap between curves
- **Good fit:** Curves close, both high

#### 6.3 Solutions

**For Overfitting:**
- Increase training data
- Reduce model complexity
- Add regularization (L1, L2, dropout)
- Use cross-validation
- Early stopping
- Feature selection
- Ensemble methods

**For Underfitting:**
- Increase model complexity
- Add more features
- Reduce regularization
- Train longer
- Use more sophisticated model

---

## 💡 Practical Tips

### Model Evaluation
1. **Always use multiple metrics** - don't rely on accuracy alone
2. **Understand your problem** - is false positive or false negative worse?
3. **Use stratified splits** for imbalanced data
4. **Check confusion matrix** to understand error types
5. **Plot ROC curves** to visualize performance across thresholds

### Handling Imbalance
1. **Try multiple techniques** - SMOTE, class weights, undersampling
2. **Apply SMOTE only to training data** - never to test data
3. **Use stratified sampling** when splitting data
4. **Consider ensemble methods** - often handle imbalance well
5. **Evaluate with appropriate metrics** - F1, ROC-AUC, not accuracy

### Hyperparameter Tuning
1. **Start simple** - baseline model first
2. **Use domain knowledge** - reasonable parameter ranges
3. **Random search first** - then grid search if needed
4. **Monitor for overfitting** - check validation scores
5. **Save best models** - for later comparison

### Model Interpretability
1. **Start with simple models** - easier to interpret
2. **Use multiple explanation methods** - LIME and SHAP
3. **Validate explanations** - do they make domain sense?
4. **Explain to stakeholders** - build trust in model
5. **Document findings** - for reproducibility

---

## 🔍 Common Pitfalls

### 1. Data Leakage
**Problem:** Information from test set leaks into training
**Examples:**
- Applying SMOTE before train-test split
- Scaling on entire dataset before splitting
- Using future information in features

**Solution:** Always split first, then preprocess

### 2. Overfitting to Validation Set
**Problem:** Tuning hyperparameters based on validation performance
**Solution:** Use separate test set for final evaluation

### 3. Ignoring Class Imbalance
**Problem:** Using accuracy on imbalanced data
**Solution:** Use F1, ROC-AUC, or balanced accuracy

### 4. Not Using Cross-Validation
**Problem:** Single split may not be representative
**Solution:** Use k-fold CV for robust estimates

### 5. Computational Waste
**Problem:** Grid search with too many parameters
**Solution:** Start with random search, narrow down

---

## 📊 Real-World Applications

### 1. Customer Churn Prediction
- **Challenge:** Imbalanced (few churners)
- **Metrics:** Recall (catch churners), F1-score
- **Techniques:** SMOTE, class weights
- **Interpretation:** SHAP to identify churn drivers

### 2. Fraud Detection
- **Challenge:** Highly imbalanced (rare fraud)
- **Metrics:** Precision (avoid false alarms), ROC-AUC
- **Techniques:** Anomaly detection, ensemble methods
- **Interpretation:** LIME for individual cases

### 3. Medical Diagnosis
- **Challenge:** High cost of false negatives
- **Metrics:** Recall (catch all diseases), specificity
- **Techniques:** Careful threshold tuning
- **Interpretation:** SHAP for doctor trust

### 4. Credit Scoring
- **Challenge:** Regulatory requirements
- **Metrics:** Balanced accuracy, fairness metrics
- **Techniques:** Interpretable models, bias detection
- **Interpretation:** LIME for loan decisions

---

## 🎓 Interview Questions

### Beginner Level

**Q1: What's wrong with using accuracy for imbalanced datasets?**
A: Accuracy can be misleading. If 95% of samples are class A, a model predicting all samples as A achieves 95% accuracy but is useless. Better metrics: F1-score, ROC-AUC, precision/recall.

**Q2: What is SMOTE and when would you use it?**
A: SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic samples for minority class by interpolating between existing samples. Use it when you have imbalanced data and want to improve minority class performance.

**Q3: What's the difference between parameters and hyperparameters?**
A: Parameters are learned from data during training (e.g., weights). Hyperparameters are set before training and control the learning process (e.g., learning rate, tree depth).

**Q4: What is cross-validation and why use it?**
A: Cross-validation splits data into K folds, trains on K-1 folds and validates on the remaining fold, repeating K times. It provides more robust performance estimates than a single train-test split.

**Q5: What does ROC-AUC measure?**
A: ROC-AUC (Area Under Receiver Operating Characteristic curve) measures a model's ability to distinguish between classes across all classification thresholds. Range: 0.5 (random) to 1.0 (perfect).

### Intermediate Level

**Q6: Explain precision vs recall tradeoff.**
A: Precision = TP/(TP+FP), Recall = TP/(TP+FN). Increasing classification threshold increases precision but decreases recall. Choose based on cost of false positives vs false negatives. F1-score balances both.

**Q7: How does Grid Search differ from Random Search?**
A: Grid Search exhaustively tries all parameter combinations (guaranteed to find best in grid but expensive). Random Search samples randomly from parameter distributions (more efficient, explores wider space, but no guarantee of optimal).

**Q8: What is stratified k-fold and when is it important?**
A: Stratified k-fold maintains class distribution in each fold. Essential for imbalanced datasets to ensure each fold has representative samples of all classes.

**Q9: How do you detect overfitting?**
A: Signs: Large gap between training and validation performance, high variance in CV scores, learning curves diverging. Solutions: More data, regularization, simpler model, cross-validation.

**Q10: What's the difference between L1 and L2 regularization?**
A: L1 (Lasso) adds sum of absolute weights, promotes sparsity (feature selection). L2 (Ridge) adds sum of squared weights, shrinks all weights proportionally. L1 for feature selection, L2 for general regularization.

### Advanced Level

**Q11: Explain how SHAP values work.**
A: SHAP uses Shapley values from game theory. For each feature, it calculates average marginal contribution across all possible feature combinations. Provides consistent, theoretically sound feature attributions for both local and global explanations.

**Q12: How would you handle a dataset with 99.9% majority class?**
A: Multiple approaches: (1) Anomaly detection instead of classification, (2) Ensemble methods with different sampling, (3) Cost-sensitive learning with high penalty for minority misclassification, (4) Collect more minority samples if possible, (5) Use appropriate metrics (precision-recall curve, not ROC).

**Q13: What are the limitations of LIME?**
A: (1) Instability - different runs give different explanations, (2) Only local explanations, (3) Sampling-based so computationally expensive, (4) Linear approximation may not capture complex relationships, (5) Choice of perturbation strategy affects results.

**Q14: How do you choose between Grid Search and Bayesian Optimization?**
A: Grid/Random Search: Simple, parallelizable, good for initial exploration. Bayesian Optimization: More efficient for expensive models, uses previous results to guide search, better for continuous parameters. Use Bayesian for deep learning or large datasets where training is expensive.

**Q15: Explain the bias-variance tradeoff in model selection.**
A: Bias: Error from wrong assumptions (underfitting). Variance: Error from sensitivity to training data (overfitting). Simple models: high bias, low variance. Complex models: low bias, high variance. Goal: Find sweet spot minimizing total error = bias² + variance + irreducible error.

---

## 🛠️ Code Templates

### Complete Model Evaluation Pipeline
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Handle imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 3. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# 4. Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 5. Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 7. Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"\nCV F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

### Hyperparameter Tuning Template
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

# Use best model
best_model = random_search.best_estimator_
```

### Model Interpretation Template
```python
import shap
import lime
import lime.lime_tabular

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance
shap.summary_plot(shap_values[1], X_test)

# Single prediction
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test.iloc[0]
)

# LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['No', 'Yes'],
    mode='classification'
)

exp = lime_explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

exp.show_in_notebook()
```

---

## 📚 Additional Resources

### Documentation
- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)

### Papers
- SMOTE: "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)
- SHAP: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
- LIME: "Why Should I Trust You?" (Ribeiro et al., 2016)

### Books
- "Interpretable Machine Learning" by Christoph Molnar
- "Hands-On Machine Learning" by Aurélien Géron (Chapters on Model Evaluation)

---

## ✅ Key Takeaways

1. **Accuracy is not enough** - use multiple metrics appropriate for your problem
2. **Handle imbalanced data** - SMOTE, class weights, or appropriate metrics
3. **Tune hyperparameters systematically** - random search then grid search
4. **Use cross-validation** - for robust performance estimates
5. **Interpret your models** - LIME and SHAP for trust and insights
6. **Watch for overfitting** - learning curves and validation scores
7. **Choose metrics wisely** - based on business costs of errors
8. **Always split before preprocessing** - avoid data leakage
9. **Document your process** - for reproducibility and communication
10. **Iterate and experiment** - model building is iterative

---

*This study guide covers the essential concepts for Week 15. Practice with real datasets and refer to the coding guides for hands-on implementation.*
