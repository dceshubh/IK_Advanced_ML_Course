# Week 14: Bagging & Boosting - Live Class Coding Guide

## 📓 Notebook: Uplevel_Bagging_&_Boosting_Live_Class_Notebook.ipynb

## 🎯 Objective
Learn ensemble methods (Bagging and Boosting) to improve model performance by combining multiple weak learners into a strong learner. Apply to Wisconsin Breast Cancer classification.

---

## 📊 Dataset: Wisconsin Breast Cancer

**Business Context:**
- Medical diagnosis: Malignant (M) vs Benign (B) tumors
- Features from digitized images of breast mass cells
- 30 features: radius, texture, perimeter, area, smoothness, etc.
- Each feature: mean, standard error, worst value

**Challenge:**
- Life-saving decisions require high accuracy
- Need reliable, interpretable predictions
- Minimize false negatives (missing cancer)

---

## 🔧 Section 1: Data Preparation

### Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

pd.set_option('display.max_columns', 100)
```

### Load and Explore Data
```python
# Load dataset
df = pd.read_csv('data.csv')

# Basic exploration
print(df.shape)
print(df.head())
print(df.info())
print(df['diagnosis'].value_counts())
```

**Key Checks:**
- Shape: (569, 32) - 569 samples, 32 columns
- Target: diagnosis (M/B)
- Features: 30 numeric features
- No missing values

### Preprocessing
```python
# Encode target variable
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Class distribution:\n{y_train.value_counts(normalize=True)}")
```

---

## 🔧 Section 2: Baseline Model (Single Decision Tree)

### Train Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

# Train baseline
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predictions
y_pred_dt = dt.predict(X_test)

# Evaluate
print("Decision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))
```

**Expected Results:**
- Accuracy: ~93-95%
- Good but can be improved
- Single tree prone to overfitting

---

## 🔧 Section 3: Bagging (Bootstrap Aggregating)

### Concept
**What is Bagging?**
- Train multiple models on different subsets of data
- Each subset: random sample with replacement (bootstrap)
- Combine predictions by voting (classification) or averaging (regression)
- Reduces variance, prevents overfitting

**Random Forest = Bagging + Random Feature Selection**

### Train Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Evaluate
print("Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
```

**Parameters:**
- `n_estimators=100`: Number of trees
- `max_depth=10`: Maximum tree depth
- `random_state=42`: Reproducibility

**Expected Improvement:**
- Accuracy: ~96-98%
- More stable than single tree
- Better generalization

### Feature Importance
```python
# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 10
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Important Features (Random Forest)')
plt.xlabel('Importance')
plt.show()
```

**Interpretation:**
- Identifies most predictive features
- Helps understand model decisions
- Can guide feature selection

---

## 🔧 Section 4: Boosting (AdaBoost)

### Concept
**What is Boosting?**
- Train models sequentially
- Each model focuses on mistakes of previous models
- Weighted combination of weak learners
- Reduces bias, improves accuracy

**AdaBoost (Adaptive Boosting):**
- Increases weight of misclassified samples
- Next model pays more attention to hard cases
- Final prediction: weighted vote

### Train AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier

# Train AdaBoost
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)

# Predictions
y_pred_ada = ada.predict(X_test)

# Evaluate
print("AdaBoost Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_ada):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_ada):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_ada):.4f}")
```

**Parameters:**
- `n_estimators=100`: Number of weak learners
- `learning_rate=1.0`: Weight of each classifier
- Lower learning_rate = more conservative

---

## 🔧 Section 5: Gradient Boosting

### Concept
**Gradient Boosting:**
- Builds trees sequentially
- Each tree corrects errors of previous ensemble
- Uses gradient descent to minimize loss
- More powerful than AdaBoost

### Train Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

# Train Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

# Predictions
y_pred_gb = gb.predict(X_test)

# Evaluate
print("Gradient Boosting Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_gb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_gb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_gb):.4f}")
```

**Parameters:**
- `learning_rate=0.1`: Shrinkage parameter
- `max_depth=3`: Shallow trees (weak learners)
- Lower learning_rate + more estimators = better performance

---

## 🔧 Section 6: XGBoost

### Concept
**XGBoost (Extreme Gradient Boosting):**
- Optimized gradient boosting
- Faster, more efficient
- Built-in regularization
- Handles missing values
- Often wins Kaggle competitions

### Train XGBoost
```python
from xgboost import XGBClassifier

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb.predict(X_test)

# Evaluate
print("XGBoost Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
```

**Advantages:**
- State-of-the-art performance
- Fast training
- Regularization prevents overfitting
- Feature importance built-in

---

## 🔧 Section 7: Model Comparison

### Compare All Models
```python
# Create comparison DataFrame
models = {
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf,
    'AdaBoost': y_pred_ada,
    'Gradient Boosting': y_pred_gb,
    'XGBoost': y_pred_xgb
}

results = []
for name, preds in models.items():
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds)
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('Accuracy', ascending=False))
```

### Visualize Comparison
```python
# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    sns.barplot(data=results_df, x='Model', y=metric, ax=ax)
    ax.set_title(f'{metric} Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylim(0.9, 1.0)

plt.tight_layout()
plt.show()
```

---

## 💡 Key Takeaways

### Bagging vs Boosting

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Sampling | Bootstrap (with replacement) | Weighted samples |
| Combination | Equal weight voting | Weighted voting |
| Example | Random Forest | AdaBoost, XGBoost |
| Best For | High variance models | High bias models |

### When to Use Each

**Random Forest (Bagging):**
- Good default choice
- Handles high-dimensional data
- Less prone to overfitting
- Parallel training (faster)

**Gradient Boosting/XGBoost:**
- Maximum performance needed
- Willing to tune hyperparameters
- Have computational resources
- Kaggle competitions

**AdaBoost:**
- Simple, interpretable
- Works with any weak learner
- Good for binary classification

---

## 🚨 Common Mistakes

1. **Not tuning hyperparameters**
   - Default parameters rarely optimal
   - Use GridSearchCV or RandomizedSearchCV

2. **Overfitting with boosting**
   - Too many estimators
   - Learning rate too high
   - Trees too deep

3. **Ignoring class imbalance**
   - Use `class_weight='balanced'`
   - Or SMOTE for resampling

4. **Not scaling features**
   - Tree-based methods don't need scaling
   - But good practice for consistency

5. **Forgetting cross-validation**
   - Single train-test split unreliable
   - Use k-fold CV for robust estimates

---

*This coding guide covers ensemble methods for the live class notebook. Practice with different datasets and hyperparameters!*
