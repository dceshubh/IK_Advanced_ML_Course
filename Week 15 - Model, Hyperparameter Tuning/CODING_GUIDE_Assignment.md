# Week 15: Bank Term Deposit Assignment - Coding Guide

## 📓 Notebook: Model_evaluation_Assignment_Solution.ipynb

## 🎯 Objective
Build a predictive model for XYZ Bank to identify customers likely to subscribe to term deposits, focusing on model evaluation, hyperparameter tuning, and handling imbalanced data.

---

## 📊 Dataset: Bank Marketing Campaign

**Business Context:**
Banks run marketing campaigns to encourage customers to subscribe to term deposits. Only a small fraction subscribe, making this an imbalanced classification problem.

**Dataset Features (17 total):**

**Customer Demographics:**
- `age`: Customer's age
- `job`: Type of job
- `marital`: Marital status
- `education`: Education level

**Financial Information:**
- `balance`: Average yearly balance (euros)
- `default`: Has credit in default?
- `housing`: Has housing loan?
- `loan`: Has personal loan?

**Campaign Information:**
- `contact`: Contact communication type
- `day`: Last contact day of month
- `month`: Last contact month
- `duration`: Last contact duration (seconds)
- `campaign`: Number of contacts this campaign
- `pdays`: Days since last contact
- `previous`: Number of contacts before this campaign
- `poutcome`: Outcome of previous campaign

**Target:**
- `y`: Subscribed to term deposit? (yes/no)

---

## 🔧 Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', 30)
```

**Why These Libraries:**
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `sklearn`: Machine learning tools
- `matplotlib/seaborn`: Visualization

---

## 🔧 Step 2: Load and Explore Data

### 2.1 Load Data
```python
# Load data with semicolon separator
data = pd.read_csv('bank-full.csv', sep=';')

# View first few rows
data.head()
```

**Important:** This dataset uses `;` as separator, not comma!


### 2.2 View Columns
```python
# Check column names
data.columns
```

**Expected Output:**
```
Index(['age', 'job', 'marital', 'education', 'default', 'balance', 
       'housing', 'loan', 'contact', 'day', 'month', 'duration', 
       'campaign', 'pdays', 'previous', 'poutcome', 'y'], dtype='object')
```

### 2.3 Check Shape
```python
# Dataset dimensions
data.shape
```

**Expected:** (45211, 17) - 45,211 customers, 17 columns

### 2.4 Data Types
```python
# Check data types and missing values
data.info()
```

**Observations:**
- 7 numeric columns (int64)
- 10 categorical columns (object)
- No missing values (all 45211 non-null)

### 2.5 Statistical Summary
```python
# Summary statistics for numeric columns
data.describe()
```

**Key Insights:**
- Age: 18-95 years, mean ~41
- Balance: -8019 to 102127 euros (negative = overdraft)
- Duration: 0-4918 seconds
- Campaign: 1-63 contacts (mean ~3)
- Pdays: -1 means never contacted before

---

## 🔧 Step 3: Exploratory Data Analysis

### 3.1 Target Distribution
```python
# Check class distribution
print(data['y'].value_counts())
print(data['y'].value_counts(normalize=True))

# Visualize
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='y')
plt.title('Term Deposit Subscription Distribution')
plt.xlabel('Subscribed')
plt.ylabel('Count')
plt.show()
```

**Expected Finding:**
- Imbalanced dataset (e.g., ~88% 'no', ~12% 'yes')
- This is the key challenge!

**Business Implication:**
- Most customers don't subscribe
- Need to identify the 12% who will
- Standard accuracy will be misleading

### 3.2 Numeric Features Distribution
```python
# Distribution of numeric features
numeric_cols = data.select_dtypes(include=['int64']).columns

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(data[col], bins=30, edgecolor='black')
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

**What to Look For:**
- Skewed distributions (may need transformation)
- Outliers (extreme values)
- Different scales (need standardization)

### 3.3 Categorical Features
```python
# Categorical feature distributions
categorical_cols = data.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.drop('y')  # Exclude target

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    data[col].value_counts().plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Insights:**
- Job: Most common are blue-collar, management, technician
- Marital: Mostly married
- Education: Secondary education most common
- Contact: Many "unknown" (data quality issue)

### 3.4 Feature Relationships with Target
```python
# Subscription rate by categorical features
for col in categorical_cols:
    print(f"\n{col}:")
    print(pd.crosstab(data[col], data['y'], normalize='index'))
    
# Visualize one example
pd.crosstab(data['job'], data['y']).plot(kind='bar', stacked=True)
plt.title('Subscription by Job Type')
plt.xlabel('Job')
plt.ylabel('Count')
plt.legend(title='Subscribed')
plt.xticks(rotation=45)
plt.show()
```

**Look For:**
- Which categories have higher subscription rates?
- Are there patterns (e.g., students subscribe more)?

### 3.5 Correlation Analysis
```python
# Correlation matrix for numeric features
correlation = data[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

**What to Check:**
- High correlations (>0.7) → potential multicollinearity
- Correlations with target (if encoded)

---

## 🔧 Step 4: Data Preprocessing

### 4.1 Handle Missing Values
```python
# Check for missing values
print(data.isnull().sum())
```

**Expected:** No missing values in this dataset

**If there were missing values:**
```python
# For numeric: impute with mean/median
data['age'].fillna(data['age'].median(), inplace=True)

# For categorical: impute with mode
data['job'].fillna(data['job'].mode()[0], inplace=True)

# Or drop rows
data = data.dropna()
```

### 4.2 Encode Target Variable
```python
# Encode target: 'yes' → 1, 'no' → 0
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Verify
print(data['y'].value_counts())
```

### 4.3 Encode Categorical Features
```python
# One-hot encoding for categorical features
categorical_cols = ['job', 'marital', 'education', 'default', 
                    'housing', 'loan', 'contact', 'month', 'poutcome']

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Check new shape
print(f"Original shape: {data.shape}")
print(f"After encoding: {data_encoded.shape}")
```

**Why drop_first=True:**
- Avoids multicollinearity (dummy variable trap)
- If we have n categories, n-1 dummies are sufficient
- Example: marital has 3 categories → need only 2 dummies

**Alternative: Label Encoding**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in categorical_cols:
    data[col + '_encoded'] = le.fit_transform(data[col])
```

**When to use which:**
- **One-hot:** Nominal categories (no order)
- **Label:** Ordinal categories (has order)

### 4.4 Feature-Target Separation
```python
# Separate features and target
X = data_encoded.drop('y', axis=1)
y = data_encoded['y']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### 4.5 Train-Test Split
```python
# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTraining target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"\nTest target distribution:\n{y_test.value_counts(normalize=True)}")
```

**Critical Points:**
1. **test_size=0.2:** 80% train, 20% test
2. **stratify=y:** Maintains class distribution in both sets
3. **random_state=42:** Reproducibility

### 4.6 Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame (optional, for interpretability)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```

**Why Scaling:**
- Features have different ranges (age: 18-95, balance: -8019 to 102127)
- Distance-based algorithms (Logistic Regression, SVM, KNN) sensitive to scale
- Helps gradient descent converge faster

**Important:** Fit scaler on training data only to avoid data leakage!

---

## 🔧 Step 5: Baseline Model

### 5.1 Train Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

# Train baseline model
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_baseline = baseline_model.predict(X_test_scaled)
y_pred_proba_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]
```

**Why Logistic Regression:**
- Simple, interpretable baseline
- Fast to train
- Provides probability estimates
- Good starting point

### 5.2 Evaluate Baseline
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Classification report
print("Baseline Model Performance:")
print(classification_report(y_test, y_pred_baseline))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_baseline)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Baseline Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba_baseline)
print(f"\nROC-AUC Score: {roc_auc:.3f}")
```

**Expected Results:**
- High accuracy (~88%) but misleading
- Low recall for class 1 (subscriptions)
- Model predicts mostly "no subscription"

**Why This Happens:**
- Imbalanced data (88% class 0)
- Model learns to predict majority class
- Not useful for business!

---

## 🔧 Step 6: Handle Imbalance with SMOTE

### 6.1 Apply SMOTE
```python
from imblearn.over_sampling import SMOTE

# Create SMOTE instance
smote = SMOTE(random_state=42)

# Apply to training data only
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Check new distribution
print("Original training distribution:")
print(y_train.value_counts())
print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())
```

**SMOTE Parameters:**
- `k_neighbors=5`: Number of neighbors to use (default)
- `sampling_strategy='auto'`: Balance to 50-50 (default)
- `random_state=42`: Reproducibility

**How SMOTE Works:**
1. For each minority sample
2. Find k nearest minority neighbors
3. Randomly select one neighbor
4. Create synthetic sample between them
5. Repeat until balanced

### 6.2 Train Model with SMOTE
```python
# Train on balanced data
smote_model = LogisticRegression(random_state=42, max_iter=1000)
smote_model.fit(X_train_smote, y_train_smote)

# Predictions on original test set
y_pred_smote = smote_model.predict(X_test_scaled)
y_pred_proba_smote = smote_model.predict_proba(X_test_scaled)[:, 1]
```

### 6.3 Evaluate SMOTE Model
```python
# Classification report
print("SMOTE Model Performance:")
print(classification_report(y_test, y_pred_smote))

# Confusion matrix
cm_smote = confusion_matrix(y_test, y_pred_smote)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens')
plt.title('SMOTE Model Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC-AUC
roc_auc_smote = roc_auc_score(y_test, y_pred_proba_smote)
print(f"\nROC-AUC Score: {roc_auc_smote:.3f}")
```

**Expected Improvements:**
- Higher recall for class 1 (catch more subscribers)
- Better F1-score
- More balanced predictions
- Better for business use case

---

## 🔧 Step 7: Hyperparameter Tuning

### 7.1 Grid Search
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Fit on SMOTE data
grid_search.fit(X_train_smote, y_train_smote)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best CV F1 Score:", grid_search.best_score_)
```

**Parameters Explained:**
- **C:** Inverse regularization strength (smaller = stronger regularization)
- **penalty:** L1 (Lasso) or L2 (Ridge) regularization
- **solver:** Optimization algorithm
- **cv=5:** 5-fold cross-validation
- **scoring='f1':** Optimize F1-score (good for imbalanced data)

### 7.2 Random Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define parameter distributions
param_dist = {
    'C': uniform(0.01, 100),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Random search
random_search = RandomizedSearchCV(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train_smote, y_train_smote)

print("Best Parameters:", random_search.best_params_)
print("Best CV F1 Score:", random_search.best_score_)
```

**Grid vs Random Search:**
- **Grid:** Tries all combinations (exhaustive)
- **Random:** Samples randomly (more efficient)
- **When to use:** Random for initial exploration, Grid for refinement

### 7.3 Evaluate Best Model
```python
# Get best model
best_model = grid_search.best_estimator_

# Predictions
y_pred_best = best_model.predict(X_test_scaled)
y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("Best Model Performance:")
print(classification_report(y_test, y_pred_best))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_best):.3f}")
```

---

## 🔧 Step 8: Advanced Models

### 8.1 Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.3f}")
```

**Random Forest Advantages:**
- Handles non-linear relationships
- Less prone to overfitting
- Feature importance built-in
- `class_weight='balanced'` handles imbalance

### 8.2 XGBoost
```python
from xgboost import XGBClassifier

# Calculate scale_pos_weight for imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("XGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.3f}")
```

**XGBoost Advantages:**
- Often best performance
- Handles missing values
- Built-in regularization
- `scale_pos_weight` handles imbalance

---

## 🔧 Step 9: Model Comparison

### 9.1 Compare All Models
```python
# Create comparison DataFrame
models_comparison = pd.DataFrame({
    'Model': ['Baseline', 'SMOTE', 'Best Tuned', 'Random Forest', 'XGBoost'],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_baseline),
        roc_auc_score(y_test, y_pred_proba_smote),
        roc_auc_score(y_test, y_pred_proba_best),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_xgb)
    ]
})

# Sort by ROC-AUC
models_comparison = models_comparison.sort_values('ROC-AUC', ascending=False)
print(models_comparison)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(data=models_comparison, x='Model', y='ROC-AUC')
plt.title('Model Comparison by ROC-AUC')
plt.xticks(rotation=45)
plt.ylim(0.5, 1.0)
plt.show()
```

### 9.2 ROC Curves Comparison
```python
from sklearn.metrics import roc_curve

# Calculate ROC curves
fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_proba_baseline)
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred_proba_smote)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr_base, tpr_base, label='Baseline')
plt.plot(fpr_smote, tpr_smote, label='SMOTE')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🔧 Step 10: Feature Importance

### 10.1 Random Forest Feature Importance
```python
# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 15 features
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Top 15 Important Features (Random Forest)')
plt.xlabel('Importance')
plt.show()

# Print top 10
print("Top 10 Important Features:")
print(feature_importance.head(10))
```

**Interpretation:**
- Higher importance = more useful for prediction
- Identifies key drivers of subscription
- Can guide marketing strategy

**Example Insights:**
- Duration (call length) often most important
- Previous campaign outcome matters
- Balance and age are significant

### 10.2 Logistic Regression Coefficients
```python
# Get coefficients
coefficients = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': best_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

# Plot top 15
plt.figure(figsize=(10, 8))
sns.barplot(data=coefficients.head(15), x='coefficient', y='feature')
plt.title('Top 15 Feature Coefficients (Logistic Regression)')
plt.xlabel('Coefficient')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.show()
```

**Interpretation:**
- **Positive coefficient:** Increases subscription probability
- **Negative coefficient:** Decreases subscription probability
- **Magnitude:** Strength of effect

---

## 🔧 Step 11: Cross-Validation

### 11.1 K-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define CV strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate best model with CV
cv_scores = cross_val_score(
    estimator=best_model,
    X=X_train_scaled,
    y=y_train,
    cv=cv,
    scoring='f1'
)

print(f"CV F1 Scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

### 11.2 Multiple Metrics CV
```python
from sklearn.model_selection import cross_validate

# Define metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Cross-validate
cv_results = cross_validate(
    estimator=best_model,
    X=X_train_scaled,
    y=y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True
)

# Print results
print("\nCross-Validation Results:")
for metric in scoring.keys():
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"{metric}:")
    print(f"  Train: {train_scores.mean():.3f} (+/- {train_scores.std():.3f})")
    print(f"  Test:  {test_scores.mean():.3f} (+/- {test_scores.std():.3f})")
```

**What to Look For:**
- **Large gap between train and test:** Overfitting
- **Both low:** Underfitting
- **High variance (std):** Model unstable

---

## 💡 Key Takeaways

### 1. Data Understanding
- **Imbalanced dataset** is the main challenge
- **Feature engineering** can improve performance
- **Domain knowledge** helps interpret results

### 2. Preprocessing
- **Split before preprocessing** to avoid leakage
- **Stratified split** maintains class distribution
- **Scaling** essential for distance-based algorithms

### 3. Handling Imbalance
- **SMOTE** creates synthetic minority samples
- **class_weight='balanced'** alternative approach
- **Appropriate metrics** (F1, ROC-AUC) essential

### 4. Model Selection
- **Start simple** (Logistic Regression baseline)
- **Try multiple models** (RF, XGBoost)
- **Tune hyperparameters** systematically

### 5. Evaluation
- **Multiple metrics** provide complete picture
- **Cross-validation** ensures robustness
- **Feature importance** provides insights

### 6. Business Impact
- **Recall matters** (catch potential subscribers)
- **Precision matters** (avoid wasting resources)
- **Balance based on business costs**

---

## 🚨 Common Mistakes to Avoid

1. **Using accuracy on imbalanced data** → Misleading metric
2. **Applying SMOTE before splitting** → Data leakage
3. **Fitting scaler on entire dataset** → Data leakage
4. **Not using stratified split** → Unrepresentative samples
5. **Ignoring feature importance** → Miss insights
6. **Overfitting to validation set** → Need separate test set
7. **Not checking class distribution** → Miss imbalance
8. **Using test data for any decisions** → Leakage

---

## 🎯 Extension Ideas

1. **Feature Engineering:**
   - Create interaction features
   - Polynomial features
   - Binning continuous variables

2. **Advanced Techniques:**
   - Ensemble methods (Voting, Stacking)
   - Cost-sensitive learning
   - Threshold optimization

3. **Model Interpretation:**
   - LIME for local explanations
   - SHAP for global importance
   - Partial dependence plots

4. **Business Analysis:**
   - Cost-benefit analysis
   - Customer segmentation
   - Campaign optimization

---

*This coding guide provides step-by-step explanations for the bank term deposit assignment. Practice with the actual notebook and experiment with different approaches!*
