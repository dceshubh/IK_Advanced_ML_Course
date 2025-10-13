# Week 15: Model Evaluation Live Class - Coding Guide

## 📓 Notebook: Model Evaluation Live Class Notebook.ipynb

## 🎯 Objective
Build a customer churn prediction model for a telecom company, focusing on proper model evaluation techniques, handling imbalanced data with SMOTE, and interpreting predictions using LIME and SHAP.

---

## 📊 Dataset: Telco Customer Churn

**Business Context:**
Telecom companies lose customers (churn) regularly. Predicting which customers are likely to churn allows proactive retention efforts.

**Dataset Features:**
- Customer demographics (age, gender, etc.)
- Service information (contract type, payment method)
- Usage patterns (call duration, data usage)
- Target: Churn (Yes/No)

**Challenge:** Imbalanced dataset - most customers don't churn

---

## 🔧 Section 1: Data Loading and Exploration

### Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
```

**Key Libraries:**
- `pandas`: Data manipulation
- `sklearn`: Machine learning algorithms and tools
- `matplotlib/seaborn`: Visualization
- `imblearn`: Handling imbalanced data (installed separately)

### Load Data
```python
# Load dataset
df = pd.read_excel('Telco_customer_churn_dataset.xlsx')

# Initial exploration
print(df.shape)
print(df.head())
print(df.info())
```

**What to Check:**
- Dataset size (rows, columns)
- Data types
- Missing values
- Target distribution

### Exploratory Data Analysis
```python
# Check target distribution
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))

# Visualize imbalance
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.show()
```

**Expected Finding:** Imbalanced dataset (e.g., 80% No Churn, 20% Churn)

**Why This Matters:**
- Standard accuracy will be misleading
- Need special techniques (SMOTE, appropriate metrics)
- Model may ignore minority class

---

## 🔧 Section 2: Data Preprocessing

### Handle Missing Values
```python
# Check for missing values
print(df.isnull().sum())

# Handle missing values
# Option 1: Drop rows with missing values
df = df.dropna()

# Option 2: Impute with mean/median/mode
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

**Strategies:**
- **Drop:** If few missing values (<5%)
- **Impute:** If many missing values
  - Mean/Median: Numeric features
  - Mode: Categorical features
  - Advanced: KNN imputation, model-based

### Encode Categorical Variables
```python
# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Categorical columns: {categorical_cols}")

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Alternative: Label encoding for ordinal features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Churn_encoded'] = le.fit_transform(df['Churn'])
```

**Encoding Methods:**
- **One-Hot:** For nominal categories (no order)
- **Label:** For ordinal categories (has order)
- **Target:** For high-cardinality features (use with caution)

**Important:** `drop_first=True` avoids multicollinearity

### Feature Scaling
```python
# Separate features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split data FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
```

**Critical Points:**
1. **Split before scaling** - prevents data leakage
2. **Fit on training only** - test set is "unseen"
3. **Use stratify** - maintains class distribution
4. **StandardScaler** - for features with different scales

---

## 🔧 Section 3: Baseline Model (Without SMOTE)

### Train Baseline Model
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
- Good for binary classification
- Provides probability estimates

### Evaluate Baseline Model
```python
# Classification report
print("Baseline Model Performance:")
print(classification_report(y_test, y_pred_baseline))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Baseline Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba_baseline)
print(f"ROC-AUC Score: {roc_auc:.3f}")
```

**Interpreting Results:**

**Confusion Matrix:**
```
                Predicted
                No    Yes
Actual  No      TN    FP
        Yes     FN    TP
```

**Classification Report:**
- **Precision (Churn):** Of predicted churners, how many actually churned?
- **Recall (Churn):** Of actual churners, how many did we catch?
- **F1-Score:** Harmonic mean of precision and recall

**Expected Baseline Issue:**
- High accuracy (e.g., 80%) but poor recall for churn class
- Model predicts "No Churn" for most samples
- Not useful for business (we need to catch churners!)

### ROC Curve
```python
# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_baseline)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Baseline (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**ROC Curve Interpretation:**
- **Closer to top-left:** Better model
- **Diagonal line:** Random classifier
- **AUC = 0.5:** Random
- **AUC = 1.0:** Perfect

---

## 🔧 Section 4: Handling Imbalance with SMOTE

### Apply SMOTE
```python
from imblearn.over_sampling import SMOTE

# Create SMOTE instance
smote = SMOTE(random_state=42)

# Apply SMOTE to training data only
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Check new distribution
print("Original distribution:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_smote).value_counts())
```

**SMOTE Process:**
1. For each minority sample, find k nearest neighbors (default k=5)
2. Randomly select one neighbor
3. Create synthetic sample along line between them
4. Repeat until classes balanced

**Important:**
- **Apply only to training data** - never to test data!
- **After splitting** - to avoid data leakage
- **Before scaling** or after (both work, but be consistent)

### Train Model with SMOTE
```python
# Train model on balanced data
smote_model = LogisticRegression(random_state=42, max_iter=1000)
smote_model.fit(X_train_smote, y_train_smote)

# Predictions (on original test set)
y_pred_smote = smote_model.predict(X_test_scaled)
y_pred_proba_smote = smote_model.predict_proba(X_test_scaled)[:, 1]
```

### Evaluate SMOTE Model
```python
# Classification report
print("SMOTE Model Performance:")
print(classification_report(y_test, y_pred_smote))

# Confusion matrix
cm_smote = confusion_matrix(y_test, y_pred_smote)
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens')
plt.title('SMOTE Model Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC-AUC
roc_auc_smote = roc_auc_score(y_test, y_pred_proba_smote)
print(f"ROC-AUC Score: {roc_auc_smote:.3f}")
```

**Expected Improvements:**
- **Higher recall** for churn class (catch more churners)
- **Better F1-score** (balanced precision-recall)
- **Slightly lower precision** (more false positives)
- **Overall better for business** (catching churners is priority)

### Compare Models
```python
# Plot ROC curves together
fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_proba_baseline)
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred_proba_smote)

plt.figure(figsize=(10, 6))
plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {roc_auc:.3f})')
plt.plot(fpr_smote, tpr_smote, label=f'SMOTE (AUC = {roc_auc_smote:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()
```

---

## 🔧 Section 5: Advanced Models

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Alternative to SMOTE
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
- Feature importance built-in
- Less prone to overfitting
- `class_weight='balanced'` handles imbalance

### Feature Importance
```python
# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Important Features')
plt.show()
```

**Interpretation:**
- Higher importance = more useful for prediction
- Helps identify key churn drivers
- Can guide feature engineering

---

## 🔧 Section 6: Model Interpretation with LIME

### Setup LIME
```python
import lime
import lime.lime_tabular

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['No Churn', 'Churn'],
    mode='classification'
)
```

### Explain Single Prediction
```python
# Select a sample to explain
sample_idx = 0
sample = X_test_scaled[sample_idx]

# Generate explanation
exp = explainer.explain_instance(
    data_row=sample,
    predict_fn=smote_model.predict_proba,
    num_features=10
)

# Visualize
exp.show_in_notebook(show_table=True)

# Alternative: as list
print(exp.as_list())
```

**LIME Output Interpretation:**
- **Green bars:** Features pushing toward "Churn"
- **Red bars:** Features pushing toward "No Churn"
- **Bar length:** Strength of contribution
- **Values:** Feature values for this sample

**Example:**
```
Contract_Month-to-Month: 0.35  (pushes toward Churn)
Tenure: -0.20  (pushes toward No Churn)
MonthlyCharges: 0.15  (pushes toward Churn)
```

### Explain Multiple Predictions
```python
# Explain several predictions
for i in range(5):
    sample = X_test_scaled[i]
    exp = explainer.explain_instance(sample, smote_model.predict_proba)
    
    print(f"\nSample {i}:")
    print(f"Actual: {y_test.iloc[i]}")
    print(f"Predicted: {smote_model.predict([sample])[0]}")
    print(f"Top features: {exp.as_list()[:3]}")
```

**Use Cases:**
- Explain individual predictions to customers
- Identify why model made specific decision
- Debug model behavior
- Build trust with stakeholders

---

## 🔧 Section 7: Model Interpretation with SHAP

### Setup SHAP
```python
import shap

# Create SHAP explainer
# For tree-based models
explainer_shap = shap.TreeExplainer(rf_model)

# For other models
# explainer_shap = shap.KernelExplainer(model.predict_proba, X_train_scaled)

# Calculate SHAP values
shap_values = explainer_shap.shap_values(X_test_scaled)
```

**SHAP Explainer Types:**
- **TreeExplainer:** Fast for tree-based models (RF, XGBoost)
- **LinearExplainer:** For linear models
- **KernelExplainer:** Model-agnostic (slower)
- **DeepExplainer:** For neural networks

### Global Feature Importance
```python
# Summary plot (global importance)
shap.summary_plot(shap_values[1], X_test_scaled, feature_names=X.columns)
```

**Summary Plot Interpretation:**
- **Y-axis:** Features ranked by importance
- **X-axis:** SHAP value (impact on prediction)
- **Color:** Feature value (red=high, blue=low)
- **Dots:** Individual samples

**Example Insights:**
- "High MonthlyCharges (red dots) push toward Churn (positive SHAP)"
- "Long Tenure (red dots) push toward No Churn (negative SHAP)"

### Force Plot (Single Prediction)
```python
# Explain single prediction
shap.force_plot(
    base_value=explainer_shap.expected_value[1],
    shap_values=shap_values[1][0],
    features=X_test_scaled[0],
    feature_names=X.columns
)
```

**Force Plot Interpretation:**
- **Base value:** Average model output
- **Red arrows:** Push prediction higher (toward Churn)
- **Blue arrows:** Push prediction lower (toward No Churn)
- **Final value:** Actual prediction

### Dependence Plot
```python
# Show relationship between feature and SHAP value
shap.dependence_plot(
    ind='MonthlyCharges',
    shap_values=shap_values[1],
    features=X_test_scaled,
    feature_names=X.columns
)
```

**Dependence Plot Interpretation:**
- **X-axis:** Feature value
- **Y-axis:** SHAP value (impact on prediction)
- **Color:** Interaction feature
- Shows how feature affects prediction across its range

### Waterfall Plot
```python
# Waterfall plot for single prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[1][0],
        base_values=explainer_shap.expected_value[1],
        data=X_test_scaled[0],
        feature_names=X.columns
    )
)
```

**Waterfall Plot Interpretation:**
- Shows cumulative effect of features
- Starts from base value
- Each bar adds/subtracts to reach final prediction

---

## 🔧 Section 8: Cross-Validation

### K-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model with CV
cv_scores = cross_val_score(
    estimator=smote_model,
    X=X_train_scaled,
    y=y_train,
    cv=cv,
    scoring='f1'
)

print(f"CV F1 Scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

**Why Cross-Validation:**
- More robust than single train-test split
- Reduces variance in performance estimate
- Uses all data for both training and validation
- Essential for hyperparameter tuning

**Stratified K-Fold:**
- Maintains class distribution in each fold
- Critical for imbalanced datasets

### Multiple Metrics
```python
from sklearn.model_selection import cross_validate

# Evaluate multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = cross_validate(
    estimator=smote_model,
    X=X_train_scaled,
    y=y_train,
    cv=cv,
    scoring=scoring
)

# Print results
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 💡 Key Takeaways

### 1. Data Preprocessing
- **Split before preprocessing** to avoid data leakage
- **Use stratified split** for imbalanced data
- **Scale features** for distance-based algorithms
- **Encode categoricals** appropriately

### 2. Handling Imbalance
- **SMOTE** creates synthetic minority samples
- **Apply only to training data**
- **Alternative:** class_weight='balanced'
- **Use appropriate metrics** (F1, ROC-AUC, not accuracy)

### 3. Model Evaluation
- **Multiple metrics:** Precision, Recall, F1, ROC-AUC
- **Confusion matrix:** Understand error types
- **ROC curve:** Performance across thresholds
- **Cross-validation:** Robust performance estimate

### 4. Model Interpretation
- **LIME:** Local explanations, intuitive
- **SHAP:** Global + local, theoretically sound
- **Feature importance:** Identify key drivers
- **Use both:** Complementary insights

### 5. Business Impact
- **Recall matters** for churn (catch churners)
- **Interpretability** builds trust
- **Actionable insights** from feature importance
- **Cost-benefit analysis** guides threshold selection

---

## 🚨 Common Mistakes to Avoid

1. **Applying SMOTE before splitting** → Data leakage
2. **Using accuracy on imbalanced data** → Misleading
3. **Fitting scaler on entire dataset** → Data leakage
4. **Not using stratified split** → Unrepresentative folds
5. **Ignoring business context** → Wrong metric optimization
6. **Overfitting to validation set** → Need separate test set
7. **Not checking class distribution** → Miss imbalance issue
8. **Using test data for any training decisions** → Leakage

---

## 🎯 Practice Exercises

1. **Try different SMOTE parameters** (k_neighbors, sampling_strategy)
2. **Compare SMOTE with class_weight='balanced'**
3. **Experiment with different classification thresholds**
4. **Try other models** (XGBoost, SVM, Neural Network)
5. **Feature engineering** (create interaction features)
6. **Hyperparameter tuning** (Grid/Random Search)
7. **Cost-sensitive learning** (custom loss function)
8. **Ensemble methods** (Voting, Stacking)

---

*This coding guide provides detailed explanations for the live class notebook. Practice with the actual notebook and experiment with different approaches!*
