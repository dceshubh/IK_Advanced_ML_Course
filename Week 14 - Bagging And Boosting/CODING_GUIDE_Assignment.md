# Week 14: Bagging & Boosting - Assignment Coding Guide

## 📓 Notebook: Bagging_&_Boosting_Assignment_Solution.ipynb

## 🎯 Objective
Apply ensemble methods to predict Pokémon rarity using bagging (Random Forest) and boosting (AdaBoost, Gradient Boosting, XGBoost) techniques.

---

## 📊 Dataset: Pokémon Rarity Classification

**Business Context:**
- Predict Pokémon rarity: Standard, Legendary, Mythic, Ultra Beast
- Multi-class classification problem
- Features: stats, abilities, types, etc.

**Rarity Classes:**
- **Standard:** Common Pokémon, easy to obtain
- **Legendary:** Rare, powerful, special events
- **Mythic:** Extremely rare, special quests
- **Ultra Beast:** Special category, unique designs

---

## 🔧 Section 1: Data Loading and Exploration

### Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

pd.set_option('display.max_columns', 100)
```

### Load Data
```python
# Load Pokémon dataset
df = pd.read_csv('pokemon_data.csv')

# Explore
print(df.shape)
print(df.head())
print(df.info())
print(df['rarity'].value_counts())
```

**Key Checks:**
- Features: HP, Attack, Defense, Speed, etc.
- Target: rarity (4 classes)
- Check for missing values
- Class distribution

### Preprocessing
```python
# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if col != 'rarity':
        df[col] = le.fit_transform(df[col])

# Encode target
df['rarity'] = le.fit_transform(df['rarity'])

# Separate features and target
X = df.drop('rarity', axis=1)
y = df['rarity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 🔧 Section 2: Baseline Model

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

# Train
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict
y_pred_dt = dt.predict(X_test)

# Evaluate
print("Decision Tree:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))
```

---

## 🔧 Section 3: Random Forest (Bagging)

### Train Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Train
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# Evaluate
print("Random Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))
```

### Feature Importance
```python
# Get importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance.head(10), x='importance', y='feature')
plt.title('Top 10 Features')
plt.show()
```

---

## 🔧 Section 4: AdaBoost

### Train AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier

# Train
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)

# Predict
y_pred_ada = ada.predict(X_test)

# Evaluate
print("AdaBoost:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada):.4f}")
print(classification_report(y_test, y_pred_ada))
```

---

## 🔧 Section 5: Gradient Boosting

### Train Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

# Train
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

# Predict
y_pred_gb = gb.predict(X_test)

# Evaluate
print("Gradient Boosting:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(classification_report(y_test, y_pred_gb))
```

---

## 🔧 Section 6: XGBoost

### Train XGBoost
```python
from xgboost import XGBClassifier

# Train
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb.predict(X_test)

# Evaluate
print("XGBoost:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))
```

---

## 🔧 Section 7: Model Comparison

### Compare All Models
```python
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
        'Accuracy': accuracy_score(y_test, preds)
    })

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print(results_df)
```

### Visualize
```python
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Model', y='Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.ylim(0.8, 1.0)
plt.show()
```

---

## 💡 Key Takeaways

### Ensemble Methods Summary
- **Bagging (Random Forest):** Reduces variance, parallel training
- **Boosting (AdaBoost, GB, XGBoost):** Reduces bias, sequential training
- **XGBoost:** Often best performance, optimized implementation

### When to Use
- **Random Forest:** Good default, handles high-dimensional data
- **Gradient Boosting:** Maximum performance, requires tuning
- **XGBoost:** Kaggle competitions, production systems

---

*Assignment complete! Practice with different datasets and hyperparameters.*
