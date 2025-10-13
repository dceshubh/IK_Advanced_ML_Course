# Coding Guide: Classification Algorithms Assignment - Customer Churn Prediction

## Overview
This assignment applies classification algorithms (Logistic Regression and KNN) to predict customer churn. The goal is to identify which customers are likely to leave (churn) so the company can take preventive action.

**Business Context**: Customer churn is when customers stop doing business with a company. Predicting churn helps companies:
- Identify at-risk customers
- Take proactive retention measures
- Reduce customer acquisition costs
- Improve customer satisfaction

---

## Section 1: Library Imports

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
```

### Why These Libraries?

**pandas (pd)**: Data manipulation
- Reading CSV files
- Data cleaning and transformation
- Handling missing values

**seaborn (sns) & matplotlib.pyplot (plt)**: Visualization
- Creating count plots for categorical data
- Visualizing churn distribution
- Analyzing feature relationships

**sklearn.model_selection**:
- `train_test_split`: Splits data into training and testing sets

**sklearn.linear_model**:
- `LogisticRegression`: Binary classification algorithm

**sklearn.neighbors**:
- `KNeighborsClassifier`: K-Nearest Neighbors algorithm

**sklearn.metrics**: Model evaluation
- `accuracy_score`: Overall correctness
- `confusion_matrix`: Detailed prediction breakdown
- `classification_report`: Precision, recall, F1-score
- `precision_score`, `recall_score`, `f1_score`: Individual metrics

**sklearn.preprocessing**:
- `StandardScaler`: Feature scaling (mean=0, std=1)

**numpy (np)**: Numerical operations
- Array manipulations
- Mathematical calculations

---

## Section 2: Data Loading

```python
data = pd.read_csv("/content/drive/MyDrive/datasets/Customer_Churn.csv")
data.head()
```

### Key Functions:

**pd.read_csv()**:
- Reads CSV file into DataFrame
- `filepath`: Path to the CSV file
- Returns pandas DataFrame

**data.head()**:
- Shows first 5 rows
- Quick data inspection

### Dataset Features:
- **customerID**: Unique identifier for each customer
- **gender**: Customer gender (Male/Female)
- **SeniorCitizen**: Whether customer is senior (0/1)
- **Partner**: Whether customer has partner (Yes/No)
- **Dependents**: Whether customer has dependents (Yes/No)
- **tenure**: Number of months customer has stayed
- **PhoneService**: Whether customer has phone service
- **MultipleLines**: Whether customer has multiple lines
- **InternetService**: Type of internet service (DSL/Fiber optic/No)
- **OnlineSecurity**: Whether customer has online security
- **OnlineBackup**: Whether customer has online backup
- **DeviceProtection**: Whether customer has device protection
- **TechSupport**: Whether customer has tech support
- **StreamingTV**: Whether customer has streaming TV
- **StreamingMovies**: Whether customer has streaming movies
- **Contract**: Contract type (Month-to-month/One year/Two year)
- **PaperlessBilling**: Whether customer uses paperless billing
- **PaymentMethod**: Payment method
- **MonthlyCharges**: Monthly charge amount
- **TotalCharges**: Total charges
- **Churn**: Target variable (Yes/No) - Did customer leave?

---

## Section 3: Data Preprocessing

### 3.1 Removing Unnecessary Columns

```python
data.drop(columns=['customerID'], inplace=True)
```

### Key Functions:

**data.drop()**:
- `columns=['customerID']`: Column to remove
- `inplace=True`: Modifies original DataFrame (doesn't create copy)

### Why Remove customerID?
- It's just an identifier
- No predictive value
- Including it would confuse the model
- Each ID is unique, so it can't generalize

---

### 3.2 Data Exploration

```python
data.info()
```

### What data.info() Shows:
- **Column names**: All feature names
- **Non-Null Count**: Number of non-missing values
- **Dtype**: Data type of each column
  - `int64`: Integer numbers
  - `float64`: Decimal numbers
  - `object`: Text/strings (usually categorical)
- **Memory usage**: RAM consumption

### Key Observations:
- Check for missing values (Non-Null Count < total rows)
- Identify categorical vs numerical features
- Spot data type issues (e.g., numbers stored as strings)

---

### 3.3 Target Variable Distribution

```python
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.show()
```

### Key Functions:

**plt.figure()**:
- `figsize=(6, 4)`: Sets figure size (width, height) in inches

**sns.countplot()**:
- `x='Churn'`: Variable to plot
- `data=data`: Source DataFrame
- Creates bar chart showing count of each category

**plt.title()**: Adds title to plot

### Why This Matters:
- Reveals class imbalance
- If 80% No Churn, 20% Churn → Imbalanced
- Affects model evaluation strategy
- May need sampling techniques

---

### 3.4 Categorical Feature Analysis

```python
categorical_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
```

### Why List Categorical Columns?
- Need to encode them for machine learning
- Want to analyze their relationship with churn
- Different preprocessing than numerical features

---

### 3.5 Visualizing Categorical Features

```python
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 16))
axes = axes.flatten()

for i, column in enumerate(categorical_columns):
    normalized_counts = (
        data.groupby(column)['Churn']
        .value_counts(normalize=True)
        .unstack()
    )
    
    normalized_counts.plot(kind='bar', ax=axes[i], stacked=False)
    axes[i].set_title(f'Churn by {column}')
    axes[i].set_ylabel('Proportion')
```

### Key Functions:

**plt.subplots()**:
- `nrows=4, ncols=4`: Creates 4x4 grid (16 subplots)
- `figsize=(18, 16)`: Large figure for readability
- Returns `fig` (figure) and `axes` (array of subplots)

**axes.flatten()**:
- Converts 2D array of axes to 1D
- Makes it easier to iterate through subplots

**data.groupby()**:
- Groups data by specified column
- Allows aggregation within groups

**value_counts()**:
- `normalize=True`: Returns proportions instead of counts
- Shows percentage of each category

**unstack()**:
- Converts multi-index to columns
- Makes data suitable for plotting

**enumerate()**:
- Returns index and value while iterating
- `i`: Index (0, 1, 2, ...)
- `column`: Column name

### Why Normalize?
- Class imbalance makes raw counts misleading
- Proportions show true relationship
- Example: If 80% don't churn, raw counts would show high bars for "No Churn" everywhere
- Normalized shows: "Of customers with X feature, what % churn?"

---

### 3.6 Handling Data Type Issues

```python
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
```

### Key Functions:

**pd.to_numeric()**:
- Converts column to numeric type
- `errors='coerce'`: Converts invalid values to NaN (Not a Number)
  - Alternative: `errors='raise'` (throws error)
  - Alternative: `errors='ignore'` (leaves as is)

### Why This is Needed:
- TotalCharges might be stored as string (object type)
- Machine learning algorithms need numbers
- Some values might be empty strings or invalid
- `coerce` handles these gracefully by making them NaN

### After Conversion:
- Check for NaN values: `data['TotalCharges'].isna().sum()`
- Decide how to handle them:
  - Drop rows: `data.dropna()`
  - Fill with mean: `data['TotalCharges'].fillna(data['TotalCharges'].mean())`
  - Fill with median: `data['TotalCharges'].fillna(data['TotalCharges'].median())`

---

## Section 4: Feature Engineering

### 4.1 One-Hot Encoding

```python
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
```

### Key Functions:

**pd.get_dummies()**:
- Converts categorical variables to binary columns
- `data`: DataFrame to encode
- `columns=categorical_columns`: Which columns to encode
- `drop_first=True`: Drops first category to avoid multicollinearity

### How It Works:

**Without drop_first**:
```
gender: Male, Female
→ gender_Male (0/1), gender_Female (0/1)
```

**With drop_first=True**:
```
gender: Male, Female
→ gender_Male (0/1)
```
- If gender_Male = 0, we know it's Female
- Reduces redundancy
- Prevents multicollinearity (perfect correlation between features)

### Example:
```
Original:
Contract: Month-to-month, One year, Two year

After encoding (drop_first=True):
Contract_One year: 0 or 1
Contract_Two year: 0 or 1

If both are 0 → Month-to-month
If Contract_One year = 1 → One year
If Contract_Two year = 1 → Two year
```

### Why One-Hot Encoding?
- Machine learning algorithms need numerical input
- Can't use label encoding (1, 2, 3) for nominal categories
- Label encoding implies order (3 > 2 > 1), which is wrong for categories
- One-hot encoding treats each category independently

---

## Section 5: Train-Test Split

```python
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Key Concepts:

**Separating Features and Target**:
- `X`: Features (all columns except Churn)
- `y`: Target (Churn column)
- `axis=1`: Drop column (axis=0 would drop row)

**train_test_split()**:
- `X, y`: Data to split
- `test_size=0.2`: 20% for testing, 80% for training
- `random_state=42`: Ensures reproducible split
  - Same split every time you run the code
  - Important for comparing results

### Why 80-20 Split?
- 80% training: Enough data to learn patterns
- 20% testing: Enough data to evaluate reliably
- Common alternatives: 70-30, 75-25
- Larger datasets can use smaller test percentage

---

## Section 6: Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Key Functions:

**StandardScaler()**:
- Standardizes features: (x - mean) / std
- Result: mean = 0, std = 1

**fit_transform()**:
- `fit`: Calculates mean and std from X_train
- `transform`: Applies transformation
- Used ONLY on training data

**transform()**:
- Applies transformation using previously calculated mean/std
- Used on test data
- Uses training statistics, not test statistics

### Why This Order?

```python
# ✓ CORRECT
scaler.fit(X_train)  # Learn from training
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply same transformation

# ✗ WRONG
scaler.fit(X_test)  # Never fit on test data!
```

### Why Scaling Matters:
- Features have different ranges:
  - tenure: 0-72 months
  - MonthlyCharges: $18-$118
  - TotalCharges: $18-$8,684
- Without scaling:
  - TotalCharges dominates (largest values)
  - tenure barely affects predictions
- After scaling:
  - All features have equal influence
  - Model converges faster
  - Better performance

---

## Section 7: Logistic Regression Model

### 7.1 Training the Model

```python
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
```

### Key Functions:

**LogisticRegression()**:
- `random_state=42`: Reproducible results
- `max_iter=1000`: Maximum iterations for convergence
  - Default is 100
  - Increase if you see convergence warnings
  - Higher = more time but better convergence

**fit()**:
- `X_train_scaled, y_train`: Training data
- Learns optimal weights
- Minimizes cross-entropy loss

**predict()**:
- `X_test_scaled`: Test features
- Returns predicted class labels (0 or 1, Yes or No)
- Uses learned weights to make predictions

### How Logistic Regression Works Here:
1. Calculates: z = w₀ + w₁×tenure + w₂×MonthlyCharges + ...
2. Applies sigmoid: p = 1 / (1 + e^(-z))
3. If p > 0.5: Predicts "Yes" (Churn)
4. If p ≤ 0.5: Predicts "No" (No Churn)

---

### 7.2 Model Evaluation

```python
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))
```

### Key Metrics:

**Accuracy**:
```
Accuracy = (Correct Predictions) / (Total Predictions)
```
- Simple overall measure
- Can be misleading with imbalanced data

**Confusion Matrix**:
```
                Predicted
                No    Yes
Actual  No    [TN]  [FP]
        Yes   [FN]  [TP]
```
- **TN**: Correctly predicted No Churn
- **FP**: Predicted Churn, actually No Churn (False Alarm)
- **FN**: Predicted No Churn, actually Churn (Missed Churn)
- **TP**: Correctly predicted Churn

**Classification Report**:
Shows for each class:
- **Precision**: TP / (TP + FP)
  - Of predicted churners, how many actually churned?
  - High precision = Few false alarms
  
- **Recall**: TP / (TP + FN)
  - Of actual churners, how many did we catch?
  - High recall = Few missed churners
  
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - Harmonic mean of precision and recall
  - Balances both metrics
  
- **Support**: Number of actual occurrences of each class

### Business Interpretation:
- **High Recall Important**: Don't want to miss churners (FN costly)
- **High Precision Important**: Don't want to waste resources on false alarms (FP costly)
- **Balance**: Usually want high recall (catch churners) even if some false alarms

---

## Section 8: K-Nearest Neighbors (KNN) Model

### 8.1 Basic KNN

```python
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
```

### Key Functions:

**KNeighborsClassifier()**:
- `n_neighbors=5`: Number of neighbors to consider
- Default value, often works well

**How KNN Works**:
1. For each test point:
   - Calculate distance to all training points
   - Find 5 nearest neighbors
   - Take majority vote of their classes
   - Assign most common class

### Why KNN Needs Scaling:
- Uses Euclidean distance: √[(x₁-x₂)² + (y₁-y₂)² + ...]
- Without scaling:
  - TotalCharges (0-8684) dominates
  - tenure (0-72) barely matters
- With scaling:
  - All features contribute equally
  - Distance calculation is fair

---

### 8.2 Hyperparameter Tuning

```python
train_scores = []
test_scores = []
k_values = range(1, 31)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))
```

### Key Concepts:

**range(1, 31)**:
- Creates sequence: [1, 2, 3, ..., 30]
- Tests different K values

**knn.score()**:
- Returns accuracy on given data
- Shortcut for `accuracy_score(y_true, knn.predict(X))`

### Why Test Multiple K Values?
- **K=1**: Very sensitive to noise, overfits
- **K=30**: Too smooth, underfits
- **Optimal K**: Balances bias and variance

### What to Look For:
- **Overfitting**: High train score, low test score
- **Underfitting**: Both scores low
- **Good fit**: High test score, small train-test gap

---

### 8.3 Visualizing K Selection

```python
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Train Accuracy', marker='o')
plt.plot(k_values, test_scores, label='Test Accuracy', marker='o')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN: K vs Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

### Key Functions:

**plt.plot()**:
- `k_values`: x-axis (K values)
- `train_scores`: y-axis (accuracy)
- `label`: Legend label
- `marker='o'`: Adds circle markers at data points

**plt.xlabel() / plt.ylabel()**: Axis labels

**plt.legend()**: Shows legend

**plt.grid(True)**: Adds grid lines

### Interpreting the Plot:
- **Training curve**: Usually decreases as K increases
- **Test curve**: Usually increases then decreases
- **Optimal K**: Where test curve peaks
- **Gap**: Distance between curves indicates overfitting

---

### 8.4 Final KNN Model

```python
optimal_k = 15  # Based on plot
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train_scaled, y_train)
y_pred_knn_final = knn_final.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_knn_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn_final))
```

### Choosing Optimal K:
- Look at the plot
- Find K where test accuracy is highest
- Consider train-test gap
- Prefer odd K for binary classification (avoids ties)

---

## Section 9: Model Comparison

### Comparing Logistic Regression vs KNN

```python
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn_final))
```

### Factors to Consider:

**Accuracy**:
- Which model has higher accuracy?
- Is the difference significant?

**Precision vs Recall**:
- Which model catches more churners (recall)?
- Which model has fewer false alarms (precision)?

**Business Context**:
- Cost of missing a churner (FN)
- Cost of false alarm (FP)
- Choose model that minimizes business cost

**Interpretability**:
- Logistic Regression: Can see feature importance (weights)
- KNN: Black box, hard to explain

**Speed**:
- Logistic Regression: Fast predictions
- KNN: Slower (calculates distances)

**Scalability**:
- Logistic Regression: Handles large datasets well
- KNN: Struggles with large datasets (memory + speed)

---

## Section 10: Key Takeaways

### Data Preprocessing
1. **Remove irrelevant columns** (customerID)
2. **Check data types** (convert TotalCharges to numeric)
3. **Handle missing values** (drop or impute)
4. **Encode categorical variables** (one-hot encoding with drop_first)
5. **Scale features** (StandardScaler for distance-based algorithms)

### Model Training
1. **Split data** (80-20 train-test)
2. **Fit scaler on training data only**
3. **Train multiple models** (Logistic Regression, KNN)
4. **Tune hyperparameters** (K for KNN)
5. **Evaluate on test set**

### Evaluation
1. **Don't rely on accuracy alone** (especially with imbalance)
2. **Check confusion matrix** (understand error types)
3. **Consider business context** (cost of FP vs FN)
4. **Compare multiple models**
5. **Choose based on business needs**

### Best Practices
1. **Always scale for KNN** (distance-based)
2. **Use drop_first in one-hot encoding** (avoid multicollinearity)
3. **Set random_state** (reproducibility)
4. **Visualize hyperparameter tuning** (understand trade-offs)
5. **Consider class imbalance** (use appropriate metrics)

---

## Conclusion

This assignment demonstrates a complete classification pipeline for customer churn prediction:
1. Load and explore data
2. Preprocess and engineer features
3. Train multiple models
4. Tune hyperparameters
5. Evaluate and compare

The key skills practiced:
- Data cleaning and type conversion
- One-hot encoding for categorical variables
- Feature scaling for distance-based algorithms
- Training Logistic Regression and KNN
- Hyperparameter tuning with visualization
- Comprehensive model evaluation

Understanding these concepts prepares you for real-world classification problems where choosing the right model and preprocessing steps is crucial for success.
