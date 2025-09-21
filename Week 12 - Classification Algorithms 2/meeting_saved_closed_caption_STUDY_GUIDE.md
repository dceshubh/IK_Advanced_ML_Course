# Classification Algorithms 2 Study Guide 📚

## 🎯 What This Guide Covers
Advanced classification algorithms including Naive Bayes, Decision Trees, and model evaluation techniques.

---

## 🌟 Part 1: Simple Explanations

### 1. What is Naive Bayes?
**Simple Explanation:**
Naive Bayes is like a detective who assumes all clues are independent but still solves cases really well!

```
🕵️ Email Spam Detection:
Clues: "FREE", "URGENT", "CLICK NOW"
Naive assumption: Each word is independent
Reality: Words might be related, but it still works!

🏥 Medical Diagnosis:
Symptoms: Fever, Cough, Fatigue
Naive assumption: Symptoms are independent
Still predicts disease accurately!
```

### 2. What are Decision Trees?
**Simple Explanation:**
Decision trees are like a flowchart of yes/no questions that lead to a decision!

```
🌳 Decision Tree Example:
"Should I go outside?"
├─ Is it raining? 
│  ├─ Yes → Stay inside
│  └─ No → Is it sunny?
│     ├─ Yes → Go outside!
│     └─ No → Maybe go outside

🏠 House Price Prediction:
├─ Size > 2000 sqft?
│  ├─ Yes → Location = Downtown?
│  │  ├─ Yes → Expensive ($500k+)
│  │  └─ No → Moderate ($300k)
│  └─ No → Cheap ($200k)
```

### 3. Why "Naive" in Naive Bayes?
```
🤔 The "Naive" Assumption:
All features are independent of each other

📧 Email Example:
Naive assumption: Seeing "FREE" doesn't affect probability of seeing "URGENT"
Reality: Spam emails often have both words together
But it still works surprisingly well!
```

---

## 🔬 Part 2: Technical Concepts

### Naive Bayes Implementation
```python
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayesClassifier:
    def __init__(self):
        self.model = None
    
    def bayes_theorem_example(self):
        """Demonstrate Bayes' theorem"""
        print("🧮 BAYES' THEOREM EXAMPLE")
        print("=" * 30)
        
        # P(Spam|contains "FREE") = P("FREE"|Spam) * P(Spam) / P("FREE")
        
        # Prior probabilities
        p_spam = 0.3  # 30% of emails are spam
        p_ham = 0.7   # 70% are legitimate
        
        # Likelihoods
        p_free_given_spam = 0.8  # 80% of spam contains "FREE"
        p_free_given_ham = 0.1   # 10% of ham contains "FREE"
        
        # Total probability of "FREE"
        p_free = p_free_given_spam * p_spam + p_free_given_ham * p_ham
        
        # Posterior probability
        p_spam_given_free = (p_free_given_spam * p_spam) / p_free
        
        print(f"P(Spam) = {p_spam}")
        print(f"P(FREE|Spam) = {p_free_given_spam}")
        print(f"P(FREE|Ham) = {p_free_given_ham}")
        print(f"P(FREE) = {p_free:.3f}")
        print(f"P(Spam|FREE) = {p_spam_given_free:.3f}")
        
        return p_spam_given_free
    
    def train_naive_bayes(self, X, y, nb_type='gaussian'):
        """Train Naive Bayes classifier"""
        
        if nb_type == 'gaussian':
            self.model = GaussianNB()
        elif nb_type == 'multinomial':
            self.model = MultinomialNB()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Naive Bayes ({nb_type}) Accuracy: {accuracy:.4f}")
        
        return self.model, accuracy

# Decision Tree Implementation
class DecisionTreeClassifier:
    def __init__(self):
        self.model = None
    
    def calculate_entropy(self, y):
        """Calculate entropy for information gain"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def calculate_information_gain(self, X_column, y, threshold):
        """Calculate information gain for a split"""
        
        # Split data
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Calculate weighted entropy after split
        total_entropy = self.calculate_entropy(y)
        
        left_entropy = self.calculate_entropy(y[left_mask])
        right_entropy = self.calculate_entropy(y[right_mask])
        
        weighted_entropy = (np.sum(left_mask) / len(y)) * left_entropy + \
                          (np.sum(right_mask) / len(y)) * right_entropy
        
        information_gain = total_entropy - weighted_entropy
        return information_gain
    
    def find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        
        best_gain = 0
        best_feature = 0
        best_threshold = 0
        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                gain = self.calculate_information_gain(feature_values, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

# Example usage
def demonstrate_classifiers():
    """Demonstrate Naive Bayes and Decision Tree"""
    
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    
    # Naive Bayes
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.bayes_theorem_example()
    nb_model, nb_accuracy = nb_classifier.train_naive_bayes(X, y)
    
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dt_model = SKDecisionTree(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    
    return nb_model, dt_model

# Run demonstration
nb_model, dt_model = demonstrate_classifiers()
```

---

## 🎤 Part 3: Interview Questions

### Q1: How does Naive Bayes work?
**Answer:**
Naive Bayes uses Bayes' theorem with the "naive" assumption that features are independent.

**Formula:** P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

**Steps:**
1. Calculate prior probabilities P(Class)
2. Calculate likelihoods P(Feature|Class) for each feature
3. Multiply all probabilities (assuming independence)
4. Normalize to get final probabilities

### Q2: What are the advantages and disadvantages of Decision Trees?
**Answer:**

**Advantages:**
- Easy to understand and interpret
- No assumptions about data distribution
- Handles both numerical and categorical data
- Automatic feature selection

**Disadvantages:**
- Prone to overfitting
- Unstable (small data changes = different tree)
- Biased toward features with more levels
- Can create overly complex trees

### Q3: When would you use Naive Bayes vs Decision Trees?
**Answer:**

**Use Naive Bayes when:**
- Features are relatively independent
- Text classification problems
- Small datasets
- Need probabilistic outputs
- Fast training/prediction needed

**Use Decision Trees when:**
- Need interpretable model
- Features have complex interactions
- Mixed data types (numerical + categorical)
- Don't need probabilistic outputs

---

## 📚 Key Concepts

### 1. **Bayes' Theorem:**
- Foundation of Naive Bayes
- Updates probabilities with new evidence
- P(A|B) = P(B|A) × P(A) / P(B)

### 2. **Decision Tree Splitting:**
- Information Gain (based on entropy)
- Gini Impurity
- Chi-square test

### 3. **Model Comparison:**
- Naive Bayes: Fast, works well with small data
- Decision Trees: Interpretable, handles interactions

---

*Focus on understanding the assumptions and trade-offs of each algorithm!* 🎯