# Unsupervised Learning Study Guide - Week 16
*Explaining Unsupervised Learning concepts like to a smart 12-year old, then diving into technical details*

## 🎯 Table of Contents
1. [Simple Explanations with Illustrations](#simple-explanations)
2. [Technical Deep Dive](#technical-concepts)
3. [Interview Questions & Answers](#interview-questions)

---

## 🌟 Simple Explanations with Illustrations {#simple-explanations}

### What is Unsupervised Learning?

**For a 12-year old:**
Imagine you have a huge box of mixed LEGO pieces - different colors, shapes, and sizes all jumbled together. You don't have any instructions or labels telling you what each piece is for. But you want to organize them somehow.

**Supervised Learning** would be like having a friend who already knows LEGO and tells you: "This red piece goes with castle sets, this blue piece goes with space sets."

**Unsupervised Learning** is like figuring out the patterns yourself! You might notice:
- "Hey, all these red and brown pieces seem similar - maybe they go together!"
- "These blue and gray pieces look like they belong in the same group!"
- "I can organize these by size, or by color, or by shape!"

### The Key Difference: Labels vs No Labels

**Supervised Learning:**
```
🐶 → "Dog"     (We tell the computer this is a dog)
🐱 → "Cat"     (We tell the computer this is a cat)
🐦 → "Bird"    (We tell the computer this is a bird)
```

**Unsupervised Learning:**
```
🐶 🐱 🐦 → "Find patterns yourself!"
```
The computer has to figure out: "These three things are different somehow, let me group them!"

### Main Types of Unsupervised Learning

#### 1. Clustering 🎯
**Simple:** "Group similar things together"

**Real-life analogy:** Organizing your music playlist
- Without labels: Computer listens to all your songs and says "These 50 songs sound similar, these 30 songs are different"
- You discover: "Oh! It grouped all my rock songs together, and all my classical music together!"

**Examples:**
- Customer segmentation: "These customers buy similar things"
- Gene analysis: "These genes behave similarly"
- Social media: "These people have similar interests"

#### 2. Dimensionality Reduction 📏
**Simple:** "Keep the important stuff, throw away the noise"

**Real-life analogy:** Packing for vacation
- You have 100 things you want to bring
- Your suitcase only fits 10 things
- You pick the 10 most important things that represent everything you need

**In data terms:**
- You have 1000 features describing something
- Most features are just noise or repetitive
- Keep the 10 most important features that capture the essence

#### 3. Anomaly Detection 🚨
**Simple:** "Find the weird one out"

**Real-life analogy:** Finding Waldo
- In a crowd of normally dressed people
- Waldo stands out because he's different (red striped shirt)
- Computer learns what "normal" looks like, then spots the unusual

**Examples:**
- Fraud detection: "This credit card transaction is weird"
- Medical diagnosis: "This patient's test results are unusual"
- Quality control: "This product is defective"

### Why Do We Need Unsupervised Learning?

#### The Real World Problem
**For a 12-year old:** Imagine you're a detective, but nobody tells you what clues to look for. You have to:
1. **Explore** the crime scene yourself
2. **Find patterns** in the evidence
3. **Group** similar clues together
4. **Discover** what's important vs what's just noise

**In the real world:**
- Most data doesn't come with labels
- Labeling data is expensive and time-consuming
- Sometimes we don't even know what we're looking for!
- We want to discover hidden patterns

### Dimensionality Reduction - The Compression Analogy

**Simple Explanation:**
Think of your favorite song. The original might be a huge file, but when you compress it to MP3:
- It becomes much smaller
- It still sounds almost the same
- You removed the "unnecessary" parts
- You kept the "essence" of the song

**In machine learning:**
- Original data: 1000 features (like a huge uncompressed file)
- After dimensionality reduction: 10 features (like a compressed MP3)
- Still captures the important patterns
- Much faster to process

### The Curse of Dimensionality

**Simple Analogy:** The "Too Many Choices" Problem

Imagine you're at an ice cream shop:
- **2 flavors:** Easy to choose! Vanilla or chocolate?
- **10 flavors:** Still manageable, you can compare them
- **1000 flavors:** Overwhelming! You can't even see them all, takes forever to decide, you might make a bad choice

**For computers:**
- **Few features:** Easy to find patterns
- **Many features:** Gets confused, takes forever, makes mistakes
- **Solution:** Reduce to the most important flavors (features)!

---

## 🔬 Technical Deep Dive {#technical-concepts}

### Fundamental Concepts

#### Supervised vs Unsupervised Learning

**Supervised Learning:**
- **Input:** Features (X) + Labels (y)
- **Goal:** Learn mapping f: X → y
- **Examples:** Classification, Regression
- **Training data:** {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}

**Unsupervised Learning:**
- **Input:** Features (X) only, no labels
- **Goal:** Discover hidden structure in data
- **Examples:** Clustering, Dimensionality Reduction, Density Estimation
- **Training data:** {x₁, x₂, ..., xₙ}

### Clustering Algorithms

#### K-Means Clustering

**Algorithm Steps:**
1. **Initialize:** Choose k cluster centers randomly
2. **Assignment:** Assign each point to nearest cluster center
3. **Update:** Move cluster centers to mean of assigned points
4. **Repeat:** Until convergence

**Mathematical Formulation:**
```
Objective: Minimize Σᵢ Σⱼ ||xᵢ - μⱼ||² 
where μⱼ is the centroid of cluster j
```

**Key Parameters:**
- **k:** Number of clusters (must be specified)
- **Distance metric:** Usually Euclidean distance
- **Initialization:** K-means++, random, or manual

**Advantages:**
- Simple and fast
- Works well with spherical clusters
- Scales well to large datasets

**Disadvantages:**
- Must specify k in advance
- Sensitive to initialization
- Assumes spherical clusters
- Sensitive to outliers

#### Hierarchical Clustering

**Two Approaches:**

**1. Agglomerative (Bottom-up):**
- Start with each point as its own cluster
- Repeatedly merge closest clusters
- Continue until one cluster remains

**2. Divisive (Top-down):**
- Start with all points in one cluster
- Repeatedly split clusters
- Continue until each point is its own cluster

**Linkage Criteria:**
- **Single linkage:** Distance between closest points
- **Complete linkage:** Distance between farthest points
- **Average linkage:** Average distance between all pairs
- **Ward linkage:** Minimizes within-cluster variance

**Advantages:**
- No need to specify number of clusters
- Creates dendrogram showing hierarchy
- Deterministic results

**Disadvantages:**
- Computationally expensive O(n³)
- Sensitive to noise and outliers
- Difficult to handle large datasets

### Dimensionality Reduction

#### Principal Component Analysis (PCA)

**Core Concept:**
PCA finds the directions (principal components) along which data varies the most.

**Mathematical Foundation:**

**1. Covariance Matrix:**
```
C = (1/n) Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ
where μ is the mean vector
```

**2. Eigenvalue Decomposition:**
```
C = QΛQᵀ
where Q contains eigenvectors, Λ contains eigenvalues
```

**3. Principal Components:**
- Eigenvectors of covariance matrix
- Ordered by corresponding eigenvalues (largest first)
- Each PC is orthogonal to others

**Algorithm Steps:**
1. **Standardize** data (mean=0, std=1)
2. **Compute** covariance matrix
3. **Find** eigenvalues and eigenvectors
4. **Sort** eigenvectors by eigenvalues (descending)
5. **Select** top k eigenvectors
6. **Transform** data to new coordinate system

**Variance Explained:**
```
Variance explained by PC i = λᵢ / Σⱼ λⱼ
Cumulative variance = Σᵢ₌₁ᵏ λᵢ / Σⱼ λⱼ
```

#### Eigenvalues and Eigenvectors

**Definition:**
For a square matrix A, vector v is an eigenvector with eigenvalue λ if:
```
Av = λv
```

**Geometric Interpretation:**
- **Eigenvector:** Direction that doesn't change under transformation
- **Eigenvalue:** How much the eigenvector is scaled

**In PCA Context:**
- **Eigenvectors:** Principal component directions
- **Eigenvalues:** Amount of variance in each direction
- **Larger eigenvalue:** More important direction

**Properties:**
- Eigenvectors of symmetric matrices are orthogonal
- Eigenvalues are real for symmetric matrices
- Sum of eigenvalues = trace of matrix
- Product of eigenvalues = determinant of matrix

#### Singular Value Decomposition (SVD)

**Mathematical Formulation:**
```
A = UΣVᵀ
where:
- U: Left singular vectors (m×m orthogonal matrix)
- Σ: Diagonal matrix of singular values
- V: Right singular vectors (n×n orthogonal matrix)
```

**Relationship to PCA:**
- **AᵀA = VΣ²Vᵀ:** Eigenvectors of AᵀA are columns of V
- **AAᵀ = UΣ²Uᵀ:** Eigenvectors of AAᵀ are columns of U
- **Singular values:** σᵢ = √λᵢ where λᵢ are eigenvalues

**Advantages of SVD:**
- Works with non-square matrices
- More numerically stable than eigendecomposition
- Handles rank-deficient matrices
- Foundation for many ML algorithms

### Advanced Concepts

#### Curse of Dimensionality

**Mathematical Perspective:**

**1. Volume Concentration:**
In high dimensions, most volume of a hypersphere is near the surface:
```
Volume ratio = (1 - ε)ᵈ → 0 as d → ∞
```

**2. Distance Concentration:**
All points become equidistant in high dimensions:
```
lim(d→∞) (max_dist - min_dist) / min_dist = 0
```

**3. Sample Sparsity:**
To maintain density, sample size must grow exponentially:
```
N ∝ (1/ε)ᵈ for ε-dense sampling
```

**Practical Implications:**
- **Nearest neighbors:** All points become equally "near"
- **Clustering:** Difficult to distinguish clusters
- **Visualization:** Cannot plot high-dimensional data directly
- **Overfitting:** Models memorize noise instead of patterns

#### Covariance Matrix

**Definition:**
```
Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]
```

**Matrix Form:**
```
C[i,j] = Cov(Xᵢ, Xⱼ)
```

**Properties:**
- **Diagonal elements:** Variances of individual features
- **Off-diagonal elements:** Covariances between feature pairs
- **Symmetric:** C[i,j] = C[j,i]
- **Positive semi-definite:** All eigenvalues ≥ 0

**Interpretation:**
- **Large positive covariance:** Features increase together
- **Large negative covariance:** One increases, other decreases
- **Near-zero covariance:** Features are uncorrelated

### Evaluation Metrics

#### Clustering Evaluation

**Internal Metrics (No ground truth needed):**

**1. Silhouette Score:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
where:
- a(i): Average distance to points in same cluster
- b(i): Average distance to points in nearest cluster
Range: [-1, 1], higher is better
```

**2. Inertia (Within-cluster sum of squares):**
```
WCSS = Σᵢ Σⱼ∈Cᵢ ||xⱼ - μᵢ||²
Lower is better
```

**3. Calinski-Harabasz Index:**
```
CH = (SSB/(k-1)) / (SSW/(n-k))
where SSB = between-cluster sum of squares
      SSW = within-cluster sum of squares
Higher is better
```

**External Metrics (Ground truth available):**

**1. Adjusted Rand Index (ARI):**
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
Range: [-1, 1], higher is better
```

**2. Normalized Mutual Information (NMI):**
```
NMI = MI(U,V) / √(H(U)H(V))
Range: [0, 1], higher is better
```

#### Dimensionality Reduction Evaluation

**1. Explained Variance Ratio:**
```
EVR = Σᵢ₌₁ᵏ λᵢ / Σⱼ₌₁ᵈ λⱼ
```

**2. Reconstruction Error:**
```
RE = ||X - X̂||²F / ||X||²F
where X̂ is reconstructed data
```

**3. Downstream Task Performance:**
- Train model on reduced data
- Compare performance to original data
- Good reduction preserves task-relevant information

---

## 🎤 Interview Questions & Detailed Answers {#interview-questions}

### Fundamental Concepts

#### Q1: What is the main difference between supervised and unsupervised learning? Provide examples of each.

**Answer:**

**Key Difference:**
- **Supervised Learning:** Uses labeled training data (input-output pairs)
- **Unsupervised Learning:** Uses only input data without labels

**Supervised Learning Examples:**
1. **Classification:** Email spam detection (emails labeled as spam/not spam)
2. **Regression:** House price prediction (houses with known prices)
3. **Object Recognition:** Image classification (images labeled with object names)

**Unsupervised Learning Examples:**
1. **Clustering:** Customer segmentation (group customers by behavior patterns)
2. **Dimensionality Reduction:** Data visualization (reduce 1000 features to 2D plot)
3. **Anomaly Detection:** Fraud detection (identify unusual transactions)
4. **Association Rules:** Market basket analysis (find items bought together)

**Mathematical Representation:**
```
Supervised: Given {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}, learn f: X → Y
Unsupervised: Given {x₁, x₂, ..., xₙ}, discover structure in X
```

**When to Use Each:**
- **Supervised:** When you have clear target variable and labeled data
- **Unsupervised:** When exploring data, no clear target, or labeling is expensive

#### Q2: Explain the curse of dimensionality and how dimensionality reduction helps address it.

**Answer:**

**Curse of Dimensionality Definition:**
The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces that don't occur in low-dimensional settings.

**Key Problems:**

**1. Exponential Growth of Space:**
- Volume of d-dimensional unit hypercube = 1
- Volume of d-dimensional unit hypersphere = π^(d/2) / Γ(d/2 + 1)
- As d increases, most volume concentrates near the boundary

**2. Distance Concentration:**
```
In high dimensions: max_dist/min_dist → 1
All points become equidistant, making nearest neighbor meaningless
```

**3. Sample Sparsity:**
- To maintain same density, need exponentially more samples
- Required samples ∝ (1/ε)^d for ε-dense coverage

**4. Computational Complexity:**
- Storage: O(d) per sample
- Distance computation: O(d) per pair
- Many algorithms scale poorly with d

**How Dimensionality Reduction Helps:**

**1. Preserves Important Information:**
- PCA keeps directions with highest variance
- Retains 95%+ of information with much fewer dimensions

**2. Improves Algorithm Performance:**
- Faster training and inference
- Better generalization (less overfitting)
- More meaningful distance metrics

**3. Enables Visualization:**
- Reduce to 2D/3D for human interpretation
- Discover hidden patterns and clusters

**4. Noise Reduction:**
- Removes irrelevant features
- Focuses on signal vs noise

**Example:**
```python
# Original: 1000 features, 10,000 samples
# After PCA: 50 features, same samples
# Result: 20x faster, better accuracy, visualizable
```

#### Q3: Compare and contrast K-means and hierarchical clustering algorithms.

**Answer:**

**K-Means Clustering:**

**Algorithm:**
1. Choose k cluster centers randomly
2. Assign points to nearest center
3. Update centers to cluster means
4. Repeat until convergence

**Advantages:**
- **Computational Efficiency:** O(nkt) where t is iterations
- **Scalability:** Works well with large datasets
- **Simplicity:** Easy to implement and understand
- **Guaranteed Convergence:** Always converges to local minimum

**Disadvantages:**
- **Must specify k:** Need to know number of clusters
- **Initialization Sensitivity:** Different starts give different results
- **Cluster Shape Assumption:** Assumes spherical, similar-sized clusters
- **Outlier Sensitivity:** Outliers can skew cluster centers

**Hierarchical Clustering:**

**Algorithm:**
- **Agglomerative:** Start with n clusters, merge closest pairs
- **Divisive:** Start with 1 cluster, recursively split

**Advantages:**
- **No k specification:** Don't need to choose number of clusters
- **Dendrogram:** Shows hierarchy of cluster relationships
- **Deterministic:** Same result every time
- **Flexible Shapes:** Can handle non-spherical clusters

**Disadvantages:**
- **Computational Cost:** O(n³) for naive implementation
- **No Global Objective:** Greedy decisions can't be undone
- **Sensitivity to Noise:** Single outlier can affect entire hierarchy
- **Scalability Issues:** Difficult for large datasets

**Comparison Table:**
```
Aspect              | K-Means        | Hierarchical
--------------------|----------------|---------------
Time Complexity     | O(nkt)         | O(n³)
Space Complexity    | O(nk)          | O(n²)
Number of Clusters  | Must specify   | Automatic
Cluster Shape       | Spherical      | Any shape
Deterministic       | No             | Yes
Scalability         | High           | Low
Interpretability    | Medium         | High (dendrogram)
```

**When to Use:**
- **K-Means:** Large datasets, known k, spherical clusters
- **Hierarchical:** Small datasets, unknown k, need hierarchy, irregular shapes

### Advanced Topics

#### Q4: Explain Principal Component Analysis (PCA). How does it work mathematically?

**Answer:**

**PCA Overview:**
PCA is a dimensionality reduction technique that finds orthogonal directions (principal components) that capture maximum variance in the data.

**Mathematical Foundation:**

**Step 1: Data Standardization**
```
X_standardized = (X - μ) / σ
where μ is mean vector, σ is standard deviation vector
```

**Step 2: Covariance Matrix**
```
C = (1/(n-1)) * X^T * X
C[i,j] = Cov(feature_i, feature_j)
```

**Step 3: Eigendecomposition**
```
C * v_i = λ_i * v_i
where v_i are eigenvectors, λ_i are eigenvalues
```

**Step 4: Principal Components**
- Eigenvectors = Principal component directions
- Eigenvalues = Variance explained by each component
- Sort by eigenvalues (descending)

**Step 5: Dimensionality Reduction**
```
X_reduced = X * V_k
where V_k contains top k eigenvectors
```

**Variance Explained:**
```
Variance explained by PC_i = λ_i / Σ(all λ_j)
Cumulative variance = Σ(λ_1 to λ_k) / Σ(all λ_j)
```

**Geometric Interpretation:**
- **First PC:** Direction of maximum variance
- **Second PC:** Direction of maximum remaining variance (orthogonal to first)
- **Subsequent PCs:** Continue this pattern

**Key Properties:**
1. **Orthogonality:** All PCs are perpendicular
2. **Variance Ordering:** PC1 > PC2 > ... > PCn in terms of variance
3. **Linear Transformation:** PCA is a rotation of coordinate system
4. **Dimensionality Reduction:** Keep top k components

**Example Application:**
```python
from sklearn.decomposition import PCA

# Original data: 1000 features
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)
print(f"Reduced to {pca.n_components_} features")
```

#### Q5: What are eigenvalues and eigenvectors? Why are they important in machine learning?

**Answer:**

**Mathematical Definition:**
For a square matrix A, vector v is an eigenvector with eigenvalue λ if:
```
A * v = λ * v
```

**Geometric Interpretation:**
- **Eigenvector (v):** Direction that remains unchanged under transformation A
- **Eigenvalue (λ):** Scaling factor applied to eigenvector

**Visual Example:**
```
If A represents stretching:
- Eigenvector: Direction of stretching axis
- Eigenvalue: Amount of stretching (λ > 1) or shrinking (λ < 1)
```

**Properties:**
1. **Orthogonality:** Eigenvectors of symmetric matrices are orthogonal
2. **Real Values:** Eigenvalues of symmetric matrices are real
3. **Trace:** Sum of eigenvalues = trace of matrix
4. **Determinant:** Product of eigenvalues = determinant of matrix

**Importance in Machine Learning:**

**1. Principal Component Analysis (PCA):**
- Eigenvectors of covariance matrix = principal components
- Eigenvalues = variance explained by each component
- Dimensionality reduction by selecting top eigenvalues

**2. Spectral Clustering:**
- Eigenvectors of graph Laplacian reveal cluster structure
- Number of zero eigenvalues = number of connected components

**3. PageRank Algorithm:**
- Dominant eigenvector of transition matrix
- Represents steady-state probability distribution

**4. Singular Value Decomposition (SVD):**
- Eigendecomposition of A^T*A and A*A^T
- Foundation for matrix factorization techniques

**5. Stability Analysis:**
- Eigenvalues determine system stability
- Used in optimization and neural network analysis

**Computational Considerations:**
```python
import numpy as np

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort by eigenvalues (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

**Real-World Applications:**
- **Image Compression:** PCA using eigenfaces
- **Recommendation Systems:** SVD for matrix factorization
- **Network Analysis:** Community detection using spectral methods
- **Quantum Mechanics:** Energy states as eigenvectors

#### Q6: How do you choose the optimal number of clusters in K-means?

**Answer:**

**The Challenge:**
K-means requires specifying k (number of clusters) beforehand, but optimal k is often unknown.

**Methods for Choosing k:**

**1. Elbow Method:**
```python
# Plot WCSS vs k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Look for "elbow" in the plot
plt.plot(range(1, 11), wcss)
```

**Interpretation:**
- Plot Within-Cluster Sum of Squares (WCSS) vs k
- Look for "elbow" where rate of decrease slows
- **Limitation:** Elbow not always clear

**2. Silhouette Analysis:**
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Choose k with highest silhouette score
```

**Silhouette Score Formula:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
where:
- a(i): Average distance to points in same cluster
- b(i): Average distance to points in nearest cluster
Range: [-1, 1], higher is better
```

**3. Gap Statistic:**
```
Gap(k) = E[log(W_k)] - log(W_k)
where:
- W_k: Within-cluster dispersion for k clusters
- E[log(W_k)]: Expected value under null distribution
```

**Algorithm:**
1. Cluster original data with k clusters
2. Generate random data with same distribution
3. Cluster random data with k clusters
4. Compare dispersions
5. Choose k where gap is largest

**4. Information Criteria:**
- **AIC (Akaike Information Criterion):** AIC = 2k - 2ln(L)
- **BIC (Bayesian Information Criterion):** BIC = k*ln(n) - 2ln(L)
- Lower values indicate better fit

**5. Cross-Validation:**
```python
# Split data, cluster on training, evaluate on validation
def cv_kmeans(X, k_range):
    scores = []
    for k in k_range:
        cv_scores = []
        for train_idx, val_idx in kfold.split(X):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X[train_idx])
            score = kmeans.score(X[val_idx])
            cv_scores.append(score)
        scores.append(np.mean(cv_scores))
    return scores
```

**6. Domain Knowledge:**
- Business requirements
- Theoretical expectations
- Practical constraints

**Best Practices:**
1. **Try Multiple Methods:** Different methods may suggest different k
2. **Consider Context:** Business needs often constrain k
3. **Validate Results:** Check if clusters make sense
4. **Stability Analysis:** Run multiple times, check consistency
5. **Hierarchical First:** Use dendrogram to get rough estimate

**Example Workflow:**
```python
# 1. Quick estimate with elbow method
plot_elbow_curve(X)

# 2. Refine with silhouette analysis
best_k = find_best_silhouette(X, range(2, 10))

# 3. Validate with gap statistic
gap_k = gap_statistic(X, range(1, 10))

# 4. Final decision considering domain knowledge
final_k = integrate_domain_knowledge(best_k, gap_k, business_needs)
```

#### Q7: Explain the difference between PCA and SVD. When would you use each?

**Answer:**

**Mathematical Relationship:**

**PCA (Principal Component Analysis):**
- Based on eigendecomposition of covariance matrix
- **Covariance Matrix:** C = (1/n) * X^T * X
- **Eigendecomposition:** C = Q * Λ * Q^T
- **Principal Components:** Columns of Q (eigenvectors)

**SVD (Singular Value Decomposition):**
- Direct decomposition of data matrix
- **Decomposition:** X = U * Σ * V^T
- **Components:** V contains right singular vectors

**Key Relationship:**
```
X^T * X = V * Σ² * V^T  (eigendecomposition of X^T*X)
X * X^T = U * Σ² * U^T  (eigendecomposition of X*X^T)

PCA eigenvectors = SVD right singular vectors (V)
PCA eigenvalues = SVD singular values squared (σ²)
```

**Detailed Comparison:**

**Computational Aspects:**

**PCA:**
```python
# Traditional PCA
C = np.cov(X.T)  # Compute covariance matrix
eigenvals, eigenvecs = np.linalg.eig(C)
```

**SVD:**
```python
# SVD approach
U, s, Vt = np.linalg.svd(X, full_matrices=False)
# V = Vt.T contains principal components
```

**Advantages and Disadvantages:**

**PCA Advantages:**
- **Interpretable:** Directly works with covariance
- **Statistical Meaning:** Clear variance interpretation
- **Standard Approach:** Well-established in statistics

**PCA Disadvantages:**
- **Numerical Issues:** Computing covariance matrix can be unstable
- **Memory Requirements:** Covariance matrix is d×d
- **Rank Limitations:** Problems when n < d

**SVD Advantages:**
- **Numerical Stability:** More robust to numerical errors
- **Memory Efficient:** Works directly with data matrix
- **Handles Rank Deficiency:** Works when n < d
- **Broader Applications:** Foundation for many algorithms

**SVD Disadvantages:**
- **Less Intuitive:** Not directly interpretable as variance
- **Computational Cost:** Can be expensive for large matrices

**When to Use Each:**

**Use PCA When:**
1. **Statistical Interpretation Needed:** Want to understand variance explained
2. **Small Datasets:** n >> d, covariance matrix is manageable
3. **Educational/Research:** Need to explain variance concepts
4. **Feature Analysis:** Understanding feature relationships important

**Use SVD When:**
1. **Large Datasets:** High-dimensional data (d large)
2. **Numerical Stability Critical:** Avoiding covariance matrix computation
3. **Rank-Deficient Data:** n < d scenarios
4. **Matrix Factorization:** Building recommender systems, NLP applications
5. **Production Systems:** Need robust, efficient implementation

**Practical Examples:**

**PCA Use Cases:**
```python
# Exploratory data analysis
pca = PCA()
pca.fit(X)
explained_variance = pca.explained_variance_ratio_
print(f"First PC explains {explained_variance[0]:.2%} of variance")

# Feature selection based on variance
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)
```

**SVD Use Cases:**
```python
# Recommender system (matrix factorization)
U, s, Vt = svd(user_item_matrix)
recommendations = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Text analysis (LSA - Latent Semantic Analysis)
U, s, Vt = svd(term_document_matrix)
topic_vectors = Vt[:k, :]  # k topics

# Image compression
U, s, Vt = svd(image_matrix)
compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

**Performance Considerations:**

**Time Complexity:**
- **PCA:** O(d³) for eigendecomposition + O(nd²) for covariance
- **SVD:** O(min(nd², n²d)) for full SVD

**Space Complexity:**
- **PCA:** O(d²) for covariance matrix
- **SVD:** O(nd) for data matrix

**Modern Implementations:**
```python
# Scikit-learn automatically uses SVD for PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # Uses randomized SVD internally

# Explicit SVD for custom applications
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
```

#### Q8: How would you detect and handle outliers in unsupervised learning?

**Answer:**

**Outlier Detection Methods:**

**1. Statistical Methods:**

**Z-Score Method:**
```python
from scipy import stats
z_scores = np.abs(stats.zscore(X))
outliers = (z_scores > 3).any(axis=1)
```
- **Assumption:** Data is normally distributed
- **Threshold:** Typically |z| > 3 (99.7% rule)
- **Limitation:** Sensitive to distribution assumptions

**Interquartile Range (IQR):**
```python
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
```
- **Robust:** Less sensitive to extreme values
- **Non-parametric:** No distribution assumptions
- **Interpretable:** Based on quartiles

**2. Distance-Based Methods:**

**K-Nearest Neighbors (KNN):**
```python
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X)
distances, indices = knn.kneighbors(X)
outlier_scores = distances.mean(axis=1)
outliers = outlier_scores > np.percentile(outlier_scores, 95)
```
- **Intuition:** Outliers have large distances to neighbors
- **Parameter:** Choice of k affects sensitivity
- **Scalability:** Can be slow for large datasets

**Local Outlier Factor (LOF):**
```python
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outlier_labels = lof.fit_predict(X)
outlier_scores = lof.negative_outlier_factor_
```
- **Local Density:** Compares local density to neighbors
- **Handles Clusters:** Works with varying densities
- **Score Interpretation:** LOF > 1 indicates outlier

**3. Clustering-Based Methods:**

**DBSCAN:**
```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
outliers = labels == -1  # Points labeled as noise
```
- **Density-Based:** Finds outliers as noise points
- **No Assumptions:** Works with arbitrary cluster shapes
- **Parameters:** eps (neighborhood size), min_samples

**Isolation Forest:**
```python
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(X)
outlier_scores = iso_forest.decision_function(X)
```
- **Tree-Based:** Isolates outliers with fewer splits
- **Efficient:** Linear time complexity
- **High-Dimensional:** Works well in high dimensions

**4. Dimensionality Reduction Methods:**

**PCA-Based Detection:**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)
reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)
outliers = reconstruction_error > np.percentile(reconstruction_error, 95)
```
- **Reconstruction Error:** Outliers have high reconstruction error
- **Dimensionality:** Effective for high-dimensional data
- **Assumption:** Normal data lies on lower-dimensional manifold

**Handling Strategies:**

**1. Removal:**
```python
# Remove outliers completely
X_clean = X[~outliers]
```
**When to Use:** Clear data quality issues, sufficient remaining data
**Risks:** Loss of information, potential bias

**2. Transformation:**
```python
# Log transformation for skewed data
X_log = np.log1p(X)

# Winsorization (cap extreme values)
from scipy.stats.mstats import winsorize
X_winsorized = winsorize(X, limits=[0.05, 0.05], axis=0)

# Robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

**3. Robust Methods:**
```python
# Use robust clustering algorithms
from sklearn.cluster import DBSCAN
# DBSCAN naturally handles outliers as noise

# Robust PCA
from sklearn.decomposition import PCA
# Use robust covariance estimation
from sklearn.covariance import EllipticEnvelope
```

**4. Separate Analysis:**
```python
# Analyze outliers separately
normal_data = X[~outliers]
outlier_data = X[outliers]

# Different models for different groups
normal_model = fit_model(normal_data)
outlier_model = fit_model(outlier_data)
```

**Best Practices:**

**1. Multiple Methods:**
```python
# Combine multiple detection methods
z_outliers = detect_zscore_outliers(X)
lof_outliers = detect_lof_outliers(X)
iso_outliers = detect_isolation_outliers(X)

# Consensus approach
consensus_outliers = (z_outliers & lof_outliers) | iso_outliers
```

**2. Domain Knowledge:**
- **Business Rules:** Some "outliers" may be valuable (VIP customers)
- **Data Collection:** Understanding measurement errors
- **Temporal Patterns:** Seasonal effects, trends

**3. Validation:**
```python
# Visual inspection
plt.scatter(X[:, 0], X[:, 1], c=outlier_labels)
plt.title("Outlier Detection Results")

# Statistical validation
print(f"Outlier percentage: {outliers.mean():.2%}")
print(f"Outlier statistics: {X[outliers].describe()}")
```

**4. Iterative Process:**
```python
# Iterative outlier removal
for iteration in range(max_iterations):
    outliers = detect_outliers(X)
    if outliers.sum() < threshold:
        break
    X = X[~outliers]
```

**Evaluation Metrics:**
- **Precision/Recall:** If ground truth available
- **Silhouette Score:** Impact on clustering quality
- **Reconstruction Error:** For dimensionality reduction
- **Business Metrics:** Impact on downstream tasks

---

## 📚 Additional Resources

### Key Concepts to Master
1. **Clustering:** K-means, Hierarchical, DBSCAN
2. **Dimensionality Reduction:** PCA, SVD, t-SNE
3. **Linear Algebra:** Eigenvalues, eigenvectors, covariance matrices
4. **Evaluation:** Silhouette score, elbow method, explained variance

### Practical Implementation
- **Libraries:** scikit-learn, numpy, scipy
- **Datasets:** Iris, Wine, Customer segmentation data
- **Visualization:** matplotlib, seaborn for cluster analysis

### Next Steps
- **Week 17:** Advanced Unsupervised Learning (DBSCAN, Gaussian Mixture Models)
- **Advanced Topics:** Manifold learning, Autoencoders, Generative models

---

*This study guide covers the fundamental concepts from Week 16's Unsupervised Learning session. Practice implementing these algorithms and understanding when to apply each technique!*