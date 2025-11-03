# Modern ML Architectures Study Guide - Week 26
*Comprehensive study material covering Autoencoders, VAEs, and Graph Neural Networks*

## 🎯 Table of Contents
1. [Simple Explanations with Illustrations](#simple-explanations)
2. [Technical Deep Dive](#technical-concepts)
3. [Live Class Key Points](#live-class-notes)
4. [Interview Questions & Answers](#interview-questions)
5. [Practical Implementation Guide](#implementation-guide)

---

## 🌟 Simple Explanations with Illustrations {#simple-explanations}

### What are Autoencoders?

**For a 12-year old:**
Imagine you have a huge LEGO castle, but you need to pack it in a small box for moving. An autoencoder is like:

1. **Taking apart the castle** (Encoder) - Breaking it down to essential pieces
2. **Packing efficiently** (Latent Space) - Fitting everything in a small box
3. **Rebuilding the castle** (Decoder) - Reconstructing from the packed pieces

**The magic:** The autoencoder learns to keep only the most important information needed to rebuild perfectly!

### Types of Modern Architectures

#### 1. Autoencoders 🔄
**Simple:** "Compress and decompress to learn what's important"

**Real-life analogy:** Like a really smart photocopier
- **Input:** Original photo
- **Encoder:** Compresses to essential features  
- **Decoder:** Recreates the photo from compressed features
- **Goal:** Make the copy as close to original as possible

**Applications:**
- **Noise removal:** Clean up blurry photos
- **Compression:** Store images in less space
- **Anomaly detection:** Find unusual patterns

#### 2. Variational Autoencoders (VAEs) 🎲
**Simple:** "Autoencoders that can create new things"

**Real-life analogy:** Like an artist who learns to paint
- **Learns the "style"** of many paintings (encoding)
- **Understands the "essence"** of what makes a good painting
- **Creates new paintings** in the same style (generation)

**Key difference from regular autoencoders:**
- **Regular:** Can only recreate what it's seen
- **VAE:** Can create brand new examples!

#### 3. Graph Neural Networks (GNNs) 🕸️
**Simple:** "Understanding relationships between connected things"

**Real-life analogy:** Social media friend recommendations
- **Nodes:** People in the network
- **Edges:** Friendships between people
- **GNN Magic:** "If you're friends with Alice and Bob, and Alice likes jazz music, maybe you'll like jazz too!"
- **Learning:** The network learns that connected people often have similar interests

**Applications:**
- **Social networks:** Friend recommendations, community detection
- **Citation networks:** Paper recommendations, research area classification
- **Molecular analysis:** Drug discovery, protein folding prediction

#### 4. Generative Adversarial Networks (GANs) 🥊
**Simple:** "Two AI systems competing to get better"

**Real-life analogy:** Art forger vs Art detective
- **Generator (Forger):** Tries to create fake paintings
- **Discriminator (Detective):** Tries to spot fake paintings
- **Competition:** They keep getting better by competing!
- **Result:** Generator becomes so good it creates perfect "real" paintings
# Dimensionality Reduction Techniques

#### 1. PCA (Principal Component Analysis) 📊
**Simple:** "Find the most important directions in your data"

**Real-life analogy:** Describing a person
- Instead of listing 100 features (height, weight, eye color, etc.)
- Find the 3 most important features that capture most information
- **Example:** "Tall, athletic, friendly" captures 80% of what makes them unique

#### 2. t-SNE 🗺️
**Simple:** "Create a map where similar things are close together"

**Real-life analogy:** Organizing a school cafeteria
- Students who are friends sit close together
- Different friend groups spread out
- **Result:** You can see all the social clusters at a glance!

#### 3. UMAP 🚀
**Simple:** "Like t-SNE but faster and better at preserving relationships"

**Real-life analogy:** Google Maps vs old paper maps
- **t-SNE (paper map):** Accurate but slow to create
- **UMAP (Google Maps):** Fast, accurate, and shows multiple zoom levels

### Unsupervised Learning Concepts

#### What is Unsupervised Learning?
**Simple:** "Learning patterns without being told what to look for"

**Supervised Learning:** Like studying with answer sheets
```
Input: Photo of cat → Label: "Cat" ✓
Input: Photo of dog → Label: "Dog" ✓
```

**Unsupervised Learning:** Like exploring without a guide
```
Input: Bunch of photos → Find patterns yourself!
Result: "These photos seem to group into furry animals, vehicles, and buildings"
```

#### Common Unsupervised Techniques:
- **K-Means Clustering:** Group similar things together
- **DBSCAN:** Find clusters of any shape, ignore outliers
- **Hierarchical Clustering:** Build a family tree of similarities

---

## 📚 Live Class Key Points {#live-class-notes}

### Unsupervised Learning Fundamentals

**What is Unsupervised Learning?**
- Machine learning approach where **no target variable** is provided
- Algorithm must find patterns and relationships in data **without supervision**
- More common in real-world applications than supervised learning
- Target variables are often difficult or expensive to obtain

**Key Difference from Supervised Learning:**
```
Supervised:   Features + Target → Prediction
Unsupervised: Features only → Discover Patterns
```

**Common Unsupervised Algorithms:**
- **K-Means Clustering**: Groups data into k clusters
- **DBSCAN**: Density-based clustering, finds clusters of any shape
- **Hierarchical Clustering**: Creates tree-like cluster structures
- **Gaussian Mixture Models**: Probabilistic clustering approach

### Dimensionality Reduction Deep Dive

**Why Dimensionality Reduction Matters:**
- **Curse of Dimensionality**: Too many features can hurt model performance
- **Computational Efficiency**: Fewer features = faster training
- **Visualization**: Reduce to 2D/3D for human understanding
- **Noise Reduction**: Remove irrelevant or redundant features
- **Storage**: Compress data while preserving information

**The Goal:**
Take 10,000 features → Reduce to 100 features while retaining 95% of information

#### Principal Component Analysis (PCA) - Mathematical Foundation

**PCA Algorithm Steps:**
1. **Compute Covariance Matrix**: C = (1/n) × X^T × X
2. **Eigenvalue Decomposition**: C = Q × Λ × Q^T
3. **Sort Eigenvalues**: λ₁ ≥ λ₂ ≥ ... ≥ λₙ
4. **Select Components**: Choose top k eigenvectors
5. **Transform Data**: X_reduced = X × Q_k

**Why PCA is Linear:**
- **Mathematical Mapping**: T(x) = Ax (matrix multiplication)
- **Satisfies Linearity**: T(x + y) = T(x) + T(y)
- **Scalar Property**: T(αx) = αT(x)
- **Linear Projection**: Projects data onto lower-dimensional subspace

**Interview Question - Prove PCA Linearity:**
```
Given: T(x) = Ax where A is the projection matrix
Prove: T(x + y) = T(x) + T(y)

Solution:
T(x + y) = A(x + y)     [Definition]
         = Ax + Ay      [Matrix distributivity]
         = T(x) + T(y)  [Definition]
```

#### Advanced Dimensionality Reduction Techniques

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- **Nonlinear**: Captures complex manifold structures
- **Local Focus**: Preserves local neighborhoods excellently
- **Visualization**: Best for 2D/3D visualization
- **Computational Cost**: O(n²) - expensive for large datasets

**UMAP (Uniform Manifold Approximation and Projection):**
- **Faster than t-SNE**: O(n log n) complexity
- **Global Structure**: Preserves both local and global relationships
- **Scalable**: Works with larger datasets
- **Flexible**: Can reduce to any number of dimensions

---

## 🔬 Technical Deep Dive {#technical-concepts}

### Autoencoder Architecture

#### Basic Autoencoder Structure

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder: Compress input to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder: Reconstruct from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For normalized inputs
        )
    
    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
```

#### Training Objective

**Reconstruction Loss:**
```python
def autoencoder_loss(model, x):
    reconstructed, latent = model(x)
    
    # Mean Squared Error between input and reconstruction
    mse_loss = F.mse_loss(reconstructed, x)
    
    return mse_loss

# Alternative: Binary Cross-Entropy for binary data
def bce_loss(model, x):
    reconstructed, latent = model(x)
    return F.binary_cross_entropy(reconstructed, x)
```

#### Mathematical Formulation

**Encoder Function:**
```
z = f_encoder(x; θ_e)
where z ∈ R^d_latent, x ∈ R^d_input, d_latent << d_input
```

**Decoder Function:**
```
x̂ = f_decoder(z; θ_d)
where x̂ ∈ R^d_input (reconstruction)
```

**Objective:**
```
min L(x, x̂) = ||x - f_decoder(f_encoder(x; θ_e); θ_d)||²
θ_e,θ_d
```

### Variational Autoencoders (VAEs)

#### Key Innovation: Probabilistic Latent Space

Instead of deterministic encoding, VAEs learn a probability distribution:

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder outputs mean and log-variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Separate heads for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
```

#### VAE Loss Function

**ELBO (Evidence Lower BOund):**
```python
def vae_loss(model, x, beta=1.0):
    reconstructed, mu, logvar = model(x)
    
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(reconstructed, x, reduction='sum')
    
    # KL divergence loss (regularization)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss (ELBO)
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
```

**Mathematical Formulation:**
```
L_VAE = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

where:
- q(z|x) = N(μ(x), σ²(x)) (encoder distribution)
- p(z) = N(0, I) (prior distribution)
- p(x|z) (decoder distribution)
```

#### Reparameterization Trick

**Problem:** Cannot backpropagate through random sampling
**Solution:** Reparameterize sampling as deterministic function + noise

```python
# Instead of: z ~ N(μ, σ²)
# Use: z = μ + σ * ε, where ε ~ N(0, 1)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)  # σ = exp(log(σ²)/2)
    eps = torch.randn_like(std)    # ε ~ N(0, 1)
    return mu + eps * std          # z = μ + σ * ε
```### Adv
anced Dimensionality Reduction

#### UMAP (Uniform Manifold Approximation and Projection)

**Key Advantages over t-SNE:**
- **Faster computation:** O(n log n) vs O(n²)
- **Preserves global structure:** Better at maintaining overall data relationships
- **Scalable:** Works with larger datasets
- **Flexible:** Can reduce to any number of dimensions

**Mathematical Foundation:**
```python
# UMAP preserves topological structure using:
# 1. Fuzzy simplicial sets
# 2. Category theory
# 3. Riemannian geometry

import umap

# Basic usage
reducer = umap.UMAP(
    n_neighbors=15,      # Local neighborhood size
    min_dist=0.1,        # Minimum distance in embedding
    n_components=2,      # Target dimensions
    metric='euclidean'   # Distance metric
)

embedding = reducer.fit_transform(high_dim_data)
```

#### Comparison of Dimensionality Reduction Techniques

| Method | Type | Speed | Global Structure | Local Structure | Parameters |
|--------|------|-------|------------------|-----------------|------------|
| **PCA** | Linear | Fast | Good | Poor | Few |
| **t-SNE** | Nonlinear | Slow | Poor | Excellent | Many |
| **UMAP** | Nonlinear | Fast | Good | Good | Moderate |

### Graph Neural Networks (GNNs)

#### Graph Convolutional Networks (GCNs)

**Core Concept:** Extend convolution to graph-structured data

```python
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second graph convolution
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

#### Message Passing Framework

**Mathematical Foundation:**
```
h_i^(l+1) = UPDATE(h_i^(l), AGGREGATE({h_j^(l) : j ∈ N(i)}))
```

**GCN Specific Formula:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

where:
- Ã = A + I (adjacency matrix + self-loops)
- D̃ = degree matrix of Ã
- H^(l) = node features at layer l
- W^(l) = learnable weight matrix
- σ = activation function
```

#### GNN Applications

**1. Node Classification:**
```python
# Predict node labels (e.g., paper categories)
def node_classification_loss(model, data):
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    return loss
```

**2. Link Prediction:**
```python
# Predict missing edges in graph
class LinkPredictor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def decode(self, z, edge_label_index):
        # Dot product of node embeddings
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        return self.decode(z, edge_label_index)
```

**3. Graph Classification:**
```python
# Classify entire graphs
from torch_geometric.nn import global_mean_pool

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.gnn = GCN(input_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Node-level representations
        node_embeddings = self.gnn(x, edge_index)
        
        # Graph-level representation
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        # Classification
        return self.classifier(graph_embedding)
```

#### Advanced GNN Architectures

**GraphSAGE (Sample and Aggregate):**
```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

**Graph Attention Networks (GAT):**
```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=0.6)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## 💻 Practical Implementation Guide {#implementation-guide}

### Building Your First Autoencoder

**Step 1: Data Preprocessing**
```python
# Normalize data to [0,1] range
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

**Step 2: Encoder Architecture**
```python
# Progressive compression
encoder_input = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
x = MaxPooling2D((2, 2))(x)  # 14x14
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)  # 7x7
encoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
```

**Step 3: Decoder Architecture**
```python
# Progressive reconstruction
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)  # 14x14
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)  # 28x28
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
```

### VAE Implementation Tips

**Reparameterization Trick:**
```python
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

**Custom Loss Function:**
```python
def vae_loss(x, x_decoded, z_mean, z_log_var):
    # Reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded))
    
    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    )
    
    return reconstruction_loss + kl_loss
```

### Graph Neural Network Setup

**Data Preparation:**
```python
# Convert to PyTorch Geometric format
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T

# Load graph data
data = from_networkx(networkx_graph)

# Add node features (if not available)
if data.x is None:
    data.x = torch.eye(data.num_nodes)  # Identity matrix as features

# Apply transforms
transform = T.Compose([
    T.NormalizeFeatures(),  # Normalize node features
    T.RandomNodeSplit(num_val=0.1, num_test=0.2)  # Create train/val/test splits
])
data = transform(data)
```

**GCN Implementation:**
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training loop
def train_gnn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute loss only on training nodes
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Evaluation
def evaluate_gnn(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        accuracy = correct / mask.sum().item()
    return accuracy
```

**Link Prediction Setup:**
```python
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

class LinkPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = GCN(input_dim, hidden_dim, hidden_dim)
    
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def decode(self, z, edge_label_index):
        # Dot product of node embeddings
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

# Training for link prediction
def train_link_prediction(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # Generate negative samples
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1)
    )
    
    # Combine positive and negative edges
    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(data.edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ])
    
    # Forward pass
    out = model(data.x, data.edge_index, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, edge_label)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## 🎤 Interview Questions & Detailed Answers {#interview-questions}

### Fundamental Concepts

#### Q1: Explain the difference between autoencoders and VAEs. When would you use each?

**Answer:**

**Autoencoders:**
- **Goal:** Learn efficient data representation through reconstruction
- **Latent space:** Deterministic encoding
- **Training:** Minimize reconstruction error
- **Use case:** Dimensionality reduction, denoising, compression

**Variational Autoencoders (VAEs):**
- **Goal:** Learn probabilistic generative model
- **Latent space:** Probabilistic encoding (mean + variance)
- **Training:** Minimize reconstruction error + KL divergence
- **Use case:** Data generation, interpolation, semi-supervised learning

**Key Differences:**

**1. Latent Space Structure:**
```python
# Autoencoder: Deterministic encoding
z = encoder(x)  # Single point in latent space

# VAE: Probabilistic encoding  
mu, logvar = encoder(x)
z = mu + exp(0.5 * logvar) * epsilon  # Distribution in latent space
```

**2. Loss Functions:**
```python
# Autoencoder loss
loss_ae = ||x - decoder(encoder(x))||²

# VAE loss (ELBO)
loss_vae = ||x - decoder(z)||² + KL(q(z|x) || p(z))
```

**3. Generative Capability:**
```python
# Autoencoder: Cannot generate new samples
# (latent space may have gaps/discontinuities)

# VAE: Can generate new samples
z_new = torch.randn(batch_size, latent_dim)  # Sample from prior
x_new = decoder(z_new)  # Generate new data
```

**When to Use Each:**

**Use Autoencoders When:**
- **Dimensionality reduction** for downstream tasks
- **Denoising** corrupted data
- **Compression** for storage/transmission
- **Feature learning** for representation
- **Anomaly detection** (reconstruction error)

**Use VAEs When:**
- **Data generation** (creating new samples)
- **Data interpolation** (smooth transitions)
- **Semi-supervised learning** (few labeled examples)
- **Disentangled representations** (controllable generation)
- **Probabilistic modeling** (uncertainty quantification)

#### Q6: Compare different unsupervised learning algorithms and their use cases.

**Answer:**

**Clustering Algorithms Comparison:**

| Algorithm | Type | Advantages | Disadvantages | Best Use Case |
|-----------|------|------------|---------------|---------------|
| **K-Means** | Centroid-based | Fast, simple, works well with spherical clusters | Requires k specification, sensitive to outliers | Customer segmentation, image compression |
| **DBSCAN** | Density-based | Finds arbitrary shapes, handles outliers, auto-determines clusters | Sensitive to hyperparameters, struggles with varying densities | Anomaly detection, spatial data analysis |
| **Hierarchical** | Tree-based | No k specification needed, creates dendrogram | O(n³) complexity, sensitive to noise | Taxonomy creation, phylogenetic analysis |
| **Gaussian Mixture** | Probabilistic | Soft clustering, handles overlapping clusters | Assumes Gaussian distributions, requires k | Image segmentation, speech recognition |

**Dimensionality Reduction Comparison:**

| Method | Type | Speed | Global Structure | Local Structure | Best For |
|--------|------|-------|------------------|-----------------|----------|
| **PCA** | Linear | Very Fast | Good | Poor | Preprocessing, noise reduction |
| **t-SNE** | Nonlinear | Slow | Poor | Excellent | Visualization, exploration |
| **UMAP** | Nonlinear | Fast | Good | Good | Large datasets, production systems |

**Selection Criteria:**
- **Data size**: PCA for large datasets, t-SNE for small
- **Cluster shape**: K-means for spherical, DBSCAN for arbitrary
- **Interpretability**: PCA for interpretable components
- **Visualization**: t-SNE/UMAP for 2D/3D plots

#### Q7: Explain Graph Neural Networks and their key applications.

**Answer:**

**What are Graph Neural Networks?**
Graph Neural Networks are deep learning models designed to work with graph-structured data, where relationships between entities are as important as the entities themselves.

**Key Components:**
1. **Nodes**: Individual entities (users, papers, molecules)
2. **Edges**: Relationships between entities (friendships, citations, bonds)
3. **Node Features**: Attributes of each node
4. **Edge Features**: Attributes of relationships (optional)

**Message Passing Framework:**
```python
# General GNN operation
for layer in range(num_layers):
    for node_i in graph.nodes:
        # Collect messages from neighbors
        messages = []
        for neighbor_j in graph.neighbors(node_i):
            message = MESSAGE(node_i.features, neighbor_j.features, edge_ij.features)
            messages.append(message)
        
        # Aggregate messages
        aggregated = AGGREGATE(messages)  # sum, mean, max, etc.
        
        # Update node representation
        node_i.features = UPDATE(node_i.features, aggregated)
```

**GCN Mathematical Formulation:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

where:
- H^(l) = node features at layer l
- Ã = A + I (adjacency + self-loops)
- D̃ = degree matrix of Ã
- W^(l) = learnable parameters
- σ = activation function
```

**Key Applications:**

**1. Social Networks:**
- **Friend recommendation**: Predict missing connections
- **Community detection**: Find groups of similar users
- **Influence analysis**: Identify key opinion leaders
- **Content recommendation**: Leverage social connections

**2. Citation Networks:**
- **Paper classification**: Categorize research papers
- **Citation prediction**: Predict future citations
- **Research trend analysis**: Identify emerging topics
- **Author disambiguation**: Resolve author identities

**3. Molecular Analysis:**
- **Drug discovery**: Predict molecular properties
- **Protein folding**: Understand 3D structure
- **Chemical reaction prediction**: Predict reaction outcomes
- **Toxicity assessment**: Evaluate drug safety

**4. Knowledge Graphs:**
- **Entity linking**: Connect entities across databases
- **Relation extraction**: Discover new relationships
- **Question answering**: Navigate knowledge for answers
- **Recommendation systems**: Leverage entity relationships

**Advantages of GNNs:**
- **Relational reasoning**: Incorporates structural information
- **Permutation invariant**: Order of nodes doesn't matter
- **Inductive learning**: Can generalize to unseen graphs
- **Interpretable**: Can analyze learned attention weights

**Challenges:**
- **Over-smoothing**: Deep GNNs may lose node distinctions
- **Scalability**: Large graphs require efficient implementations
- **Heterophily**: Performance drops when connected nodes are dissimilar
- **Dynamic graphs**: Handling temporal changes in structure

#### Q8: Compare different GNN architectures and their use cases.

**Answer:**

**GNN Architecture Comparison:**

| Architecture | Key Innovation | Strengths | Weaknesses | Best Use Case |
|--------------|----------------|-----------|------------|---------------|
| **GCN** | Spectral convolution | Simple, fast, interpretable | Fixed receptive field, over-smoothing | Node classification, small graphs |
| **GraphSAGE** | Sampling + aggregation | Scalable, inductive | Sampling may lose information | Large graphs, inductive tasks |
| **GAT** | Attention mechanism | Adaptive weights, interpretable | Computational overhead | Heterogeneous graphs, explainability |
| **GIN** | Injective aggregation | Theoretically powerful | May overfit on small graphs | Graph classification, molecular data |

**Detailed Comparison:**

**Graph Convolutional Networks (GCN):**
```python
# Simple and effective for homophilic graphs
class GCN(nn.Module):
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)  # Aggregate neighbors
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```
- **Best for**: Citation networks, social networks with homophily
- **Limitations**: Assumes similar neighbors, fixed aggregation

**GraphSAGE:**
```python
# Scalable to large graphs through sampling
class GraphSAGE(nn.Module):
    def forward(self, x, edge_index):
        # Sample fixed number of neighbors
        x = self.conv1(x, edge_index)  # Mean aggregation
        x = F.relu(x)
        return self.conv2(x, edge_index)
```
- **Best for**: Large-scale graphs, inductive learning
- **Innovation**: Neighbor sampling for scalability

**Graph Attention Networks (GAT):**
```python
# Learn attention weights for different neighbors
class GAT(nn.Module):
    def forward(self, x, edge_index):
        # Compute attention weights
        x, attention = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        return self.conv2(x, edge_index)
```
- **Best for**: Heterogeneous graphs, interpretability needed
- **Innovation**: Adaptive neighbor weighting

**Graph Isomorphism Networks (GIN):**
```python
# Maximally powerful for graph classification
class GIN(nn.Module):
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)  # (1 + ε) * x + Σ neighbors
        x = F.relu(x)
        return global_add_pool(x, batch)  # Graph-level representation
```
- **Best for**: Graph classification, molecular property prediction
- **Innovation**: Theoretically most expressive aggregation

**Selection Guidelines:**

**Choose GCN when:**
- Graph exhibits homophily (similar nodes connect)
- Computational resources are limited
- Interpretability is important
- Graph is relatively small

**Choose GraphSAGE when:**
- Graph is very large (millions of nodes)
- Need inductive learning (new nodes at test time)
- Memory constraints are important
- Graph structure changes over time

**Choose GAT when:**
- Graph is heterogeneous (different node/edge types)
- Need to understand which neighbors are important
- Nodes have rich feature representations
- Explainability is crucial

**Choose GIN when:**
- Task is graph-level classification
- Working with molecular or chemical data
- Need maximum expressive power
- Graph structure is the primary signal

#### Q2: How does the reparameterization trick work in VAEs and why is it necessary?

**Answer:**

**The Problem:**
In VAEs, we need to sample from the learned distribution q(z|x) = N(μ(x), σ²(x)), but sampling is not differentiable, preventing backpropagation.

**Mathematical Challenge:**
```python
# This doesn't work for backpropagation:
mu, logvar = encoder(x)
sigma = exp(0.5 * logvar)
z = sample_from_normal(mu, sigma)  # Non-differentiable!
reconstructed = decoder(z)
```

**The Reparameterization Trick:**

**Core Idea:** Express random sampling as deterministic function + external noise:
```
Instead of: z ~ N(μ, σ²)
Use: z = μ + σ * ε, where ε ~ N(0, 1)
```

**Implementation:**
```python
def reparameterize(mu, logvar):
    """
    mu: mean of latent distribution [batch_size, latent_dim]
    logvar: log variance of latent distribution [batch_size, latent_dim]
    """
    # Compute standard deviation
    std = torch.exp(0.5 * logvar)  # σ = exp(log(σ²)/2)
    
    # Sample noise from standard normal
    eps = torch.randn_like(std)    # ε ~ N(0, 1)
    
    # Reparameterized sample
    z = mu + eps * std             # z = μ + σ * ε
    
    return z
```

**Why This Works:**

**1. Differentiability:**
```python
# Gradients can flow through deterministic operations
∂z/∂μ = 1                    # Direct gradient path
∂z/∂σ = ε                    # Gradient through std
∂σ/∂logvar = 0.5 * exp(0.5 * logvar)  # Chain rule
```

**2. Stochasticity Preserved:**
```python
# z still follows N(μ, σ²) distribution
# But now it's expressed as deterministic function of μ, σ, and ε
E[z] = E[μ + σ * ε] = μ + σ * E[ε] = μ + σ * 0 = μ
Var[z] = Var[σ * ε] = σ² * Var[ε] = σ² * 1 = σ²
```

**3. Gradient Estimation:**
```python
# Monte Carlo gradient estimation becomes possible
∇_θ E_q(z|x)[f(z)] = E_q(z|x)[∇_θ f(z)]  # Before: intractable
                    = E_p(ε)[∇_θ f(μ + σ * ε)]  # After: tractable
```

**Complete VAE Forward Pass:**
```python
def vae_forward(self, x):
    # Encode to distribution parameters
    h = self.encoder_hidden(x)
    mu = self.encoder_mu(h)
    logvar = self.encoder_logvar(h)
    
    # Reparameterized sampling
    z = self.reparameterize(mu, logvar)
    
    # Decode
    reconstructed = self.decoder(z)
    
    return reconstructed, mu, logvar, z
```

**Training Loop:**
```python
def train_vae(model, dataloader, optimizer, beta=1.0):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, mu, logvar, z = model(data)
        
        # Compute loss
        recon_loss = F.binary_cross_entropy(reconstructed, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + beta * kl_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)
```

**Alternative Reparameterization for Other Distributions:**

**Bernoulli (for binary latent variables):**
```python
def reparameterize_bernoulli(logits, temperature=1.0):
    """Gumbel-Softmax reparameterization for discrete variables"""
    eps = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(eps + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / temperature
    return torch.sigmoid(y)
```

**Beta Distribution:**
```python
def reparameterize_beta(alpha, beta):
    """Reparameterization for Beta distribution"""
    # Use Gamma reparameterization: Beta(α,β) = Gamma(α)/(Gamma(α)+Gamma(β))
    gamma1 = reparameterize_gamma(alpha)
    gamma2 = reparameterize_gamma(beta)
    return gamma1 / (gamma1 + gamma2)
```

#### Q3: Compare PCA, t-SNE, and UMAP for dimensionality reduction. When would you use each?

**Answer:**

**Detailed Comparison:**

**Principal Component Analysis (PCA):**

**Mathematical Foundation:**
```python
# PCA finds principal components as eigenvectors of covariance matrix
C = (1/n) * X.T @ X  # Covariance matrix
eigenvals, eigenvecs = torch.linalg.eig(C)

# Transform data
X_reduced = X @ eigenvecs[:, :k]  # Keep top k components
```

**Characteristics:**
- **Linear transformation:** Preserves linear relationships
- **Global method:** Considers entire dataset simultaneously
- **Deterministic:** Same result every time
- **Fast:** O(min(n³, d³)) complexity
- **Interpretable:** Components have clear meaning

**When to Use PCA:**
- **Preprocessing:** Before applying other ML algorithms
- **Noise reduction:** When data has Gaussian noise
- **Feature selection:** When features are correlated
- **Visualization:** Quick 2D/3D plots
- **Large datasets:** When speed is critical

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**

**Mathematical Foundation:**
```python
# t-SNE minimizes KL divergence between high-D and low-D distributions
# High-dimensional similarities (Gaussian)
p_ij = exp(-||x_i - x_j||² / 2σ²) / Σ_k exp(-||x_i - x_k||² / 2σ²)

# Low-dimensional similarities (t-distribution)  
q_ij = (1 + ||y_i - y_j||²)^(-1) / Σ_k (1 + ||y_i - y_k||²)^(-1)

# Objective: minimize KL(P || Q)
```

**Characteristics:**
- **Nonlinear:** Can capture complex manifold structures
- **Local focus:** Preserves local neighborhoods excellently
- **Stochastic:** Different runs give different results
- **Slow:** O(n²) complexity
- **2D/3D only:** Typically used for visualization

**When to Use t-SNE:**
- **Data exploration:** Understanding cluster structure
- **Visualization:** Creating publication-quality plots
- **Cluster validation:** Verifying clustering results
- **Manifold discovery:** When data lies on nonlinear manifold
- **Small-medium datasets:** When computational cost acceptable

**UMAP (Uniform Manifold Approximation and Projection):**

**Mathematical Foundation:**
```python
# UMAP uses topological data analysis and fuzzy sets
# Constructs fuzzy topological representation
# Optimizes layout in low-dimensional space

import umap

reducer = umap.UMAP(
    n_neighbors=15,        # Local neighborhood size
    min_dist=0.1,         # Minimum distance between points
    n_components=2,       # Output dimensions
    metric='euclidean'    # Distance metric
)

embedding = reducer.fit_transform(data)
```

**Characteristics:**
- **Nonlinear:** Captures complex structures like t-SNE
- **Fast:** O(n log n) complexity
- **Scalable:** Works with large datasets
- **Flexible:** Any number of output dimensions
- **Preserves both:** Local and global structure

**When to Use UMAP:**
- **Large datasets:** When t-SNE is too slow
- **Production systems:** When speed matters
- **Higher dimensions:** Reducing to >3 dimensions
- **Balanced preservation:** Need both local and global structure
- **Interactive analysis:** Real-time dimensionality reduction

**Practical Decision Framework:**

```python
def choose_dim_reduction_method(data_size, target_dims, goal):
    """Decision framework for dimensionality reduction"""
    
    if goal == 'preprocessing':
        return 'PCA'  # Fast, interpretable
    
    elif goal == 'visualization':
        if data_size < 10000:
            return 't-SNE'  # Best visualization quality
        else:
            return 'UMAP'   # Faster alternative
    
    elif goal == 'feature_extraction':
        if target_dims > 3:
            return 'PCA' if linear_relationships else 'UMAP'
        else:
            return 'UMAP'
    
    elif goal == 'anomaly_detection':
        return 'PCA'  # Reconstruction error interpretable
    
    else:
        return 'UMAP'  # Generally good default

# Usage example
method = choose_dim_reduction_method(
    data_size=50000, 
    target_dims=2, 
    goal='visualization'
)
print(f"Recommended method: {method}")
```

**Hyperparameter Guidelines:**

**PCA:**
```python
# Choose number of components
pca = PCA()
pca.fit(X)
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumsum_var >= 0.95) + 1  # 95% variance
```

**t-SNE:**
```python
# Key parameters
tsne = TSNE(
    perplexity=30,        # 5-50, balance local/global
    learning_rate=200,    # 10-1000, affects convergence
    n_iter=1000,         # More iterations = better results
    early_exaggeration=12 # Emphasizes clusters early
)
```

**UMAP:**
```python
# Key parameters
umap_reducer = umap.UMAP(
    n_neighbors=15,       # 2-100, local neighborhood size
    min_dist=0.1,        # 0.0-0.99, minimum separation
    spread=1.0,          # Effective scale of embedded points
    n_epochs=200         # Training iterations
)
```

#### Q4: Prove mathematically that PCA is a linear transformation.

**Answer:**

**Given:** PCA transformation T(x) = Ax, where A is the projection matrix containing principal components.

**To Prove:** T is linear, meaning it satisfies:
1. T(x + y) = T(x) + T(y) (additivity)
2. T(αx) = αT(x) (homogeneity)

**Proof:**

**Part 1 - Additivity:**
```
T(x + y) = A(x + y)           [Definition of T]
         = Ax + Ay            [Matrix distributivity]
         = T(x) + T(y)        [Definition of T]
```

**Part 2 - Homogeneity:**
```
T(αx) = A(αx)                 [Definition of T]
      = α(Ax)                 [Scalar multiplication property]
      = αT(x)                 [Definition of T]
```

**Therefore:** PCA is linear because it's fundamentally matrix multiplication, which preserves linear relationships.

**Why This Matters:**
- **Interpretability**: Linear transformations preserve relative distances
- **Computational Efficiency**: Matrix operations are highly optimized
- **Mathematical Properties**: Enables theoretical analysis and guarantees

#### Q5: What are the applications of autoencoders in practice?

**Answer:**

**1. Dimensionality Reduction**

```python
class DimensionalityReductionAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

# Use latent representations for downstream tasks
def extract_features(model, data):
    model.eval()
    with torch.no_grad():
        latent_features = model.encoder(data)
    return latent_features
```

**2. Denoising**

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, noise_factor=0.3):
        super().__init__()
        self.noise_factor = noise_factor
        # ... encoder and decoder definitions
    
    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        return torch.clamp(x + noise, 0, 1)
    
    def forward(self, x):
        # Add noise to input
        noisy_x = self.add_noise(x)
        
        # Encode noisy input
        latent = self.encoder(noisy_x)
        
        # Decode to clean output
        clean_x = self.decoder(latent)
        
        return clean_x, latent

# Training with clean targets
def train_denoising_ae(model, clean_data):
    noisy_data = model.add_noise(clean_data)
    reconstructed, _ = model(noisy_data)
    loss = F.mse_loss(reconstructed, clean_data)  # Compare to clean
    return loss
```

**3. Anomaly Detection**

```python
class AnomalyDetectionAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # ... standard autoencoder architecture
        
    def anomaly_score(self, x):
        """Compute reconstruction error as anomaly score"""
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            # Higher reconstruction error = more anomalous
            error = F.mse_loss(reconstructed, x, reduction='none')
            return error.mean(dim=1)  # Per-sample error

# Usage for fraud detection
def detect_anomalies(model, transactions, threshold_percentile=95):
    scores = model.anomaly_score(transactions)
    threshold = torch.quantile(scores, threshold_percentile / 100)
    anomalies = scores > threshold
    return anomalies, scores
```

**4. Data Compression**

```python
class CompressionAE(nn.Module):
    def __init__(self, input_dim, compression_ratio=10):
        super().__init__()
        latent_dim = input_dim // compression_ratio
        
        # Encoder: Compress data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, latent_dim)
        )
        
        # Decoder: Decompress data
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

def compress_and_store(model, data):
    """Compress data for storage"""
    model.eval()
    with torch.no_grad():
        compressed = model.encoder(data)
    return compressed

def decompress_and_retrieve(model, compressed_data):
    """Decompress stored data"""
    model.eval()
    with torch.no_grad():
        decompressed = model.decoder(compressed_data)
    return decompressed
```

**5. Feature Learning for Transfer Learning**

```python
class PretrainedFeatureExtractor(nn.Module):
    def __init__(self, pretrained_autoencoder):
        super().__init__()
        # Use encoder as feature extractor
        self.feature_extractor = pretrained_autoencoder.encoder
        
        # Freeze pretrained weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.feature_extractor(x)

# Transfer learning pipeline
def transfer_learning_with_ae(pretrained_ae, downstream_data, num_classes):
    # Extract features using pretrained encoder
    feature_extractor = PretrainedFeatureExtractor(pretrained_ae)
    
    # Build classifier on top
    classifier = nn.Sequential(
        feature_extractor,
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    
    return classifier
```

**6. Data Augmentation**

```python
class VariationalDataAugmentation:
    def __init__(self, vae_model):
        self.vae = vae_model
        
    def augment_dataset(self, original_data, augmentation_factor=2):
        """Generate additional training samples using VAE"""
        augmented_data = []
        
        for x in original_data:
            # Encode to latent space
            mu, logvar = self.vae.encode(x.unsqueeze(0))
            
            # Sample multiple times from learned distribution
            for _ in range(augmentation_factor):
                z = self.vae.reparameterize(mu, logvar)
                # Add small perturbation for diversity
                z_perturbed = z + 0.1 * torch.randn_like(z)
                
                # Decode to generate new sample
                x_new = self.vae.decode(z_perturbed)
                augmented_data.append(x_new.squeeze(0))
        
        return torch.stack(augmented_data)
```

**Real-World Applications:**

**1. Computer Vision:**
- **Image denoising:** Medical imaging, satellite imagery
- **Image compression:** Lossy compression with neural networks
- **Style transfer:** Learning artistic styles

**2. Natural Language Processing:**
- **Document embeddings:** Semantic document representation
- **Text generation:** Variational text autoencoders
- **Language modeling:** Unsupervised pretraining

**3. Recommender Systems:**
- **Collaborative filtering:** User-item interaction modeling
- **Content-based filtering:** Item feature learning
- **Cold start problem:** Handling new users/items

**4. Time Series:**
- **Anomaly detection:** Equipment failure prediction
- **Forecasting:** Learning temporal patterns
- **Compression:** Efficient time series storage

**5. Healthcare:**
- **Medical imaging:** Tumor detection, image enhancement
- **Drug discovery:** Molecular representation learning
- **Electronic health records:** Patient similarity modeling

---

## 📚 Additional Resources

### Key Concepts to Master
1. **Autoencoder Architecture:** Encoder-decoder structure, latent space
2. **VAE Theory:** Reparameterization trick, ELBO loss, probabilistic modeling
3. **Graph Neural Networks:** Message passing, node/link/graph-level tasks
4. **Dimensionality Reduction:** PCA vs t-SNE vs UMAP trade-offs
5. **Applications:** Denoising, compression, anomaly detection, generation, social networks

### Practical Implementation
- **Libraries:** PyTorch, PyTorch Geometric, scikit-learn, UMAP-learn
- **Datasets:** MNIST, CIFAR-10, CelebA for images; Cora, CiteSeer for graphs
- **Visualization:** matplotlib, seaborn for embeddings, NetworkX for graphs

### Next Steps
- **Advanced Architectures:** β-VAE, WAE, AAE (Adversarial Autoencoders)
- **Advanced GNNs:** Graph Transformers, Heterogeneous GNNs, Temporal GNNs
- **Generative Models:** GANs, Diffusion Models, Flow-based Models
- **Applications:** Build recommendation systems, anomaly detectors, social network analyzers

---

*This study guide covers the fundamental concepts from Week 26's Modern ML Architectures session. Understanding these architectures is crucial for modern AI applications!*