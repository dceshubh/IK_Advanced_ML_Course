# Modern ML Architectures Study Guide - Week 26
*Explaining Modern ML Architectures like to a smart 12-year old, then diving into technical details*

## 🎯 Table of Contents
1. [Simple Explanations with Illustrations](#simple-explanations)
2. [Technical Deep Dive](#technical-concepts)
3. [Interview Questions & Answers](#interview-questions)

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

#### 3. Generative Adversarial Networks (GANs) 🥊
**Simple:** "Two AI systems competing to get better"

**Real-life analogy:** Art forger vs Art detective
- **Generator (Forger):** Tries to create fake paintings
- **Discriminator (Detective):** Tries to spot fake paintings
- **Competition:** They keep getting better by competing!
- **Result:** Generator becomes so good it creates perfect "real" paintings##
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

#### Q4: What are the applications of autoencoders in practice?

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
3. **Dimensionality Reduction:** PCA vs t-SNE vs UMAP trade-offs
4. **Applications:** Denoising, compression, anomaly detection, generation

### Practical Implementation
- **Libraries:** PyTorch, scikit-learn, UMAP-learn
- **Datasets:** MNIST, CIFAR-10, CelebA for image tasks
- **Visualization:** matplotlib, seaborn for embeddings

### Next Steps
- **Advanced Architectures:** β-VAE, WAE, AAE (Adversarial Autoencoders)
- **Generative Models:** GANs, Diffusion Models, Flow-based Models
- **Applications:** Build recommendation systems, anomaly detectors

---

*This study guide covers the fundamental concepts from Week 26's Modern ML Architectures session. Understanding these architectures is crucial for modern AI applications!*