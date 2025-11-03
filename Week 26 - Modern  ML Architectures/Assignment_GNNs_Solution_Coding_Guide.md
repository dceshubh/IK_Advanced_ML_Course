# Assignment - GNNs (Graph Neural Networks) Solution - Coding Guide

## Overview
This notebook demonstrates **Graph Neural Networks (GNNs)** for **link prediction** and **node clustering** on the Facebook social network dataset. It uses Graph Convolutional Networks (GCNs) to learn node embeddings and predict missing connections in the social graph.

## Key Learning Objectives
- Understand Graph Neural Network architecture and applications
- Implement link prediction using GCNs
- Learn node embeddings for social network analysis
- Apply clustering on learned graph embeddings
- Work with real-world social network data

---

## 1. Library Installation and Imports

```python
# Install PyTorch Geometric for graph neural networks
!pip3 install torch_geometric igraph

# Core imports
from google.colab import drive
from torch_geometric.utils import from_networkx
import torch
import numpy as np
import torch_geometric.transforms as T
import networkx as nx
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import igraph
```

### Why These Libraries?
- **torch_geometric**: PyTorch extension for graph neural networks
- **igraph**: Efficient graph processing and analysis
- **networkx**: Python graph library for graph manipulation
- **GCNConv**: Graph Convolutional Network layer implementation
- **negative_sampling**: Generate negative edges for link prediction training
- **roc_auc_score**: Evaluation metric for binary classification (link exists/doesn't exist)

---

## 2. Data Loading and Preprocessing

### 2.1 Facebook Social Network Dataset

```python
# Download Facebook combined social network dataset
!wget -P /content/gdrive/MyDrive/gnn-data https://snap.stanford.edu/data/facebook_combined.txt.gz
!cd /content/gdrive/MyDrive/gnn-data && gunzip facebook_combined.txt.gz

# Load graph using igraph and convert to NetworkX
ig_graph = igraph.Graph.Read_Edgelist('/content/gdrive/MyDrive/gnn-data/facebook_combined.txt', directed=False)
nx_graph = ig_graph.to_networkx()
data = from_networkx(nx_graph)
```

**Dataset Details:**
- **Facebook Combined**: Social circles from Facebook
- **Nodes**: Individual users in the network
- **Edges**: Friendship connections between users
- **Undirected**: Friendships are mutual relationships
- **Size**: ~4,000 nodes, ~88,000 edges

### 2.2 Feature Engineering

```python
# Create node features from adjacency matrix
adj_matrix = nx.adjacency_matrix(nx_graph)
data.x = torch.tensor(adj_matrix.toarray(), dtype=torch.float32)
```

**Key Insight:**
- **No initial node features**: Social network data often lacks node attributes
- **Adjacency matrix as features**: Each node's features = its connections to all other nodes
- **High-dimensional**: Feature dimension equals number of nodes (~4,000)
- **Sparse representation**: Most entries are 0 (not connected to most users)

**Why Use Adjacency Matrix?**
- **Structural information**: Captures local neighborhood structure
- **Homophily principle**: Connected nodes tend to be similar
- **Bootstrap features**: Provides initial signal for GNN to learn from

---

## 3. Data Splitting for Link Prediction

```python
transform = T.Compose([
    T.NormalizeFeatures(),  # Normalize node features
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])

train_data, val_data, test_data = transform(data)
```

### Transform Components:

**T.NormalizeFeatures():**
- Normalizes node features to unit norm
- Prevents features with large magnitudes from dominating
- Improves training stability

**T.RandomLinkSplit():**
- **num_val=0.05**: 5% of edges for validation
- **num_test=0.1**: 10% of edges for testing  
- **is_undirected=True**: Maintains undirected graph property
- **add_negative_train_samples=False**: Generate negative samples during training

**Link Prediction Setup:**
- **Positive samples**: Existing edges in the graph
- **Negative samples**: Non-existing edges (randomly sampled)
- **Task**: Predict whether an edge should exist between two nodes

---

## 4. Graph Neural Network Architecture

### 4.1 GCN-Based Encoder-Decoder Model

```python
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        """Encode nodes into embeddings"""
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
    def decode(self, z, edge_label_index):
        """Decode edge probabilities from node embeddings"""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def decode_all(self, z):
        """Decode all possible edges"""
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
```

### Architecture Breakdown:

**Encoder (GCN Layers):**
```python
# Layer 1: Input features → Hidden representation
x = self.conv1(x, edge_index).relu()
# Layer 2: Hidden → Final node embeddings  
return self.conv2(x, edge_index)
```

**GCN Layer Operation:**
```
h_i^(l+1) = σ(W^(l) * Σ(h_j^(l) / √(d_i * d_j)))
```
Where:
- **h_i^(l)**: Node i's representation at layer l
- **W^(l)**: Learnable weight matrix at layer l
- **σ**: Activation function (ReLU)
- **d_i, d_j**: Degrees of nodes i and j
- **Aggregation**: Sum over neighbors with normalization

**Decoder (Link Prediction):**
```python
# Element-wise multiplication + sum
return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
```
- **Intuition**: Similar embeddings → higher dot product → higher link probability
- **Output**: Scalar score for each edge (higher = more likely to exist)

---

## 5. Training Process

### 5.1 Model Configuration

```python
input_dimension = data.num_features    # ~4000 (adjacency matrix size)
hidden_dimension = 32                  # Compressed representation
output_dimension = 16                  # Final embedding size
num_epochs = 20
learning_rate = 0.01

model = Net(input_dimension, hidden_dimension, output_dimension)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()
```

**Hyperparameter Choices:**
- **Hidden dimension (32)**: Balances expressiveness vs efficiency
- **Output dimension (16)**: Final embedding size for downstream tasks
- **Adam optimizer**: Adaptive learning rates for stable training
- **BCEWithLogitsLoss**: Binary cross-entropy for link prediction

### 5.2 Training Loop

```python
def train():
    model.train()
    optimizer.zero_grad()
    
    # Encode all nodes
    z = model.encode(train_data.x, train_data.edge_index)
    
    # Generate negative samples for current epoch
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, 
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), 
        method='sparse'
    )
    
    # Combine positive and negative edges
    edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)
    
    # Predict edge probabilities
    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    
    loss.backward()
    optimizer.step()
    return loss
```

**Training Steps Explained:**

1. **Forward Pass**: Encode nodes → Get embeddings
2. **Negative Sampling**: Generate non-existing edges as negative examples
3. **Edge Prediction**: Decode edge probabilities from embeddings
4. **Loss Calculation**: Compare predictions with true labels
5. **Backpropagation**: Update model parameters

**Why Negative Sampling?**
- **Class imbalance**: Far more non-edges than edges in most graphs
- **Computational efficiency**: Don't need to consider all possible edges
- **Dynamic sampling**: Different negative samples each epoch for better generalization

### 5.3 Evaluation Function

```python
@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
```

**Evaluation Process:**
- **No gradients**: @torch.no_grad() for efficiency
- **Sigmoid activation**: Convert logits to probabilities [0,1]
- **ROC-AUC metric**: Area under ROC curve (0.5 = random, 1.0 = perfect)
- **Interpretation**: How well can we distinguish existing vs non-existing edges?

---

## 6. Model Training and Validation

```python
best_val_auc = final_test_auc = 0
for epoch in range(1, num_epochs):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')
```

**Training Strategy:**
- **Early stopping**: Save model with best validation performance
- **Monitoring**: Track loss and AUC scores during training
- **Final evaluation**: Report test performance of best validation model

**Expected Results:**
- **Random baseline**: AUC ≈ 0.5
- **Good performance**: AUC > 0.8
- **Excellent performance**: AUC > 0.9

---

## 7. Node Clustering Analysis

### 7.1 Extract Node Embeddings

```python
# Get final node embeddings
model.eval()
z = model.encode(data.x, data.edge_index)
z_np = z.cpu().detach().numpy()
```

**Embedding Properties:**
- **Dimensionality**: 16-dimensional vectors per node
- **Learned representations**: Capture both local and global graph structure
- **Similarity**: Nodes with similar embeddings likely to be connected or in same community

### 7.2 Clustering Analysis

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Test different numbers of clusters
min_clusters = 2
max_clusters = 50
silhouette_scores_test = []
within_sum = []

for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_assignments = kmeans.fit_predict(z_np)
    
    # Evaluate clustering quality
    silhouette_avg_test = silhouette_score(z_np, cluster_assignments)
    silhouette_scores_test.append(silhouette_avg_test)
    within_sum.append(kmeans.inertia_)
```

**Clustering Evaluation Metrics:**

**Silhouette Score:**
- **Range**: [-1, 1]
- **Interpretation**: 
  - Close to 1: Well-separated clusters
  - Close to 0: Overlapping clusters  
  - Negative: Poor clustering
- **Formula**: (b - a) / max(a, b)
  - a: Average distance to same cluster
  - b: Average distance to nearest cluster

**Within-Cluster Sum of Squares (Inertia):**
- **Lower is better**: Tighter clusters
- **Elbow method**: Look for "elbow" in the curve
- **Trade-off**: More clusters → lower inertia but potential overfitting

### 7.3 Optimal Cluster Selection

```python
# Elbow method visualization
plt.figure(figsize=(10,5))
plt.plot(range(min_clusters, max_clusters+1), within_sum)
plt.xlabel('Number of Clusters')
plt.ylabel('Within Sum')
plt.title('Elbow Plot')

# Highlight optimal point
plt.plot(17, within_sum[15], 'g^', markersize=12)
print('Optimal Clusters:', 17)
```

**Cluster Interpretation:**
- **Social communities**: Clusters likely represent friend groups or communities
- **Homophily**: People in same cluster share similar connections
- **Network structure**: Reflects underlying social organization

---

## 8. Advanced Concepts and Extensions

### 8.1 Graph Convolutional Networks (GCN) Theory

**Message Passing Framework:**
```
m_ij^(l) = Message(h_i^(l), h_j^(l), e_ij)
h_i^(l+1) = Update(h_i^(l), Aggregate({m_ij^(l) : j ∈ N(i)}))
```

**GCN Specific:**
- **Message**: Normalized neighbor features
- **Aggregation**: Weighted sum based on node degrees
- **Update**: Linear transformation + activation

### 8.2 Alternative GNN Architectures

**GraphSAGE:**
```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
```

**Graph Attention Networks (GAT):**
```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels*8, out_channels, heads=1, dropout=0.6)
```

### 8.3 Applications Beyond Link Prediction

**Node Classification:**
```python
# Predict node labels (e.g., user interests, demographics)
classifier = torch.nn.Linear(output_dimension, num_classes)
node_predictions = classifier(z)
```

**Graph Classification:**
```python
# Classify entire graphs (e.g., molecular properties)
from torch_geometric.nn import global_mean_pool
graph_embedding = global_mean_pool(z, batch)
```

**Recommendation Systems:**
```python
# User-item bipartite graphs for collaborative filtering
# Predict user preferences based on graph structure
```

---

## 9. Practical Considerations

### 9.1 Scalability Issues

**Large Graphs:**
- **Memory constraints**: Adjacency matrices become prohibitively large
- **Solution**: Use sparse representations, mini-batching, or sampling techniques

**GraphSAINT Sampling:**
```python
from torch_geometric.loader import GraphSAINTRandomWalkSampler

loader = GraphSAINTRandomWalkSampler(
    data, batch_size=6000, walk_length=2, num_steps=5
)
```

### 9.2 Feature Engineering Alternatives

**Node2Vec Embeddings:**
```python
from torch_geometric.nn import Node2Vec

node2vec = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                   context_size=10, walks_per_node=10)
```

**Structural Features:**
```python
# Degree centrality, betweenness, clustering coefficient
degrees = torch.tensor([nx_graph.degree(n) for n in nx_graph.nodes()])
clustering = torch.tensor([nx.clustering(nx_graph, n) for n in nx_graph.nodes()])
```

### 9.3 Evaluation Improvements

**Cross-Validation:**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(edges):
    # Train model on train_idx edges, validate on val_idx
```

**Additional Metrics:**
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Precision-Recall AUC (better for imbalanced datasets)
ap_score = average_precision_score(y_true, y_scores)
```

---

## 10. Real-World Applications

### Social Network Analysis
- **Community Detection**: Find groups of closely connected users
- **Influence Prediction**: Identify influential users in networks
- **Link Prediction**: Suggest new friendships or connections

### Biological Networks
- **Protein-Protein Interactions**: Predict functional relationships
- **Drug Discovery**: Identify potential drug-target interactions
- **Disease Networks**: Understand disease progression pathways

### Knowledge Graphs
- **Entity Linking**: Connect entities across different knowledge bases
- **Relation Extraction**: Discover new relationships between entities
- **Question Answering**: Navigate knowledge graphs to answer queries

### Recommendation Systems
- **Collaborative Filtering**: User-item interaction graphs
- **Content-Based**: Item similarity graphs
- **Hybrid Approaches**: Combine multiple graph types

---

## Summary

This notebook demonstrates the power of Graph Neural Networks for:

1. **Learning meaningful node representations** from graph structure
2. **Link prediction** in social networks using encoder-decoder architecture
3. **Community detection** through clustering of learned embeddings
4. **Handling real-world graph data** with proper preprocessing and evaluation

**Key Takeaways:**
- GNNs can learn from graph structure even without rich node features
- Link prediction is a fundamental task with many practical applications
- Learned embeddings capture both local neighborhoods and global graph properties
- Proper evaluation requires careful train/validation/test splits for graphs

The combination of graph structure learning and downstream tasks makes GNNs powerful tools for analyzing complex relational data across many domains.