# Paper Category Classification with GNN - Coding Guide

## Overview
This notebook demonstrates **node classification** using Graph Neural Networks (GNNs) on the **Cora citation network dataset**. It compares GNN performance against traditional Multi-Layer Perceptron (MLP) approaches for classifying research papers into different categories based on their content and citation relationships.

## Key Learning Objectives
- Understand node classification in citation networks
- Implement Graph Convolutional Networks (GCNs) for paper classification
- Compare GNN vs MLP performance on graph-structured data
- Analyze graph properties and node degree distributions
- Visualize confusion matrices and training curves

---

## 1. Library Installation and Imports

```python
# Install PyTorch Geometric for graph neural networks
!pip install torch_geometric

# Core PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# PyTorch Geometric imports
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as gnn

# Graph analysis and visualization
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing utilities
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
```

### Why These Libraries?
- **torch_geometric**: Specialized library for graph neural networks
- **Planetoid**: Contains citation network datasets (Cora, CiteSeer, PubMed)
- **networkx**: Graph analysis and manipulation
- **sklearn**: Traditional ML algorithms and preprocessing tools
- **seaborn/matplotlib**: Advanced visualization for confusion matrices

---

## 2. Dataset Loading and Exploration

### 2.1 Cora Citation Network Dataset

```python
# Load the Cora dataset
dataset = Planetoid(root='.', name='Cora')
```

**Dataset Details:**
- **Cora**: Citation network of machine learning papers
- **Nodes**: 2,708 research papers
- **Edges**: 5,278 citation relationships (undirected)
- **Node Features**: 1,433-dimensional bag-of-words vectors
- **Classes**: 7 research areas (Neural Networks, Rule Learning, etc.)
- **Task**: Classify papers into research categories

### 2.2 Graph Analysis

```python
# Extract graph components
data = dataset[0]
edge_index = data.edge_index  # Edge connectivity
x = data.x                    # Node features (bag-of-words)
y = data.y                    # Node labels (paper categories)

# Create NetworkX graph for analysis
G = nx.Graph()
G.add_edges_from(edge_index.t().tolist())

# Basic statistics
num_nodes = G.number_of_nodes()           # 2,708 papers
num_edges = G.number_of_edges()           # 5,278 citations
num_classes = y.max().item() + 1          # 7 categories
num_node_features = data.num_node_features # 1,433 features
num_edge_features = data.num_edge_features # 0 (no edge features)

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f'Is the Graph undirected: {data.is_undirected()}')
print(f"Number of Node Features: {num_node_features}")
print(f"Number of classes: {num_classes}")
```

**Key Insights:**
- **Sparse graph**: Average degree ≈ 3.9 (each paper cites ~4 others)
- **Undirected**: Citations treated as bidirectional relationships
- **Rich features**: 1,433-dimensional bag-of-words representation
- **Multi-class**: 7 different research areas to classify

### 2.3 Citation Distribution Analysis

```python
# Analyze node degrees (citation counts)
node_degrees = [val for (node, val) in G.degree()]
display(pd.DataFrame(pd.Series(node_degrees).describe()).transpose())

# Visualize degree distribution
plt.figure(figsize=(10, 6))
sns.histplot(node_degrees, bins=100)
plt.xlabel("node degree")
plt.title("Histogram of Node Degrees (number of citations for papers)")
plt.show()
```

**Distribution Characteristics:**
- **Mean degree**: ~3.9 citations per paper
- **Highly skewed**: Most papers have few citations, few papers have many
- **Long tail**: Some highly cited papers (up to 168 citations)
- **Research impact**: Degree reflects paper influence in the field

---

## 3. Graph Neural Network Implementation

### 3.1 GCN Model Architecture

```python
class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GNNModel, self).__init__()
        
        # Two Graph Convolutional layers
        self.conv1 = gnn.GCNConv(num_features, hidden_size)
        self.conv2 = gnn.GCNConv(hidden_size, hidden_size)
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer + ReLU activation
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # Second GCN layer + ReLU activation
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Final classification with log-softmax
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)
```

### Architecture Breakdown:

**GCN Layer Operations:**
```
h_i^(l+1) = σ(W^(l) * AGG({h_j^(l) : j ∈ N(i) ∪ {i}}))
```

**Layer-by-layer Flow:**
1. **Input**: 1,433-dimensional bag-of-words features
2. **GCN Layer 1**: 1,433 → 16 dimensions with neighborhood aggregation
3. **ReLU Activation**: Non-linear transformation
4. **GCN Layer 2**: 16 → 16 dimensions with further aggregation
5. **ReLU Activation**: Second non-linearity
6. **Linear Classification**: 16 → 7 class probabilities
7. **Log-Softmax**: Convert to log-probabilities for NLL loss

**Why This Architecture?**
- **Two GCN layers**: Capture 2-hop neighborhood information
- **Hidden size 16**: Compact representation while preserving information
- **ReLU activations**: Enable non-linear feature learning
- **Log-softmax output**: Compatible with Negative Log-Likelihood loss

---

## 4. Training Process

### 4.1 Model Setup and Configuration

```python
# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
num_features = dataset.num_node_features  # 1,433
hidden_size = 16                          # Compressed representation
num_classes = dataset.num_classes         # 7 categories

# Initialize model, loss, and optimizer
model = GNNModel(num_features, hidden_size, num_classes).to(device)
criterion = nn.NLLLoss()                  # Negative Log-Likelihood
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Move data to device
data = dataset[0].to(device)
```

**Hyperparameter Choices:**
- **Hidden size (16)**: Balance between expressiveness and overfitting
- **Learning rate (0.01)**: Standard rate for Adam optimizer
- **NLLLoss**: Appropriate for multi-class classification with log-softmax

### 4.2 Training and Evaluation Loop

```python
def train_and_evaluate():
    # Training phase
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(data)
    
    # Calculate training loss and accuracy
    train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    train_pred = output.argmax(dim=1)
    train_acc = (train_pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    
    # Backward pass
    train_loss.backward()
    optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
        val_pred = output.argmax(dim=1)
        val_acc = (val_pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    
    return train_loss.item(), train_acc, val_loss.item(), val_acc

# Training loop
num_epochs = 50
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc, val_loss, val_acc = train_and_evaluate()
    
    print(f'Epoch: {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, '
          f'Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, '
          f'Validation Accuracy: {val_acc:.4f}')
    
    # Store metrics for plotting
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
```

**Training Strategy:**
- **Transductive learning**: All nodes visible during training (standard for citation networks)
- **Mask-based splits**: Use predefined train/validation/test masks
- **Early monitoring**: Track both loss and accuracy for overfitting detection
- **Full-batch training**: Process entire graph in each iteration

### 4.3 Final Test Evaluation

```python
# Test evaluation
model.eval()
with torch.no_grad():
    output = model(data)
    test_pred = output.argmax(dim=1)
    test_acc = (test_pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
print(f'Test Accuracy: {test_acc}')
```

---

## 5. Performance Visualization

### 5.1 Training Curves

```python
# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

**What to Look For:**
- **Convergence**: Both loss and accuracy should stabilize
- **Overfitting**: Large gap between train and validation metrics
- **Underfitting**: Poor performance on both train and validation
- **Optimal stopping**: Best validation performance point

### 5.2 Confusion Matrix Analysis

```python
# Generate predictions for confusion matrix
model.eval()
with torch.no_grad():
    output = model(data)
    _, predicted = torch.max(output, 1)

# Extract test set predictions and true labels
predicted = predicted[data.test_mask].cpu().numpy()
true_labels = data.y[data.test_mask].cpu().numpy()

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted)
class_labels = np.unique(true_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Add labels and text annotations
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

# Add text labels inside each cell
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()
```

**Confusion Matrix Interpretation:**
- **Diagonal elements**: Correct classifications
- **Off-diagonal elements**: Misclassifications between categories
- **Class imbalances**: Some categories may be harder to classify
- **Similar categories**: Confusion between related research areas

---

## 6. Baseline Comparison: Multi-Layer Perceptron

### 6.1 MLP Implementation

```python
# Prepare data for MLP (ignore graph structure)
node_features = data.x.numpy()
node_labels = data.y.numpy()

# Encode labels
label_encoder = LabelEncoder()
node_labels = label_encoder.fit_transform(node_labels)

# Convert to tensors
node_features = torch.tensor(node_features)
node_labels = torch.tensor(node_labels)

# Extract train/validation/test splits
X_train = node_features[data.train_mask]
X_val = node_features[data.val_mask]
X_test = node_features[data.test_mask]
y_train = node_labels[data.train_mask]
y_val = node_labels[data.val_mask]
y_test = node_labels[data.test_mask]

# MLP Architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return torch.log_softmax(out, dim=1)

# Initialize MLP
input_dim = X_train.shape[1]    # 1,433 features
hidden_dim = 16                 # Same as GNN for fair comparison
output_dim = np.unique(node_labels).shape[0]  # 7 classes

mlp = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.NLLLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)  # Lower LR for MLP
```

**MLP vs GNN Differences:**
- **No graph structure**: MLP only uses node features, ignores citations
- **Independent predictions**: Each paper classified in isolation
- **Same architecture size**: Fair comparison with 16 hidden units
- **Lower learning rate**: MLPs often need more careful tuning

### 6.2 MLP Training Loop

```python
num_epochs = 100
batch_size = 32
num_batches = len(X_train) // batch_size

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    train_acc_sum = 0
    
    # Mini-batch training
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size
        
        optimizer.zero_grad()
        
        # Forward pass on batch
        batch_inputs = X_train[start_idx:end_idx]
        batch_labels = y_train[start_idx:end_idx]
        
        outputs = mlp(batch_inputs)
        train_loss = criterion(outputs, batch_labels)
        
        # Calculate batch accuracy
        train_labels_predicted = torch.argmax(outputs, dim=1).numpy()
        train_acc_sum += (train_labels_predicted == batch_labels.numpy()).mean()
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
    
    # Average training accuracy
    train_acc = train_acc_sum / num_batches
    
    # Validation evaluation
    with torch.no_grad():
        val_outputs = mlp(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_labels_predicted = torch.argmax(val_outputs, dim=1).numpy()
        val_labels_true = y_val.numpy()
        val_acc = (val_labels_predicted == val_labels_true).mean()
    
    print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {train_loss.item():.4f}, "
          f"Training Accuracy: {train_acc:.2f}, Validation Loss: {val_loss.item():.4f}, "
          f"Validation Accuracy: {val_acc}")
    
    # Store metrics
    train_accuracies.append(train_acc)
    train_losses.append(train_loss.item())
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())
```

**MLP Training Differences:**
- **Mini-batch processing**: More memory efficient than full-batch
- **More epochs**: MLPs often need longer training
- **Batch accuracy calculation**: Average over mini-batches
- **No graph operations**: Standard feedforward processing

### 6.3 MLP Test Evaluation

```python
# Final test evaluation
with torch.no_grad():
    test_outputs = mlp(X_test)
    predicted_labels = torch.argmax(test_outputs, dim=1).numpy()
    true_labels = y_test.numpy()
    
    accuracy = (predicted_labels == true_labels).mean()
    print(f"Test Accuracy: {accuracy:.4f}")

# Confusion matrix for MLP
cm = confusion_matrix(true_labels, predicted_labels)
unique_labels = np.unique(np.concatenate((true_labels, predicted_labels)))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
           xticklabels=unique_labels, yticklabels=unique_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
```

---

## 7. Performance Comparison and Analysis

### 7.1 Expected Results

**Typical Performance:**
- **GNN Test Accuracy**: ~80-85%
- **MLP Test Accuracy**: ~70-75%
- **Performance Gap**: GNNs typically outperform MLPs by 5-15%

### 7.2 Why GNNs Outperform MLPs

**Graph Structure Benefits:**
1. **Homophily**: Connected papers tend to be in similar categories
2. **Citation context**: References provide additional classification signals
3. **Neighborhood aggregation**: Combines local and global information
4. **Transductive learning**: Leverages unlabeled nodes during training

**Feature Enhancement:**
```python
# GNN effectively creates enhanced features:
# Original: x_i (1433-dim bag-of-words)
# Enhanced: AGG(x_i, {x_j : j ∈ neighbors(i)})
```

### 7.3 Ablation Studies

**Removing Graph Structure:**
- MLP uses only node features → Lower performance
- Demonstrates value of citation relationships

**Different GNN Architectures:**
```python
# Could experiment with:
# - GraphSAGE: Better for large graphs
# - GAT: Attention-based aggregation
# - GIN: More expressive than GCN
```

---

## 8. Advanced Analysis

### 8.1 Node Degree vs Classification Accuracy

```python
# Analyze how node degree affects classification
degrees = dict(G.degree())
node_degrees = [degrees[i] for i in range(len(data.y))]

# Get per-node accuracy
model.eval()
with torch.no_grad():
    output = model(data)
    predictions = output.argmax(dim=1)
    correct = (predictions == data.y).cpu().numpy()

# Analyze accuracy by degree
degree_accuracy = {}
for degree in set(node_degrees):
    mask = np.array(node_degrees) == degree
    if mask.sum() > 0:
        degree_accuracy[degree] = correct[mask].mean()

# Plot degree vs accuracy
degrees_sorted = sorted(degree_accuracy.keys())
accuracies = [degree_accuracy[d] for d in degrees_sorted]

plt.figure(figsize=(10, 6))
plt.scatter(degrees_sorted, accuracies, alpha=0.7)
plt.xlabel('Node Degree')
plt.ylabel('Classification Accuracy')
plt.title('Node Degree vs Classification Accuracy')
plt.show()
```

### 8.2 Class-wise Performance Analysis

```python
# Analyze performance by research category
from sklearn.metrics import classification_report

model.eval()
with torch.no_grad():
    output = model(data)
    predictions = output.argmax(dim=1)

# Get test set predictions
test_predictions = predictions[data.test_mask].cpu().numpy()
test_true = data.y[data.test_mask].cpu().numpy()

# Generate detailed classification report
print(classification_report(test_true, test_predictions))
```

---

## 9. Real-World Applications

### Academic Research
- **Paper recommendation**: Suggest relevant papers based on citations
- **Research trend analysis**: Identify emerging research areas
- **Collaboration prediction**: Predict future co-authorships

### Industry Applications
- **Patent classification**: Categorize patents by technology area
- **Document clustering**: Group similar documents in large corpora
- **Knowledge graph completion**: Predict missing relationships

### Social Networks
- **Community detection**: Find groups of related users
- **Influence analysis**: Identify key opinion leaders
- **Content recommendation**: Suggest relevant content based on network

---

## 10. Key Takeaways

### Technical Insights
1. **Graph structure matters**: Citation relationships provide valuable classification signals
2. **Neighborhood aggregation**: GCNs effectively combine local and global information
3. **Transductive learning**: Using all nodes during training improves performance
4. **Feature enhancement**: Graph convolutions create richer representations

### Practical Considerations
1. **Scalability**: Full-batch training limits scalability to very large graphs
2. **Overfitting**: Small datasets may require careful regularization
3. **Hyperparameter tuning**: Learning rates and architectures need domain-specific tuning
4. **Evaluation**: Proper train/validation/test splits crucial for citation networks

### Future Directions
1. **Attention mechanisms**: GATs for better interpretability
2. **Heterogeneous graphs**: Handle different node/edge types
3. **Dynamic graphs**: Incorporate temporal citation patterns
4. **Large-scale methods**: Sampling techniques for massive graphs

---

## Summary

This notebook demonstrates the power of Graph Neural Networks for node classification tasks. By leveraging both node features (paper content) and graph structure (citation relationships), GNNs significantly outperform traditional MLPs that only use node features. The Cora dataset provides an excellent testbed for understanding how graph structure enhances machine learning performance in real-world scenarios.

The comparison between GNN and MLP approaches clearly shows the value of incorporating relational information, making this a fundamental example for understanding modern graph-based machine learning techniques.