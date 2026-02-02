# Sampling - Coding Guide

## 📋 Overview
This notebook demonstrates various **sampling strategies** for text generation with language models. It covers the fundamental concepts discussed in the class about how to convert model probability distributions into actual text tokens.

---

## 🎯 Learning Objectives
- Understand different sampling strategies (greedy, top-k, top-p)
- Learn how to implement custom sampling functions
- Master the relationship between logits, probabilities, and token selection
- Visualize sampling decisions using graph representations

---

## 📚 Key Libraries and Imports

```python
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
```

**Why these imports?**
- `transformers`: Hugging Face library for pre-trained models (GPT-2)
- `torch`: PyTorch for tensor operations and neural networks
- `matplotlib.pyplot`: For creating visualizations
- `networkx`: For creating and visualizing decision trees
- `numpy`: Numerical operations and statistics
- `time`: For performance benchmarking

---

## 🔬 Core Concepts Explained

### 1. From Logits to Tokens: The Complete Pipeline

**Step 1: Model Forward Pass**
```python
text = "Once upon a time"
input_ids = tokenizer.encode(text, return_tensors='pt')
outputs = model.generate(input_ids, max_new_tokens=10)
```

**What happens internally:**
1. Text → Token IDs (tokenization)
2. Token IDs → Model → Logits (forward pass)
3. Logits → Probabilities (softmax)
4. Probabilities → Selected Token (sampling)
5. Selected Token → Text (detokenization)

### 2. Understanding Model Output Structure

```python
output = model(input_ids)
print(f"Input shape: {input_ids.shape}")      # [1, 4] - batch_size=1, seq_len=4
print(f"Output shape: {output.logits.shape}") # [1, 4, 50257] - batch_size=1, seq_len=4, vocab_size=50257
```

**Key Insights:**
- Model outputs logits for **every** position in the sequence
- For generation, we only care about the **last** position: `output.logits[0, -1, :]`
- Vocabulary size (50257 for GPT-2) determines the number of possible next tokens

---

## 🏗️ Implementation Breakdown

### 1. Basic Text Generation

```python
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

text = "Once upon a time"
input_ids = tokenizer.encode(text, return_tensors='pt')

outputs = model.generate(input_ids, max_new_tokens=10)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Step-by-Step Breakdown:**
1. **Load Model**: Get pre-trained GPT-2 model and tokenizer
2. **Tokenize**: Convert text to numerical token IDs
3. **Generate**: Use model's built-in generation (uses sampling internally)
4. **Detokenize**: Convert token IDs back to readable text

### 2. Manual Model Inference

```python
output = model(input_ids)
pred = output.logits[0, -1, :]  # Get logits for next token
print(f"Logits shape: {pred.shape}")  # [50257] - one score per vocabulary token
print(f"Most likely token ID: {torch.argmax(pred)}")
```

**Understanding Logits:**
- Raw model outputs before softmax
- Higher values = more likely tokens
- Need to be converted to probabilities for sampling

### 3. Utility Functions for Sampling

```python
def get_log_prob(logits, token_id):
    """Convert logits to log probabilities and get specific token's log prob"""
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities)
    token_log_probability = log_probabilities[token_id].item()
    return token_log_probability

def get_predicted_logits(model, input_ids):
    """Get logits for next token prediction"""
    outputs = model(input_ids)
    predictions = outputs.logits
    logits = predictions[0, -1, :]  # Last position only
    return logits
```

**Why Log Probabilities?**
- Numerical stability (avoids very small numbers)
- Easier to work with for scoring sequences
- Standard practice in NLP

---

## 🎲 Sampling Strategies Implementation

### 1. Greedy Search

```python
def greedy_search(input_ids, node, length=5):
    if length == 0:
        return input_ids
    
    print(input_ids)
    logits = get_predicted_logits(model, input_ids)
    
    ### This is where we choose (sample) the next token ###
    token_id = torch.argmax(logits).unsqueeze(0)
    
    # Compute the score of the predicted token
    token_score = get_log_prob(logits, token_id)
    
    # Add the predicted token to the list of input ids
    new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)
    
    # Recursive call for next token
    input_ids = greedy_search(new_input_ids, current_node, length-1)
    
    return input_ids
```

**Greedy Search Characteristics:**
- **Strategy**: Always pick the highest probability token
- **Pros**: Fast, deterministic, often coherent
- **Cons**: Can be repetitive, gets stuck in loops
- **Use Case**: When you want the most likely continuation

### 2. Visualization Setup

```python
# Create a balanced tree for visualization
length = 5
graph = nx.balanced_tree(1, length, create_using=nx.DiGraph())

# Add attributes to each node
for node in graph.nodes:
    graph.nodes[node]['tokenscore'] = 100
    graph.nodes[node]['token'] = text
```

**Visualization Purpose:**
- Shows decision tree of token choices
- Each node represents a token selection
- Edge weights show probability scores
- Helps understand sampling behavior

### 3. Advanced Plotting Function

```python
def plot_graph(graph, length, beams, score):
    fig, ax = plt.subplots(figsize=(3+1.2*beams**length, max(5, 2+length)), dpi=300, facecolor='white')
    
    # Create positions for each node
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    
    # Normalize colors based on token scores
    if score == 'token':
        scores = [data['tokenscore'] for _, data in graph.nodes(data=True) if data['token'] is not None]
    elif score == 'sequence':
        scores = [data['sequencescore'] for _, data in graph.nodes(data=True) if data['token'] is not None]
    
    vmin = min(scores)
    vmax = max(scores)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "y", "g"], N=256)
    
    # Draw nodes with color coding
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_shape='o', alpha=1, linewidths=4,
                          node_color=scores, cmap=cmap)
    
    # Draw edges and labels
    nx.draw_networkx_edges(graph, pos)
    
    if score == 'token':
        labels = {node: data['token'].split('_')[0] + f"\\n{data['tokenscore']:.2f}%" 
                 for node, data in graph.nodes(data=True) if data['token'] is not None}
    
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', pad=0, label='Token probability (%)')
    plt.show()
```

**Visualization Features:**
- **Color Coding**: Green = high probability, Red = low probability
- **Node Labels**: Show token text and probability percentage
- **Tree Structure**: Shows sequence of decisions
- **Colorbar**: Legend for probability interpretation

---

## 🔧 Advanced Concepts

### 1. Temperature Scaling (Implied)

While not explicitly implemented in this notebook, temperature scaling is crucial:

```python
def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits"""
    return logits / temperature

# Usage in sampling
logits = get_predicted_logits(model, input_ids)
scaled_logits = apply_temperature(logits, temperature=0.8)
probabilities = torch.softmax(scaled_logits, dim=-1)
```

**Temperature Effects:**
- **T < 1**: Sharper distribution (more confident)
- **T = 1**: Original distribution
- **T > 1**: Flatter distribution (more random)

### 2. Top-K Sampling (Extension)

```python
def top_k_sampling(logits, k=50):
    """Sample from top-k most likely tokens"""
    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k)
    
    # Create probability distribution over top-k
    top_k_probs = torch.softmax(top_k_values, dim=-1)
    
    # Sample from top-k distribution
    sampled_index = torch.multinomial(top_k_probs, 1)
    
    # Get original token ID
    token_id = top_k_indices[sampled_index]
    
    return token_id
```

### 3. Top-P (Nucleus) Sampling (Extension)

```python
def top_p_sampling(logits, p=0.9):
    """Sample from smallest set of tokens with cumulative probability >= p"""
    # Sort probabilities in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff point
    cutoff = torch.where(cumulative_probs >= p)[0][0] + 1
    
    # Keep only top-p tokens
    top_p_probs = sorted_probs[:cutoff]
    top_p_indices = sorted_indices[:cutoff]
    
    # Renormalize and sample
    top_p_probs = top_p_probs / top_p_probs.sum()
    sampled_index = torch.multinomial(top_p_probs, 1)
    
    return top_p_indices[sampled_index]
```

---

## 🎯 Key Takeaways for Beginners

### 1. **The Sampling Pipeline**
```
Text → Tokens → Model → Logits → Probabilities → Sample → Token → Text
```

### 2. **Why Different Sampling Strategies?**
- **Greedy**: Fast, deterministic, but can be boring
- **Random**: Creative, but can be incoherent
- **Top-K**: Balanced creativity and coherence
- **Top-P**: Adaptive vocabulary size based on confidence

### 3. **Model Output Understanding**
- Models output probability distributions, not words
- The sampling strategy determines the final text
- Same model + different sampling = different outputs

### 4. **Practical Considerations**
- Greedy for factual tasks (translation, summarization)
- Sampling for creative tasks (story generation, dialogue)
- Temperature controls randomness level

---

## 🔍 Common Pitfalls and Solutions

### 1. **Repetition in Greedy Search**
- **Problem**: Model gets stuck in loops
- **Solution**: Use repetition penalty or different sampling strategy

### 2. **Incoherent Random Sampling**
- **Problem**: Pure random sampling produces nonsense
- **Solution**: Use top-k or top-p to constrain choices

### 3. **Memory Issues with Long Sequences**
- **Problem**: Attention computation grows quadratically
- **Solution**: Use KV caching or sliding window attention

---

## 📈 Performance and Quality Trade-offs

### 1. **Speed vs. Quality**
- Greedy: Fastest, decent quality
- Top-K: Moderate speed, good quality
- Top-P: Slower, best quality

### 2. **Determinism vs. Creativity**
- Deterministic: Reproducible, but potentially boring
- Stochastic: Creative, but harder to control

### 3. **Memory vs. Context**
- Longer context: Better coherence, more memory
- Shorter context: Less memory, potential coherence loss

This notebook provides a comprehensive foundation for understanding and implementing various sampling strategies in language model text generation.