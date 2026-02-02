# KV_Caching - Coding Guide

## 📋 Overview
This notebook demonstrates **Key-Value (KV) Caching**, a crucial optimization technique for efficient inference in transformer models. KV caching dramatically reduces computational overhead during autoregressive text generation.

---

## 🎯 Learning Objectives
- Understand why KV caching is essential for transformer inference
- Learn how to implement KV cache in attention mechanisms
- Master the concept of autoregressive generation
- Implement efficient sampling strategies with caching

---

## 📚 Key Libraries and Imports

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional
```

**Why these imports?**
- `torch`: Core PyTorch framework for tensor operations
- `transformers`: Hugging Face library for pre-trained models
- `dataclasses`: For creating structured configuration classes
- `typing.Optional`: Type hints for optional parameters

---

## 🔬 Core Concepts Explained

### 1. The Problem: Redundant Computation

**Without KV Caching:**
```
Step 1: "I have" → compute K,V for ["I", "have"]
Step 2: "I have a" → compute K,V for ["I", "have", "a"] (redundant!)
Step 3: "I have a dream" → compute K,V for ["I", "have", "a", "dream"] (very redundant!)
```

**With KV Caching:**
```
Step 1: "I have" → compute K,V for ["I", "have"], cache them
Step 2: "I have a" → reuse cached K,V, only compute K,V for ["a"]
Step 3: "I have a dream" → reuse cached K,V, only compute K,V for ["dream"]
```

### 2. Why KV Caching Works

**Attention Mechanism Recap:**
```
Attention(Q, K, V) = softmax(QK^T / √d)V
```

**Key Insight:**
- During generation, we only add one new token at a time
- Previous tokens' K and V matrices never change
- We can cache them and reuse for all future steps

---

## 🏗️ Implementation Breakdown

### 1. Model Configuration

```python
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None
```

**Key Parameters for KV Caching:**
- `max_batch_size`: Maximum number of sequences processed simultaneously
- `max_seq_len`: Maximum sequence length (determines cache size)
- These parameters pre-allocate memory for the cache

### 2. Sampler Base Class

```python
class Sampler:
    def __init__(self, model_name: str = 'gpt2-medium') -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu").to(self.device)
```

**Design Patterns:**
- **Base Class**: Provides common functionality for all sampling strategies
- **Device Management**: Automatically detects and uses GPU if available
- **Model Loading**: Uses Hugging Face transformers for easy model access

### 3. Core Sampling Methods

```python
def encode(self, text):
    return self.tokenizer.encode(text, return_tensors='pt').to(self.device)

def decode(self, ids):
    return self.tokenizer.decode(ids)

def get_next_token_prob(self, input_ids: torch.Tensor):
    with torch.no_grad():
        logits = self.model(input_ids=input_ids).logits
    logits = logits[0, -1, :]  # Get logits for the last token
    return logits
```

**Key Methods Explained:**
- `encode()`: Converts text to token IDs
- `decode()`: Converts token IDs back to text
- `get_next_token_prob()`: Gets probability distribution for next token

### 4. Greedy Sampling Implementation

```python
class GreedySampler(Sampler):
    def __call__(self, prompt, max_new_tokens=10):
        predictions = []
        result = prompt  # string
        
        for i in range(max_new_tokens):
            print(f"step {i} input: {result}")
            input_ids = self.encode(result)
            next_token_probs = self.get_next_token_prob(input_ids=input_ids)
            
            # Choose the token with the highest probability
            id = torch.argmax(next_token_probs, dim=-1).item()
            result += self.decode(id)
            
            predictions.append(next_token_probs[id].item())
        
        return result
```

**Greedy Strategy:**
- Always picks the most likely next token
- Deterministic output (same input → same output)
- Fast but can lead to repetitive text

### 5. KV Cache Implementation

```python
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # Initialize KV cache tensors
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
```

**Cache Initialization:**
- Pre-allocate memory for maximum possible size
- Dimensions: [batch_size, sequence_length, num_heads, head_dimension]
- Zero initialization (will be filled during generation)

### 6. Forward Pass with KV Caching

```python
def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
    batch_size, seq_len, _ = x.shape  # (B, 1, Dim)
    
    # Compute Q, K, V for current token
    xq = self.wq(x)  # (B, 1, H_Q * Head_Dim)
    xk = self.wk(x)  # (B, 1, H_KV * Head_Dim)
    xv = self.wv(x)  # (B, 1, H_KV * Head_Dim)
    
    # Update cache with new K, V
    self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
    self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
    
    # Retrieve all cached K, V (including new ones)
    keys = self.cache_k[:batch_size, : start_pos + seq_len]
    values = self.cache_v[:batch_size, : start_pos + seq_len]
```

**Key Operations:**
1. **Compute Current**: Only compute K,V for the new token
2. **Update Cache**: Store new K,V at the correct position
3. **Retrieve All**: Get all K,V from start to current position

---

## 🚀 Performance Comparison

### Benchmark Results

```python
# Performance comparison code
for use_cache in (True, False):
    times = []
    for _ in range(10):
        start = time.time()
        model.generate(**tokenizer("What is KV caching?", return_tensors="pt").to(device), 
                      use_cache=use_cache, max_new_tokens=1000)
        times.append(time.time() - start)
    print(f"{'with' if use_cache else 'without'} KV caching: {round(np.mean(times), 3)} +- {round(np.std(times), 3)} seconds")
```

**Typical Results:**
- **With KV caching**: ~70 seconds
- **Without KV caching**: ~2544 seconds
- **Speedup**: ~36x faster!

---

## 🔧 Advanced Concepts

### 1. Memory vs. Computation Trade-off

**Memory Usage:**
- Cache size: `batch_size × max_seq_len × num_heads × head_dim × 2 (K and V)`
- For GPT-2: ~32 × 2048 × 12 × 64 × 2 = ~100MB per batch

**Computation Savings:**
- Without cache: O(n²) operations for sequence length n
- With cache: O(n) operations for sequence length n

### 2. Cache Management Strategies

**Static Allocation:**
```python
# Pre-allocate maximum size
cache_k = torch.zeros((max_batch, max_seq, n_heads, head_dim))
```

**Dynamic Allocation:**
```python
# Grow cache as needed
if start_pos + seq_len > cache_k.size(1):
    cache_k = torch.cat([cache_k, torch.zeros(...)], dim=1)
```

### 3. Multi-Head Attention with KV Cache

**Grouped Query Attention:**
- Some models use fewer K,V heads than Q heads
- Reduces memory usage while maintaining performance
- Example: 32 Q heads, 8 KV heads

---

## 🎯 Key Takeaways for Beginners

### 1. **Why KV Caching is Essential**
- Autoregressive generation is inherently sequential
- Each step recomputes the same K,V matrices
- Caching eliminates this redundancy

### 2. **Memory Considerations**
- Cache size grows with sequence length
- Need to balance memory usage vs. speed
- Consider batch size and maximum sequence length

### 3. **Implementation Details**
- Cache must be updated correctly at each step
- Position tracking is crucial (`start_pos` parameter)
- Attention mask may need adjustment for cached sequences

### 4. **When to Use KV Caching**
- **Always** for inference in production
- During training: usually not needed (parallel processing)
- For interactive applications: essential for responsiveness

---

## 🔍 Common Pitfalls and Solutions

### 1. **Incorrect Position Tracking**
- **Problem**: Wrong `start_pos` leads to overwriting cache
- **Solution**: Carefully track current position in sequence

### 2. **Memory Leaks**
- **Problem**: Cache tensors not properly cleared between sequences
- **Solution**: Reset cache or use proper indexing

### 3. **Batch Size Mismatches**
- **Problem**: Cache allocated for different batch size
- **Solution**: Ensure consistent batch dimensions

---

## 📈 Extensions and Optimizations

### 1. **Sliding Window Cache**
- Keep only recent tokens in cache
- Useful for very long sequences
- Trade-off between memory and context length

### 2. **Quantized KV Cache**
- Store K,V in lower precision (int8, int4)
- Reduces memory usage significantly
- Minimal impact on generation quality

### 3. **Multi-Query Attention**
- Share K,V across all attention heads
- Dramatically reduces cache size
- Used in models like PaLM, LLaMA

This implementation demonstrates the fundamental concepts of KV caching and its dramatic impact on inference efficiency in transformer models.