# Week 30 - Generative AI Part 2: Comprehensive Study Guide

## 🎯 Learning Objectives
By the end of this session, you should understand:
- How decoder-only models work (GPT architecture)
- Sampling techniques for text generation (Greedy, Top-K, Top-P, Temperature)
- Key-Value (KV) caching for efficient inference
- Knowledge distillation in language models
- Comprehensive LLM optimization techniques
- Training vs inference differences (teacher forcing vs autoregressive)
- The mathematics behind transformer inference
- Model compression and deployment strategies

---

## 📚 Table of Contents
1. [Core Concepts Explained Simply](#core-concepts-explained-simply)
2. [Technical Deep Dive](#technical-deep-dive)
3. [LLM Optimization Techniques](#llm-optimization-techniques)
4. [Model Compression Strategies](#model-compression-strategies)
5. [Key Points from Class](#key-points-from-class)
6. [Interview Questions & Answers](#interview-questions--answers)
7. [Summary](#summary)

---

## 🌟 Core Concepts Explained Simply

### Think of Language Models Like a Smart Autocomplete

Imagine you're texting and your phone suggests the next word. That's essentially what a Large Language Model (LLM) does, but much more sophisticated!

**The Magic Recipe:**
1. **Input**: You give it some text (like "The weather today is...")
2. **Processing**: The model thinks about all possible next words
3. **Output**: It gives you probabilities for each word in its vocabulary
4. **Selection**: A sampling method picks the actual word to show you

### 🏗️ Architecture Differences Made Simple

**Encoder-Decoder vs Decoder-Only:**

Think of it like two different ways to build a translator:

**Encoder-Decoder (Old Way):**
- Like having a person who first reads and understands the entire sentence (encoder)
- Then another person who writes the translation (decoder)
- The decoder can ask the encoder questions about the original text

**Decoder-Only (Modern Way - GPT):**
- Like having one super-smart person who reads from left to right
- They can only look at what they've read so far (no peeking ahead!)
- Simpler but needs to be really good at remembering context

### 🎲 Sampling: How Models Choose Words

Imagine the model gives you a histogram showing how likely each word is:

```
Word Probabilities:
"sunny" ████████████ 60%
"rainy" ████████ 30%  
"cloudy" ███ 10%
```

**Different Ways to Pick:**

1. **Greedy Sampling**: Always pick the highest bar (always "sunny")
2. **Top-K Sampling**: Only consider the top K words, then randomly pick
3. **Top-P Sampling**: Keep adding words until you reach P% probability, then pick

---

## 🔬 Technical Deep Dive

### Decoder-Only Architecture Details

**Key Differences from Encoder-Decoder:**

```
Encoder-Decoder Layer:
┌─────────────────┐
│ Cross-Attention │  ← Gets info from encoder
├─────────────────┤
│ Self-Attention  │  ← Masked (causal)
├─────────────────┤
│ Feed-Forward    │
└─────────────────┘

Decoder-Only Layer:
┌─────────────────┐
│ Self-Attention  │  ← Only this (masked/causal)
├─────────────────┤
│ Feed-Forward    │
└─────────────────┘
```

**Why Decoder-Only is Better:**
- Simpler architecture (less computation per layer)
- Can add more layers with saved computation
- No lossy compression from encoder
- Better scaling properties

### Mathematical Foundation

**The Core Probability:**
```
P(y_t | x, y_1, y_2, ..., y_{t-1})
```
Where:
- `y_t` = token at time step t
- `x` = input prompt
- `y_1, ..., y_{t-1}` = previously generated tokens

**From Hidden State to Vocabulary:**
```
hidden_state: [1 × d_model] 
    ↓ (Linear transformation)
logits: [1 × vocab_size]
    ↓ (Softmax)
probabilities: [1 × vocab_size] (sums to 1)
```

### Sampling Algorithms Deep Dive

**Top-K Sampling Algorithm:**
```python
# Pseudocode
def top_k_sampling(logits, k):
    1. Sort tokens by probability (high to low)
    2. Keep only top K tokens
    3. Renormalize probabilities (softmax over K tokens)
    4. Sample from this distribution
```

**Top-P (Nucleus) Sampling:**
```python
def top_p_sampling(logits, p):
    1. Sort tokens by probability (high to low)
    2. Keep adding tokens until cumulative probability ≥ p
    3. Renormalize probabilities
    4. Sample from this distribution
```

### Key-Value Caching

**The Problem:**
- Each token generation requires full forward pass
- Previous computations are repeated unnecessarily

**The Solution:**
```
Time step t:   [prompt] [token_1] [token_2] ... [token_t-1] → [token_t]
Time step t+1: [prompt] [token_1] [token_2] ... [token_t-1] [token_t] → [token_t+1]
```

**KV Cache stores:**
- Key and Value matrices from previous time steps
- Only compute K,V for new token
- Massive speedup for inference

### Training vs Inference Differences

**Training (Teacher Forcing):**
- Always condition on ground truth tokens
- Can see the correct answer for previous positions
- Parallel processing possible

**Inference (Autoregressive):**
- Must condition on model's own predictions
- Sequential generation (one token at a time)
- No parallel processing for generation

---

## � LLM Optnimization Techniques

### Why LLM Inference is Expensive

**Computational Bottlenecks:**
1. **Many Decoder Layers**: Each layer contains heavy computation
2. **Multi-Head Attention**: Complex attention calculations in each layer
3. **Feed-Forward Networks**: Large matrix multiplications (often 4x expansion)
4. **Final Projection**: Huge matrix multiplication (d_model × vocab_size)
5. **Autoregressive Nature**: Must generate tokens one by one

**The Challenge:**
- Each token generation requires full forward pass through all layers
- Large vocabulary size (128K tokens) makes final projection expensive
- Sequential generation prevents parallelization

### 1. Vocabulary and Model Dimension Optimizations

**Reduce Vocabulary Size (N):**
```
Benefits: Smaller final projection matrix (d_model × N)
Trade-offs: 
- Reduced expressiveness
- More unknown tokens (UNK)
- Difficulty with rare/complex words
- Example: "Schadenfreude" might become multiple UNK tokens
```

**Reduce Model Dimension (d_model):**
```
Benefits: Smaller matrices throughout the model
Trade-offs:
- Lossy compression of information
- Reduced model capacity
- May hurt performance on complex tasks
```

### 2. Architecture-Level Optimizations

**Reduce Number of Layers (M):**
```
Benefits: 
- Fewer parameters
- Faster inference
- Less memory usage
Trade-offs:
- Reduced model capacity
- May not handle complex reasoning tasks
```

**Multi-Head Attention Optimizations:**

**a) Reduce Number of Heads (H):**
- Fewer parallel attention computations
- Less model complexity
- Trade-off: Reduced ability to capture different aspects

**b) Multi-Query Attention (MQA):**
```python
# Standard Multi-Head Attention
for each head h:
    W_Q[h], W_K[h], W_V[h]  # Separate matrices per head

# Multi-Query Attention  
for each head h:
    W_Q[h]  # Unique query matrix per head
    W_K, W_V  # SHARED key/value matrices across all heads
```

**c) Grouped Query Attention (GQA):**
```python
# Compromise between MHA and MQA
# Group heads and share K,V within groups
Group 1: heads 1-4 share W_K[1], W_V[1]
Group 2: heads 5-8 share W_K[2], W_V[2]
# etc.
```

### 3. Attention Mechanism Optimizations

**Window Attention:**
```python
# Instead of attending to all N tokens
attention_matrix = Q @ K.T  # Shape: [N, N]

# Only attend to local window of size K
attention_matrix = Q @ K_window.T  # Shape: [N, K] where K << N
```

**Benefits:**
- Reduces computation from O(N²) to O(N×K)
- Useful for tasks where local context is most important
- Used in models like Swin Transformer

**Latent Space Attention:**
```python
# Project to smaller dimension M before attention
Q_latent = project_to_latent(Q)  # [N, d] -> [M, d] where M << N
K_latent = project_to_latent(K)
V_latent = project_to_latent(V)

# Compute attention in latent space: O(M²) instead of O(N²)
attention = softmax(Q_latent @ K_latent.T) @ V_latent
```

### 4. Feed-Forward Network Optimizations

**Standard FFN Structure:**
```python
# Typical 4x expansion
x -> Linear(d_model, 4*d_model) -> ReLU -> Linear(4*d_model, d_model)
```

**Optimization: Reduce Expansion Factor:**
```python
# Reduce from 4x to 2x or 3x
x -> Linear(d_model, 2*d_model) -> ReLU -> Linear(2*d_model, d_model)
```

**Trade-offs:**
- Smaller matrices = faster computation
- Less model capacity for complex transformations
- Must be applied consistently across all layers

### 5. Distributed Training and Model Parallelism

**Why Distributed Training is Needed:**
- Large models (billions of parameters) don't fit on single GPU
- Training time becomes prohibitively long
- Memory requirements exceed single device capacity

**Types of Parallelism:**

**1. Data Parallelism:**
```python
# Same model on multiple GPUs, different data batches
GPU_1: model_copy_1.forward(batch_1)
GPU_2: model_copy_2.forward(batch_2)
GPU_3: model_copy_3.forward(batch_3)

# Gradients are averaged across all GPUs
final_gradient = (grad_1 + grad_2 + grad_3) / 3
```

**2. Model Parallelism:**
```python
# Different parts of model on different GPUs
GPU_1: layers_1_to_4.forward(x)
GPU_2: layers_5_to_8.forward(intermediate_output)
GPU_3: layers_9_to_12.forward(intermediate_output_2)
```

**3. Pipeline Parallelism:**
```python
# Pipeline different stages of computation
Time_1: GPU_1 processes batch_1
Time_2: GPU_1 processes batch_2, GPU_2 processes batch_1_output
Time_3: GPU_1 processes batch_3, GPU_2 processes batch_2_output, GPU_3 processes batch_1_final
```

**Challenges:**
- **Communication Overhead**: GPUs must synchronize frequently
- **Load Balancing**: Ensuring all GPUs are equally utilized
- **Memory Management**: Coordinating memory across devices
- **Fault Tolerance**: Handling GPU failures gracefully

**Network Bottlenecks:**
- High-speed interconnects (NVLink, InfiniBand) required
- Gradient synchronization can become bottleneck
- Bandwidth requirements scale with model size

---

## 🗜️ Model Compression Strategies

### 1. Knowledge Distillation

**Concept:**
- Large "teacher" model guides training of smaller "student" model
- Student learns from teacher's soft predictions, not just hard labels

**Temperature Scaling:**
```python
# Teacher predictions with temperature
teacher_probs = softmax(teacher_logits / temperature)

# Student learns to match these soft targets
student_logits = student_model(x)
loss = KL_divergence(softmax(student_logits/T), teacher_probs)
```

**Advanced Distillation (TinyBERT approach):**
- Multiple loss functions at different layers
- Embedding layer alignment
- Intermediate layer matching
- Final output matching

**Benefits:**
- Student model much smaller than teacher
- Often better than training small model from scratch
- Preserves much of teacher's knowledge

**Limitations:**
- Student only good at distillation task
- Requires teacher model for training (expensive)
- May need retraining for new domains

### 2. Model Pruning

**Concept:**
- Start with large, well-trained model
- Systematically remove unnecessary components
- Fine-tune to recover performance

**Types of Pruning:**

**Unstructured Pruning:**
```python
# Zero out individual weights
if abs(weight) < threshold:
    weight = 0
```

**Structured Pruning:**
```python
# Remove entire layers, heads, or channels
# More hardware-friendly but potentially more destructive
```

**Iterative Process:**
1. Prune some weights/layers
2. Evaluate performance drop
3. Fine-tune to recover performance
4. Repeat until desired size/performance trade-off

### 3. Model Quantization

**Concept:**
- Reduce precision of model weights and activations
- Store numbers in fewer bits (FP32 → FP16 → INT8 → INT4)

**What Can Be Quantized:**
1. **Weights**: Model parameters (most important)
2. **Activations**: Intermediate outputs during inference
3. **Gradients**: During training (less common)

**Precision Considerations:**
- **Weights**: Have wider dynamic range, need careful quantization
- **Activations**: More stable, easier to quantize
- **Critical Operations**: Some operations need full precision

**Types of Quantization:**

**Post-Training Quantization (PTQ):**
```python
1. Train model in full precision (FP32)
2. Use calibration dataset to analyze weight/activation ranges
3. Quantize based on observed statistics
4. May need light fine-tuning
```

**Quantization-Aware Training (QAT):**
```python
1. Include quantization in training process
2. Model learns to be robust to quantization errors
3. Better final performance but more complex training
```

**Mixed Precision Training:**
```python
# Keep sensitive operations in FP32
# Use FP16 for most operations
# Automatic loss scaling to prevent underflow
```

**Hardware Benefits:**
- Modern GPUs (like Blackwell) support INT4/INT8 operations
- Faster matrix multiplications
- Larger effective batch sizes
- Better memory utilization

---

## ⚡ Advanced Inference Optimizations

### 1. Speculative Decoding

**Problem:**
- Autoregressive generation is inherently sequential
- Large models are slow for each token

**Solution:**
```python
# Use small, fast model to generate multiple token hypotheses
small_model_predictions = small_model.generate(context, num_tokens=3)

# Use large model to validate/score these hypotheses
for hypothesis in small_model_predictions:
    score = large_model.score(context + hypothesis)
    if score > threshold:
        accept_hypothesis(hypothesis)
        jump_ahead(3_tokens)
    else:
        reject_and_generate_normally()
```

**Benefits:**
- Can skip multiple tokens when small model guesses correctly
- Large model only does scoring (faster than generation)
- Maintains quality of large model

### 2. Linear Attention Alternatives

**Traditional Attention Complexity:**
- O(N²) in sequence length
- Becomes prohibitive for very long sequences

**Linear Attention Approaches:**
- **Mamba**: State-space models with linear complexity
- **Linear Transformers**: Approximate attention with linear operations
- **Sparse Attention**: Only attend to subset of positions

**Trade-offs:**
- Linear complexity vs. full attention expressiveness
- Good for long sequences, may hurt short sequence performance

### 3. Beam Search Decoding

**Problem with Greedy/Sampling:**
- Greedy always picks highest probability (deterministic)
- Sampling can pick suboptimal paths
- Both make local decisions without considering future consequences

**Beam Search Solution:**
```python
# Instead of picking one token, maintain K best sequences
beam_width = 3
beams = [initial_sequence]

for each_time_step:
    candidates = []
    for beam in beams:
        # Get top K tokens for this beam
        top_k_tokens = model.get_top_k(beam, k=beam_width)
        for token in top_k_tokens:
            new_sequence = beam + [token]
            score = calculate_sequence_probability(new_sequence)
            candidates.append((new_sequence, score))
    
    # Keep only top beam_width sequences
    beams = select_top_k(candidates, k=beam_width)
```

**Benefits:**
- Considers multiple possible paths simultaneously
- Can recover from early suboptimal decisions
- Better quality than greedy for many tasks

**Drawbacks:**
- Much slower than greedy (K times more computation)
- Still makes greedy decisions at sequence level
- Memory usage increases with beam width

### 4. Temperature Scaling Deep Dive

**Mathematical Definition:**
```python
# Standard softmax
probabilities = softmax(logits)

# Temperature-scaled softmax
probabilities = softmax(logits / temperature)
```

**Physical Analogy:**
Think of temperature like heating particles in physics:
- **Low Temperature (T < 1)**: Particles move less, distribution becomes "sharper"
- **High Temperature (T > 1)**: Particles move more, distribution becomes "flatter"
- **T = 1**: Standard softmax (no scaling)
- **T → 0**: Approaches greedy decoding (deterministic)
- **T → ∞**: Approaches uniform distribution (completely random)

**Effect on Probability Distribution:**
```python
# Example with logits = [3.0, 2.0, 1.0]

# T = 0.5 (Low temperature - sharper)
# softmax([6.0, 4.0, 2.0]) = [0.84, 0.14, 0.02]

# T = 1.0 (Standard)
# softmax([3.0, 2.0, 1.0]) = [0.67, 0.24, 0.09]

# T = 2.0 (High temperature - flatter)
# softmax([1.5, 1.0, 0.5]) = [0.46, 0.31, 0.23]
```

**Practical Usage:**
- **Creative Writing**: Higher temperature (1.2-2.0) for diversity
- **Code Generation**: Lower temperature (0.1-0.7) for precision
- **Factual QA**: Very low temperature (0.1-0.3) for consistency

---

## 📝 Key Points from Class

### Architecture Insights
1. **Post-Layer Norm vs Pre-Layer Norm**: Modern models use pre-layer norm (apply normalization before attention/MLP)
2. **Residual Connections**: Critical for training deep networks
3. **Attention Mechanism**: Self-attention with causal masking in decoder-only models
4. **Decoder-Only Advantage**: Simpler architecture allows deeper networks with same compute budget
5. **No Lossy Compression**: Unlike encoder-decoder, no information bottleneck

### Critical Mathematical Concepts
- **Final Projection**: Linear transformation from d_model to vocab_size
- **Softmax Normalization**: Ensures probabilities sum to 1
- **Sampling vs Direct Output**: LLMs output probability distributions, not words directly
- **Autoregressive Generation**: Each token depends on all previous tokens

### Probability and Statistics Review
- **CDF (Cumulative Distribution Function)**: P(X ≤ a)
- **Standard Normal Properties**: Symmetric around 0, μ=0, σ=1
- **Z-table Identity**: Φ(-a) = 1 - Φ(a)
- **PDF Integration**: Area under curve equals 1 for any valid PDF

### Training vs Inference Critical Differences
- **Training (Teacher Forcing)**: 
  - Uses ground truth tokens as context
  - Parallel processing possible
  - Can see "future" tokens during training
  - Prevents error accumulation during learning
- **Inference (Autoregressive)**:
  - Must use model's own predictions
  - Sequential generation only
  - No access to future tokens
  - Potential distribution shift from training

### KV Caching Mechanics
- **Problem**: Recomputing same K,V matrices repeatedly
- **Solution**: Cache previous computations, only compute new token
- **Memory Growth**: Linear with sequence length
- **Speedup**: Can achieve 36x improvement in practice
- **Implementation**: Pre-allocate cache tensors, update incrementally

### Training Details
- **Mini-batch Gradient Descent**: Standard approach (not SGD or full batch)
- **Epoch Definition**: One pass through entire dataset
- **Loss Calculation**: Average loss over batch examples

### Inference Challenges
- **Variable Input/Output Lengths**: Makes batching difficult
- **Autoregressive Nature**: Each token depends on previous ones
- **Computational Cost**: Multiple forward passes per response
- **Memory Management**: KV cache grows with sequence length
- **Latency Variability**: Different response lengths = different inference times

### Production Considerations
- **Padding Strategy**: Extend shorter sequences to maximum length
- **Attention Masking**: Prevent attention to padded positions
- **Batching Complexity**: Group similar-length sequences when possible
- **Error Propagation**: Bugs in sampling can cause random word generation

---

## 🎯 Interview Questions & Answers

### Top 5 Most Important Topics for Generative AI Interviews
*As recommended by the instructor:*

1. **Transformers** (in atomic detail)
2. **LoRA** (Low-Rank Adaptation)
3. **Knowledge Distillation**
4. **Loss Functions & Attention Mechanisms**
5. **Quantization**

### Questions Discussed in Class

**Q1: What's the difference between encoder-decoder and decoder-only architectures?**

**Answer:**
- **Encoder-Decoder**: Two attention mechanisms per decoder layer (self-attention + cross-attention). Cross-attention allows decoder to attend to encoder outputs.
- **Decoder-Only**: Single self-attention mechanism per layer. Simpler, allows for deeper networks with same compute budget.
- **Trade-off**: Decoder-only loses some context compression but gains simplicity and scalability.
- **Why Decoder-Only Won**: Avoids lossy compression from encoder, enables deeper networks, better scaling properties.

**Q2: Why do we need sampling in language models?**

**Answer:**
- LLMs output probability distributions over vocabulary, not direct words
- Need mechanism to convert probabilities to actual token selection
- Different sampling strategies affect creativity vs consistency:
  - **Greedy**: Always most likely (deterministic, safe)
  - **Top-K**: Controlled randomness with fixed candidate set
  - **Top-P**: Adaptive candidate set based on confidence
  - **Temperature**: Controls sharpness of probability distribution

**Q3: Explain teacher forcing and why it's used.**

**Answer:**
- **Definition**: During training, condition on ground truth tokens rather than model predictions
- **Why**: Prevents error accumulation during learning
- **Process**: At each time step, use actual correct previous tokens as context
- **Inference Gap**: Model must rely on its own predictions during inference, leading to potential distribution shift
- **Causal Masking**: Still applied during training - can't look at future tokens

**Q4: How does temperature affect text generation?**

**Answer:**
- **Mathematical**: `softmax(logits / temperature)`
- **Low Temperature (T < 1)**: Sharper distribution, more deterministic
- **High Temperature (T > 1)**: Flatter distribution, more random
- **Physical Analogy**: Like heating particles - higher temperature = more movement
- **Effect**: Higher temperature makes rare words more likely to be selected
- **Usage**: Creative tasks use higher T, factual tasks use lower T

### Advanced LLM Optimization Questions

**Q5: How would you optimize inference speed for a large language model?**

**Answer:**
- **KV Caching**: Store key-value pairs from previous time steps (36x speedup!)
- **Vocabulary Reduction**: Smaller vocab = smaller final projection matrix
- **Model Compression**: Reduce layers, heads, or model dimension
- **Attention Optimizations**: Multi-query attention, grouped query attention, window attention
- **Quantization**: FP32 → FP16 → INT8 → INT4
- **Speculative Decoding**: Use small model to propose, large model to verify

**Q6: Explain different types of attention optimizations.**

**Answer:**
- **Multi-Head Attention (MHA)**: Each head has separate Q, K, V matrices
- **Multi-Query Attention (MQA)**: Shared K, V across all heads, separate Q per head
- **Grouped Query Attention (GQA)**: Compromise - group heads and share K, V within groups
- **Window Attention**: Only attend to local neighborhood (O(N×K) vs O(N²))
- **Latent Attention**: Project to smaller space before attention computation

**Q7: What are the trade-offs in model compression techniques?**

**Answer:**
- **Knowledge Distillation**: 
  - Pro: Preserves teacher knowledge, often better than training small model from scratch
  - Con: Requires teacher model, expensive training, task-specific
- **Pruning**: 
  - Pro: Can remove truly unnecessary weights/layers
  - Con: Iterative process, may need extensive fine-tuning
- **Quantization**: 
  - Pro: Hardware acceleration, memory savings, minimal performance loss
  - Con: Requires careful calibration, some operations need full precision

**Q8: How does KV caching work and why is it important?**

**Answer:**
- **Problem**: Autoregressive generation recomputes same K, V matrices repeatedly
- **Solution**: Cache K, V from previous time steps, only compute for new token
- **Implementation**: Pre-allocate cache tensors, update at each step
- **Benefits**: Dramatic speedup (36x in practice), linear memory growth
- **Considerations**: Memory usage grows with sequence length, batch size constraints

### Model Architecture Questions

**Q9: What are the computational bottlenecks in LLM inference?**

**Answer:**
- **Multiple Decoder Layers**: Each layer has heavy computation (attention + FFN)
- **Multi-Head Attention**: O(N²) complexity in sequence length
- **Feed-Forward Networks**: Large matrix multiplications (often 4x expansion)
- **Final Projection**: Huge matrix (d_model × vocab_size)
- **Autoregressive Nature**: Sequential generation prevents parallelization

**Q10: How do you handle variable-length sequences in transformer models?**

**Answer:**
- **Padding**: Extend shorter sequences to maximum length
- **Attention Masking**: Prevent attention to padded positions  
- **Position Encoding**: Ensure model knows actual vs padded positions
- **Batching Strategy**: Group similar-length sequences when possible
- **Dynamic Batching**: More complex but efficient for production systems

### Quantization Deep Dive Questions

**Q11: Explain the difference between PTQ and QAT.**

**Answer:**
- **Post-Training Quantization (PTQ)**:
  - Train in full precision, then quantize
  - Use calibration dataset to determine quantization parameters
  - Faster to implement, may have performance drop
- **Quantization-Aware Training (QAT)**:
  - Include quantization in training process
  - Model learns to be robust to quantization errors
  - Better performance but more complex training

**Q12: What can be quantized in a neural network?**

**Answer:**
- **Weights**: Model parameters (most important, wider dynamic range)
- **Activations**: Intermediate outputs (more stable, easier to quantize)
- **Gradients**: During training (less common)
- **Considerations**: Weights need more careful quantization due to larger variance

### Beam Search and Temperature Questions

**Q13: Compare beam search with greedy decoding and sampling methods.**

**Answer:**
- **Greedy Decoding**: 
  - Always picks highest probability token
  - Fast but can get stuck in suboptimal paths
  - Deterministic output
- **Sampling Methods**: 
  - Introduces randomness for diversity
  - Can pick suboptimal tokens
  - Non-deterministic output
- **Beam Search**: 
  - Maintains multiple candidate sequences
  - Can recover from early mistakes
  - Higher quality but much slower (K times more computation)
  - Still makes greedy decisions at sequence level

**Q14: How does temperature affect the probability distribution in language models?**

**Answer:**
- **Mathematical Effect**: Divides logits by temperature before softmax
- **Low Temperature (T < 1)**: Sharpens distribution, more deterministic
- **High Temperature (T > 1)**: Flattens distribution, more random
- **Physical Analogy**: Like heating particles - higher temperature = more movement
- **Practical Impact**: 
  - Creative tasks: Higher temperature for diversity
  - Factual tasks: Lower temperature for consistency
  - Code generation: Very low temperature for precision

**Q15: What are the trade-offs of beam search?**

**Answer:**
- **Advantages**: 
  - Better quality than greedy for many tasks
  - Can explore multiple paths simultaneously
  - Recovers from early suboptimal decisions
- **Disadvantages**: 
  - K times slower than greedy decoding
  - Higher memory usage (stores K sequences)
  - Still makes locally optimal decisions
  - Doesn't guarantee global optimum

### Distributed Training Questions

**Q16: What are the different types of parallelism in distributed training?**

**Answer:**
- **Data Parallelism**: Same model on multiple GPUs, different data batches
- **Model Parallelism**: Different model layers on different GPUs
- **Pipeline Parallelism**: Different stages of computation pipelined across GPUs
- **Hybrid Approaches**: Combination of above methods for very large models

**Q17: What are the main challenges in distributed training?**

**Answer:**
- **Communication Overhead**: GPUs must synchronize gradients frequently
- **Load Balancing**: Ensuring equal utilization across all devices
- **Memory Management**: Coordinating memory usage across devices
- **Network Bottlenecks**: Bandwidth requirements for gradient synchronization
- **Fault Tolerance**: Handling device failures gracefully
- **Debugging Complexity**: Harder to debug distributed systems

### Advanced Optimization Questions

**Q18: Explain speculative decoding and its benefits.**

**Answer:**
- **Concept**: Use small, fast model to generate token hypotheses, large model to verify
- **Process**: Small model generates multiple tokens, large model scores them
- **Benefits**: Can skip multiple tokens when small model is correct
- **Trade-offs**: Maintains large model quality while improving speed
- **Implementation**: Requires careful threshold tuning for acceptance/rejection

**Q19: What is Mamba and how does it differ from traditional attention?**

**Answer:**
- **Mamba**: State-space model with linear complexity in sequence length
- **Traditional Attention**: O(N²) complexity becomes prohibitive for long sequences
- **Linear Attention**: Approximates attention with linear operations
- **Trade-offs**: Linear complexity vs. full attention expressiveness
- **Use Cases**: Better for very long sequences, may hurt short sequence performance

---

## 📊 Summary

### Core Takeaways

**Architecture Evolution:**
- Moved from encoder-decoder to decoder-only for simplicity and scalability
- Decoder-only models use only causal self-attention
- Enables deeper networks with same computational budget
- No lossy compression bottleneck from encoder

**Comprehensive Sampling Strategies:**
- **Greedy**: Deterministic, safe but potentially repetitive
- **Top-K**: Fixed-size candidate set, consistent randomness
- **Top-P**: Adaptive candidate set, balances confidence and diversity
- **Beam Search**: Multiple path exploration, higher quality but slower
- **Temperature Scaling**: Controls randomness through probability sharpening/flattening

**Training vs Inference Critical Gap:**
- Training uses teacher forcing with ground truth context
- Inference relies on autoregressive generation with model predictions
- KV caching essential for efficient inference (36x speedup possible)
- Distribution shift between training and inference phases

**LLM Optimization Landscape:**
- **Model-Level**: Reduce layers, heads, dimensions, vocabulary size
- **Attention-Level**: Multi-query attention, grouped query attention, window attention
- **Compression**: Quantization (FP32→INT4), pruning, knowledge distillation
- **Inference**: Speculative decoding, KV caching, linear attention alternatives
- **Distributed**: Data/model/pipeline parallelism for large-scale training

**Key Mathematical Foundations:**
- Probability distributions and sampling theory
- Softmax normalization ensures valid probabilities
- Linear transformations for dimension projection
- Temperature scaling for distribution control
- CDF and PDF relationships for statistical understanding

### Practical Implications

**For ML Engineers:**
- Master sampling parameters in generation APIs (temperature, top-k, top-p)
- Implement KV caching for production deployments
- Plan for variable inference times in capacity planning
- Understand quantization trade-offs for deployment
- Consider beam search for quality-critical applications

**For Researchers:**
- Teacher forcing creates fundamental train/inference mismatch
- Decoder-only architecture enables better scaling properties
- Sampling strategy significantly affects output quality and diversity
- Optimization techniques have complex trade-offs between speed, memory, and quality
- Distributed training essential for large model development

**For System Designers:**
- Network bandwidth becomes bottleneck in distributed training
- Memory management critical for KV caching implementation
- Error propagation in sampling can cause production issues
- Batching strategies must handle variable sequence lengths

### Advanced Topics for Further Study
- Implement all sampling algorithms from scratch
- Experiment with temperature scaling effects
- Study KV caching implementation details
- Explore knowledge distillation techniques
- Investigate speculative decoding implementations
- Research Mamba and linear attention alternatives
- Understand distributed training frameworks (DeepSpeed, FairScale)
- Practice quantization techniques (PTQ vs QAT)

---

*This study guide covers the fundamental concepts from Week 30's Generative AI Part 2 session. The concepts build upon each other, so ensure you understand each section before moving to the next.*