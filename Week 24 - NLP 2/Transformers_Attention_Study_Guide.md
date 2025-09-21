# Transformers & Attention Study Guide - Week 24
*Explaining Transformers and Attention mechanisms like to a smart 12-year old, then diving into technical details*

## 🎯 Table of Contents
1. [Simple Explanations with Illustrations](#simple-explanations)
2. [Technical Deep Dive](#technical-concepts)
3. [Interview Questions & Answers](#interview-questions)

---

## 🌟 Simple Explanations with Illustrations {#simple-explanations}

### What are Transformers?

**For a 12-year old:**
Imagine you're in a classroom where everyone needs to work together on a group project, but instead of passing notes one by one (like in old schools), everyone can talk to everyone else at the same time!

**Old way (RNNs/LSTMs):** Like a telephone game
```
Student 1 → Student 2 → Student 3 → Student 4
(Information gets lost and takes forever)
```

**New way (Transformers):** Like a group video call
```
Student 1 ↔ Student 2
    ↕         ↕
Student 4 ↔ Student 3
(Everyone talks to everyone instantly!)
```

### What is Attention?

**Simple:** "Paying attention to the most important parts"

**Real-life analogy:** Reading a book
- When you read "The big red **apple** was delicious", your brain automatically connects:
  - "big" and "red" describe the **apple**
  - "delicious" tells us about eating the **apple**
  - Your brain "pays attention" to these connections!

**In AI:** The computer learns to make the same connections between words automatically.

### Why Transformers are Revolutionary 🚀

#### Problem with Old Models (Seq2Seq)
**Like:** Having to read a book one word at a time, and only remembering the last few words
- **Bottleneck:** All information squeezed through one tiny "context vector"
- **Sequential:** Can't process words in parallel (very slow!)
- **Memory loss:** Forgets earlier parts of long sentences

#### Transformer Solution
**Like:** Reading the whole page at once and understanding all connections
- **No bottleneck:** Every word can connect to every other word
- **Parallel processing:** All words processed simultaneously (super fast!)
- **Perfect memory:** Never forgets any part of the input
### 
The Transformer Architecture 🏗️

**Simple Analogy:** Think of it like a translation factory with two main departments:

#### 1. Encoder Department 📥
**Job:** "Understand the input really, really well"
- Takes English sentence: "I love cats"
- Creates deep understanding of what each word means
- Builds connections between all words
- **Output:** Rich understanding ready for translation

#### 2. Decoder Department 📤  
**Job:** "Generate the output step by step"
- Takes the encoder's understanding
- Generates French translation: "J'aime les chats"
- Uses attention to focus on relevant English words for each French word

#### 3. Attention Mechanism 🔍
**Job:** "The connection system between all words"
- **Self-Attention:** Words in same sentence talking to each other
- **Cross-Attention:** English words helping generate French words
- **Multi-Head Attention:** Multiple "attention experts" working in parallel

### Types of Attention

#### 1. Self-Attention 🪞
**Simple:** "Words in the same sentence paying attention to each other"

**Example:** "The animal didn't cross the street because **it** was too tired"
- The word "**it**" pays attention to "**animal**" (not "street")
- The model learns this connection automatically!

#### 2. Cross-Attention 🌉
**Simple:** "Words from different sentences paying attention to each other"

**Example:** Translating "I love cats" → "J'aime les chats"
- French "J'aime" pays attention to English "I love"  
- French "chats" pays attention to English "cats"

#### 3. Multi-Head Attention 👥
**Simple:** "Multiple attention experts working together"

**Analogy:** Like having different specialists:
- **Expert 1:** Focuses on grammar connections
- **Expert 2:** Focuses on meaning connections  
- **Expert 3:** Focuses on context connections
- **Combined:** Much better understanding than any single expert!---


## 🔬 Technical Deep Dive {#technical-concepts}

### Transformer Architecture Overview

**Mathematical Foundation:**
The Transformer follows an encoder-decoder architecture where:
```
Input → Encoder → Decoder → Output
```

**Key Innovation:** Attention mechanism replaces recurrence and convolution entirely.

#### Core Components

**1. Multi-Head Attention**
**2. Position-wise Feed-Forward Networks**  
**3. Positional Encoding**
**4. Layer Normalization**
**5. Residual Connections**

### Self-Attention Mechanism

#### Mathematical Formulation

**Attention Function:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What can I offer?"  
- **V (Value):** "What information do I actually provide?"
- **d_k:** Dimension of key vectors (for scaling)

#### Step-by-Step Calculation

**Step 1: Create Q, K, V matrices**
```python
# Input embeddings: X ∈ R^(seq_len × d_model)
Q = X @ W_Q  # Query matrix
K = X @ W_K  # Key matrix  
V = X @ W_V  # Value matrix

# Where W_Q, W_K, W_V ∈ R^(d_model × d_k)
```

**Step 2: Compute attention scores**
```python
# Compute dot-product attention
scores = Q @ K.T / math.sqrt(d_k)  # Shape: (seq_len, seq_len)
```

**Step 3: Apply softmax**
```python
attention_weights = softmax(scores)  # Normalize to probabilities
```

**Step 4: Apply attention to values**
```python
output = attention_weights @ V  # Weighted combination of values
```#
## Multi-Head Attention

#### Concept
Instead of single attention, use multiple "attention heads" in parallel:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)  
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. Linear projections
        Q = self.W_Q(query)  # (batch, seq_len, d_model)
        K = self.W_K(key)
        V = self.W_V(value)
        
        # 2. Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 5. Final linear projection
        return self.W_O(attention_output)
```

#### Why Multiple Heads?
- **Different perspectives:** Each head can focus on different types of relationships
- **Richer representations:** Captures various linguistic phenomena simultaneously
- **Parallel processing:** All heads computed simultaneously

### Positional Encoding

#### Problem
Transformers have no inherent notion of sequence order (unlike RNNs).

#### Solution
Add positional information to input embeddings:

```python
def positional_encoding(seq_len, d_model):
    pos_enc = torch.zeros(seq_len, d_model)
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    
    return pos_enc
```

**Mathematical Formula:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties:**
- **Deterministic:** Same position always gets same encoding
- **Relative positions:** Model can learn relative distances
- **Extrapolation:** Can handle sequences longer than training### 
Complete Transformer Architecture

#### Encoder Stack

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection  
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x
```

#### Decoder Stack

```python
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        self_attn = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn)
        
        # Cross-attention with encoder
        cross_attn = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + cross_attn)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x
```

### Key Architectural Innovations

#### 1. Residual Connections
```python
# Instead of: output = layer(input)
# Use: output = input + layer(input)
```
**Benefits:**
- **Gradient flow:** Helps with vanishing gradient problem
- **Training stability:** Easier to train very deep networks
- **Identity mapping:** Network can learn to ignore layers if needed

#### 2. Layer Normalization
```python
def layer_norm(x, eps=1e-6):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)
```
**Benefits:**
- **Training stability:** Normalizes activations
- **Faster convergence:** Reduces internal covariate shift
- **Better gradients:** Prevents exploding/vanishing gradients---


## 🎤 Interview Questions & Detailed Answers {#interview-questions}

### Fundamental Concepts

#### Q1: Explain the attention mechanism. How does it work mathematically?

**Answer:**

**Intuitive Explanation:**
Attention allows the model to focus on relevant parts of the input when making predictions. It's like highlighting important words in a text while reading.

**Mathematical Formulation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Step-by-Step Process:**

**1. Create Query, Key, Value matrices:**
```python
# Given input X ∈ R^(seq_len × d_model)
Q = X @ W_Q  # What am I looking for?
K = X @ W_K  # What can I offer?  
V = X @ W_V  # What information do I provide?
```

**2. Compute similarity scores:**
```python
scores = Q @ K.T / math.sqrt(d_k)
# Shape: (seq_len, seq_len)
# scores[i,j] = similarity between position i and j
```

**3. Normalize with softmax:**
```python
attention_weights = softmax(scores)
# Each row sums to 1 (probability distribution)
```

**4. Weighted combination:**
```python
output = attention_weights @ V
# Weighted sum of all value vectors
```

**Why This Works:**
- **Q·K similarity:** Measures how much position i should attend to position j
- **Softmax normalization:** Ensures attention weights sum to 1
- **Weighted V:** Combines information based on attention weights
- **Scaling by √d_k:** Prevents softmax saturation for large dimensions

**Example:**
For sentence "The cat sat on the mat":
- When processing "sat", attention might focus heavily on "cat" (subject-verb relationship)
- Attention weights: [0.1, 0.7, 0.1, 0.05, 0.03, 0.02] 
- Output combines all word representations weighted by these scores

#### Q2: What are the key differences between Transformers and RNNs/LSTMs?

**Answer:**

**Architectural Differences:**

| Aspect | RNNs/LSTMs | Transformers |
|--------|------------|--------------|
| **Processing** | Sequential | Parallel |
| **Memory** | Hidden state | Attention mechanism |
| **Long-range dependencies** | Difficult (vanishing gradients) | Easy (direct connections) |
| **Computational complexity** | O(n) per layer | O(n²) per layer |
| **Parallelization** | Not possible | Fully parallelizable |

**Detailed Comparison:**

**1. Sequential vs Parallel Processing:**
```python
# RNN: Must process sequentially
h_1 = RNN(x_1, h_0)
h_2 = RNN(x_2, h_1)  # Must wait for h_1
h_3 = RNN(x_3, h_2)  # Must wait for h_2

# Transformer: All positions processed simultaneously  
output = Attention(X, X, X)  # All positions at once
```

**2. Memory Mechanism:**
```python
# RNN: Fixed-size hidden state (bottleneck)
h_t = f(x_t, h_{t-1})  # Information compressed into h_t

# Transformer: Direct access to all positions
attention_weights = softmax(Q @ K.T / √d_k)
output = attention_weights @ V  # No information loss
```

**3. Long-range Dependencies:**
- **RNN:** Information must pass through many hidden states
  - Path length: O(n) for positions n steps apart
  - Gradient vanishing over long sequences
- **Transformer:** Direct connections between all positions
  - Path length: O(1) between any two positions
  - No gradient vanishing across positions

**4. Computational Complexity:**
- **RNN:** O(n·d²) total, but O(n) sequential steps
- **Transformer:** O(n²·d) total, but O(1) parallel steps

**When to Use Each:**
- **RNNs:** Small sequences, limited compute, streaming applications
- **Transformers:** Large sequences, abundant compute, batch processing

#### Q3: Explain multi-head attention. Why is it better than single-head attention?

**Answer:**

**Multi-Head Attention Concept:**
Instead of using one attention mechanism, use multiple "heads" in parallel, each learning different types of relationships.

**Mathematical Formulation:**
```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
```

**Implementation:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64 per head
        
        # Separate projections for each component
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. Linear projections
        Q = self.W_Q(query)  # (batch, seq_len, d_model)
        K = self.W_K(key)
        V = self.W_V(value)
        
        # 2. Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 3. Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 4. Apply attention to each head
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        
        # 5. Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        
        # 6. Final linear projection
        return self.W_O(attention_output)
```

**Why Multi-Head is Better:**

**1. Different Relationship Types:**
```python
# Example: "The cat that I saw yesterday was sleeping"
# Head 1: Syntactic relationships (cat ↔ was, that ↔ saw)
# Head 2: Semantic relationships (cat ↔ sleeping, saw ↔ yesterday)  
# Head 3: Coreference (cat ↔ I, that ↔ cat)
```

**2. Representation Subspaces:**
- Each head operates in different subspace (d_k = d_model/h)
- Allows specialization without interference
- Richer overall representation

**3. Ensemble Effect:**
- Multiple "experts" voting on attention
- Reduces overfitting to single attention pattern
- More robust predictions

**4. Empirical Benefits:**
- **Better performance:** Consistently outperforms single-head
- **Interpretability:** Different heads capture different phenomena
- **Robustness:** Less sensitive to initialization

**Typical Configuration:**
- **BERT:** 12 heads, d_model=768, d_k=64 per head
- **GPT:** 12-96 heads depending on model size
- **T5:** 12-32 heads depending on variant##
## Q4: How do Transformers handle variable-length sequences and what is masking?

**Answer:**

**Variable-Length Sequence Challenges:**
1. **Batch processing:** Need uniform tensor shapes
2. **Attention computation:** Prevent attending to padding tokens
3. **Loss calculation:** Ignore padding in loss computation

**Solutions:**

**1. Padding:**
```python
# Example batch with variable lengths
sequences = [
    "Hello world",           # Length: 2
    "How are you today",     # Length: 4  
    "Good morning"           # Length: 2
]

# After padding (pad_token_id = 0)
padded_sequences = [
    [101, 7592, 2088, 0],    # "Hello world" + padding
    [101, 2129, 2024, 2017], # "How are you today"  
    [101, 2204, 2851, 0]     # "Good morning" + padding
]
```

**2. Attention Masking:**

**Padding Mask:**
```python
def create_padding_mask(seq, pad_token_id=0):
    """Mask padding tokens in attention computation"""
    # seq shape: (batch_size, seq_len)
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
    # mask shape: (batch_size, 1, 1, seq_len)
    return mask

def apply_attention_mask(scores, mask):
    """Apply mask to attention scores"""
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    return scores

# Usage in attention
scores = Q @ K.T / math.sqrt(d_k)
scores = apply_attention_mask(scores, padding_mask)
attention_weights = softmax(scores)  # Masked positions → 0 probability
```

**Look-Ahead Mask (for Decoder):**
```python
def create_look_ahead_mask(seq_len):
    """Prevent decoder from seeing future tokens"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # Lower triangular matrix

# Example for sequence length 4:
# [[True,  False, False, False],
#  [True,  True,  False, False], 
#  [True,  True,  True,  False],
#  [True,  True,  True,  True ]]
```

**Combined Masking:**
```python
def create_decoder_mask(tgt_seq, pad_token_id=0):
    """Combine padding and look-ahead masks"""
    seq_len = tgt_seq.size(1)
    
    # Padding mask
    padding_mask = create_padding_mask(tgt_seq, pad_token_id)
    
    # Look-ahead mask  
    look_ahead_mask = create_look_ahead_mask(seq_len)
    
    # Combine masks (both must be True to attend)
    combined_mask = padding_mask & look_ahead_mask
    return combined_mask
```

**3. Loss Masking:**
```python
def masked_cross_entropy_loss(predictions, targets, pad_token_id=0):
    """Compute loss ignoring padding tokens"""
    # predictions: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len)
    
    # Create mask for non-padding tokens
    mask = (targets != pad_token_id).float()
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(
        predictions.view(-1, predictions.size(-1)),
        targets.view(-1),
        reduction='none'
    )
    
    # Apply mask and compute mean over non-padding tokens
    masked_loss = loss * mask.view(-1)
    return masked_loss.sum() / mask.sum()
```

**4. Positional Encoding with Variable Lengths:**
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        # Pre-compute positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]  # Slice to actual sequence length
```

**Best Practices:**
1. **Consistent padding:** Use same pad_token_id across dataset
2. **Efficient masking:** Pre-compute masks when possible
3. **Memory optimization:** Use attention_mask parameter in libraries
4. **Validation:** Ensure masks are applied correctly in all components

#### Q5: Compare BERT and GPT architectures. What are their key differences?

**Answer:**

**Architectural Overview:**

| Aspect | BERT | GPT |
|--------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Attention** | Bidirectional | Unidirectional (causal) |
| **Training Objective** | Masked Language Modeling | Next Token Prediction |
| **Use Case** | Understanding tasks | Generation tasks |
| **Input Processing** | Full sequence at once | Autoregressive |

**Detailed Comparison:**

**1. Architecture Differences:**

**BERT (Encoder-only):**
```python
class BERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)  # Stack of encoder layers
        self.pooler = BERTPooler(config)
        
    def forward(self, input_ids, attention_mask=None):
        # Bidirectional attention - can see all tokens
        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        return encoder_outputs

class BERTEncoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention (bidirectional)
        self_attention_outputs = self.attention(
            hidden_states, hidden_states, hidden_states, attention_mask
        )
        # No causal masking - can attend to all positions
        return self_attention_outputs
```

**GPT (Decoder-only):**
```python
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = GPTEmbeddings(config)
        self.decoder = GPTDecoder(config)  # Stack of decoder layers
        
    def forward(self, input_ids, attention_mask=None):
        # Causal attention - can only see previous tokens
        embedding_output = self.embeddings(input_ids)
        decoder_outputs = self.decoder(embedding_output, attention_mask)
        return decoder_outputs

class GPTDecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        # Masked self-attention (causal)
        causal_mask = self.create_causal_mask(hidden_states.size(1))
        self_attention_outputs = self.attention(
            hidden_states, hidden_states, hidden_states, 
            attention_mask & causal_mask  # Combine with causal mask
        )
        return self_attention_outputs
```

**2. Training Objectives:**

**BERT - Masked Language Modeling (MLM):**
```python
# Original: "The cat sat on the mat"
# Masked:   "The [MASK] sat on the [MASK]"
# Target:   Predict "cat" and "mat"

def bert_mlm_loss(model, input_ids, labels):
    # input_ids: [101, 1996, 103, 2938, 2006, 1996, 103, 102]
    #            [CLS] The [MASK] sat  on   the [MASK] [SEP]
    
    outputs = model(input_ids)
    predictions = outputs.logits
    
    # Only compute loss on masked positions
    mask_positions = (input_ids == 103)  # [MASK] token
    loss = F.cross_entropy(
        predictions[mask_positions], 
        labels[mask_positions]
    )
    return loss
```

**GPT - Next Token Prediction:**
```python
# Input:  "The cat sat on"
# Target: "cat sat on the"

def gpt_causal_loss(model, input_ids):
    # input_ids: [1996, 4937, 2938, 2006]  # "The cat sat on"
    # targets:   [4937, 2938, 2006, 1996]  # "cat sat on the"
    
    outputs = model(input_ids)
    logits = outputs.logits
    
    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss
```

**3. Attention Patterns:**

**BERT Attention (Bidirectional):**
```python
# Can attend to all positions
attention_mask = torch.ones(seq_len, seq_len)
# [[1, 1, 1, 1],
#  [1, 1, 1, 1],
#  [1, 1, 1, 1], 
#  [1, 1, 1, 1]]
```

**GPT Attention (Causal):**
```python
# Can only attend to previous positions
attention_mask = torch.tril(torch.ones(seq_len, seq_len))
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

**4. Use Cases and Applications:**

**BERT Applications:**
- **Text Classification:** Sentiment analysis, spam detection
- **Named Entity Recognition:** Extract entities from text
- **Question Answering:** Find answers in passages
- **Text Similarity:** Compare document similarity

**GPT Applications:**
- **Text Generation:** Creative writing, code generation
- **Conversational AI:** Chatbots, virtual assistants  
- **Text Completion:** Auto-complete, suggestions
- **Few-shot Learning:** In-context learning

**5. Fine-tuning Approaches:**

**BERT Fine-tuning:**
```python
# Add task-specific head
class BERTForClassification(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        logits = self.classifier(pooled_output)
        return logits
```

**GPT Fine-tuning:**
```python
# Continue pre-training or use prompting
class GPTForGeneration(nn.Module):
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt = gpt_model
        
    def generate(self, prompt_ids, max_length=100):
        generated = prompt_ids
        
        for _ in range(max_length):
            outputs = self.gpt(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
        return generated
```

**Key Takeaways:**
- **BERT:** Better for understanding and classification tasks
- **GPT:** Better for generation and completion tasks  
- **Architecture choice:** Depends on whether you need bidirectional context
- **Modern trend:** Decoder-only models (GPT-style) becoming dominant-
--

## 📚 Additional Resources

### Key Papers to Read
1. **"Attention Is All You Need" (2017):** Original Transformer paper
2. **"BERT: Pre-training of Deep Bidirectional Transformers" (2018)**
3. **"Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)**
4. **"The Illustrated Transformer"** by Jay Alammar (blog post)

### Practical Implementation
- **Libraries:** Transformers (Hugging Face), PyTorch, TensorFlow
- **Pre-trained Models:** BERT, GPT, T5, RoBERTa, DistilBERT
- **Fine-tuning:** Classification, NER, Question Answering, Generation

### Key Concepts to Master
1. **Self-Attention Mechanism:** Query, Key, Value matrices
2. **Multi-Head Attention:** Parallel attention heads
3. **Positional Encoding:** Adding sequence order information
4. **Encoder-Decoder Architecture:** Understanding both components
5. **Masking:** Padding masks and causal masks
6. **Layer Normalization & Residual Connections:** Training stability

### Next Steps
- **Week 25 (NLP 3):** BERT fine-tuning, GPT applications, Transfer Learning
- **Advanced Topics:** Vision Transformers, CLIP, GPT-4, ChatGPT
- **Practical Projects:** Build your own transformer, fine-tune BERT

### Interview Preparation Tips
1. **Understand the math:** Be able to derive attention formulation
2. **Know the differences:** BERT vs GPT, Encoder vs Decoder
3. **Practical experience:** Implement attention mechanism from scratch
4. **Current trends:** Stay updated with latest transformer variants
5. **Applications:** Know when to use which architecture

---

*This study guide covers the fundamental concepts from Week 24's Transformers and Attention session. The transformer architecture is the foundation of modern NLP, so make sure to understand these concepts thoroughly!*