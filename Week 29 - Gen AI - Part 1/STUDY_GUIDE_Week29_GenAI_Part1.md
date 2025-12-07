# Week 29 - Generative AI Part 1: Comprehensive Study Guide

## 📚 Table of Contents
1. [Introduction to Generative AI](#introduction)
2. [Understanding Transformers - The Foundation](#transformers-foundation)
3. [Key Concepts Explained Simply](#key-concepts-simple)
4. [Technical Deep Dive](#technical-deep-dive)
5. [Interview Questions from Class](#interview-questions-class)
6. [Interview Questions for MLE/SDE-ML Roles](#interview-questions-roles)
7. [Summary](#summary)

---

## 🎯 Introduction to Generative AI {#introduction}

### What is Generative AI?
Imagine you have a super-smart robot that can create new things - write stories, draw pictures, or have conversations. That's Generative AI! Unlike regular AI that just classifies things (like saying "this is a cat" or "this is a dog"), Generative AI actually **creates** new content.

**Simple Analogy for a 12-year-old:**
Think of it like this:
- **Regular AI (Discriminative)**: Like a teacher grading your test - "This answer is correct" or "This is wrong"
- **Generative AI**: Like a creative writer who writes a whole new story based on what you tell them

### The Big Picture
This week focuses on **Text-to-Text Generation** - where you give the AI some text (a question or prompt), and it generates text back (an answer or continuation).

---

## 🏗️ Understanding Transformers - The Foundation {#transformers-foundation}

### Why Do We Need Transformers?

**The Old Way (RNNs and LSTMs):**
Imagine reading a book one word at a time, and you can only remember what you just read. By the time you get to page 100, you've forgotten what happened on page 1! That was the problem with older AI models called RNNs (Recurrent Neural Networks).

**Problems with RNNs:**
1. **Sequential Processing**: Like reading a book word-by-word, you can't skip ahead
2. **Memory Loss**: Information from earlier gets "forgotten" (vanishing gradients)
3. **Slow**: Can't process multiple words at the same time

**The Transformer Solution:**
Transformers are like having a photographic memory - they can "see" the entire sentence at once and remember all parts equally well!

### The Magic Ingredient: Attention Mechanism

**Simple Explanation:**
Imagine you're in a classroom, and the teacher asks, "Who can help with math homework?" 
- You **pay attention** to students who are good at math
- You **ignore** students who aren't helpful for this task
- Different questions make you pay attention to different people

That's exactly what the **Attention Mechanism** does! It helps the AI figure out which words in a sentence are important for understanding the current word.

---

## 🎨 Key Concepts Explained Simply {#key-concepts-simple}

### 1. Tokens and Embeddings

**What are Tokens?**
Think of tokens as puzzle pieces of language. The sentence "I am happy" might be broken into tokens: ["I", "am", "happy"]

**What are Embeddings?**
Embeddings are like giving each word a secret code (a list of numbers) that captures its meaning. Words with similar meanings get similar codes!

**Illustration:**
```
Word "dog" → [0.2, 0.8, 0.1, 0.9, ...]  (embedding vector)
Word "cat" → [0.3, 0.7, 0.2, 0.8, ...]  (similar to dog!)
Word "car" → [0.9, 0.1, 0.8, 0.2, ...]  (very different!)
```

### 2. Query, Key, and Value (Q, K, V)

**Simple Analogy:**
Imagine a library:
- **Query (Q)**: "I'm looking for books about space" (what you're searching for)
- **Key (K)**: The labels on each bookshelf (how books are organized)
- **Value (V)**: The actual books on the shelf (the content you get)

The attention mechanism:
1. Compares your Query with all the Keys (which shelves match your search?)
2. Gets the Values from the matching shelves (retrieves the relevant books)

### 3. Multi-Head Attention

**Why Multiple Heads?**
Imagine looking at a painting:
- One person focuses on colors
- Another focuses on shapes
- Another focuses on emotions

Each "head" in multi-head attention looks at the sentence from a different perspective!

**Example:**
For the sentence "The bank is by the river":
- Head 1 might focus on: "bank" (financial institution)
- Head 2 might focus on: "river" (riverbank)
- Together they understand: It's talking about a riverbank, not a financial bank!

### 4. Positional Encoding

**The Problem:**
Transformers process all words at once (parallel), but word order matters!
- "Dog bites man" ≠ "Man bites dog"

**The Solution:**
Add special numbers to each word's embedding that tell its position:
- Word 1 gets position code #1
- Word 2 gets position code #2
- And so on...

**Analogy:**
Like numbering pages in a book - even if pages get shuffled, you can put them back in order using the page numbers!

---

## 🔬 Technical Deep Dive {#technical-deep-dive}

### Architecture Overview

#### Encoder-Decoder vs Decoder-Only

**Original Transformer (Encoder-Decoder):**
```
Input → Encoder (compresses) → Decoder (generates) → Output
```

**Modern LLMs (Decoder-Only):**
```
Input + Previous Output → Decoder → Next Token
```

**Why Decoder-Only?**
- Simpler architecture
- Scales better with more data
- Can work directly with raw input (no compression needed)
- Examples: GPT-3, GPT-4, LLaMA

### Inside a Transformer Layer

Each transformer layer contains:

1. **Multi-Head Attention Block**
   - Projects input into Q, K, V spaces
   - Computes attention scores
   - Aggregates information from relevant positions

2. **Feed-Forward Network (MLP)**
   - Two linear transformations with activation
   - Processes each position independently

3. **Layer Normalization**
   - Stabilizes training
   - Applied before or after main computations

4. **Residual Connections**
   - Helps gradient flow during training
   - Prevents vanishing gradients

### Mathematical Formulation

**Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- Q = Query matrix
- K = Key matrix  
- V = Value matrix
- d_k = dimension of key vectors
- √d_k = scaling factor (prevents saturation)

**Why divide by √d_k?**
- Prevents dot products from becoming too large
- Keeps softmax in regions with good gradients
- Avoids saturation (where gradients become zero)

### Causal Masking (for Decoders)

**Purpose:** Prevent the model from "cheating" by looking at future tokens

**How it works:**
```
Sentence: "The cat sat on"
Position:   1   2   3  4

When predicting position 3 ("sat"):
- Can see: "The", "cat"
- Cannot see: "on" (future token)
```

**Implementation:** Set attention scores for future positions to -∞ before softmax

### Vocabulary Projection and Softmax

**Final Layer Operations:**
1. **Linear Projection**: Maps hidden state to vocabulary size
   - Input: [1 × d_model] (e.g., 1 × 1024)
   - Weight Matrix: [d_model × vocab_size] (e.g., 1024 × 128K)
   - Output: [1 × vocab_size] (logits for each word)

2. **Softmax**: Converts logits to probability distribution
   - Ensures all probabilities sum to 1
   - Each position represents P(word | context)

**Note:** This projection is computationally expensive! (Large matrix multiplication)

### Gradient Issues and Solutions

**Vanishing Gradients:**
- **Problem**: Gradients become very small during backpropagation
- **Solution**: Residual connections (skip connections)
- **How it helps**: Provides direct gradient path

**Exploding Gradients:**
- **Problem**: Gradients become very large
- **Solution**: Gradient clipping
- **How it works**: Cap gradients at a maximum value

---

## 💡 Interview Questions from Class {#interview-questions-class}

### Q1: Why did people move from RNNs to Transformers?

**Answer:**
1. **Sequential Processing Bottleneck**: RNNs process tokens one at a time, making them slow and hard to parallelize
2. **Vanishing Gradients**: Information from early tokens gets lost over long sequences
3. **Limited Context**: Hidden state acts as a bottleneck, compressing all history into a fixed-size vector
4. **Transformers solve these**: Parallel processing, attention mechanism for long-range dependencies, no compression bottleneck

### Q2: What's the difference between encoder and decoder attention?

**Answer:**
- **Encoder Attention**: Bi-directional (can see all tokens in input)
  - Used for understanding/representation
  - Example: BERT
  
- **Decoder Attention**: Causal/Masked (can only see previous tokens)
  - Used for generation
  - Prevents "cheating" during training
  - Example: GPT models

### Q3: Why do we need positional encoding?

**Answer:**
- Transformers process all tokens in parallel (no inherent order)
- Word order matters: "Dog bites man" ≠ "Man bites dog"
- Positional encoding adds position information to embeddings
- Without it, "I love you" and "You love I" would be identical to the model

### Q4: What is the purpose of multi-head attention?

**Answer:**
- Different heads learn different aspects of language
- One head might focus on syntax, another on semantics
- Provides multiple "views" of the same input
- Heads are learned independently through different projection matrices (WQ, WK, WV)
- Final output concatenates all heads

### Q5: Why divide by √d_k in attention formula?

**Answer:**
- Prevents dot products from becoming too large
- Large values push softmax into regions with very small gradients
- Small gradients = poor learning
- √d_k scaling keeps values in a reasonable range

### Q6: What's the difference between residual connections and gradient clipping?

**Answer:**
- **Residual Connections**: Solve vanishing gradients
  - Provide skip connections around layers
  - Allow gradients to flow directly through network
  
- **Gradient Clipping**: Solves exploding gradients
  - Caps gradient values at a maximum threshold
  - Prevents numerical overflow

### Q7: Why do modern LLMs use decoder-only architecture?

**Answer:**
- **Scaling Laws**: Decoder-only scales better with more data and parameters
- **Simplicity**: No need for separate encoder
- **Direct Input**: Can work with raw prompts without compression
- **Flexibility**: Same architecture for various tasks
- **Performance**: Empirically works better at scale

---

## 🎯 Interview Questions for MLE/SDE-ML Roles {#interview-questions-roles}

### Conceptual Questions

**Q1: Explain the attention mechanism to a non-technical person.**

**Answer:**
"Imagine reading a sentence and highlighting the most important words that help you understand it. Attention does exactly that - it helps the AI figure out which words to focus on when processing language. For example, in 'The animal didn't cross the street because it was too tired,' attention helps the AI understand that 'it' refers to 'animal,' not 'street.'"

**Q2: What are the computational bottlenecks in transformer models?**

**Answer:**
1. **Attention Computation**: O(n²) complexity where n = sequence length
   - Quadratic growth with sequence length
   - Memory intensive for long sequences

2. **Vocabulary Projection**: Final linear layer
   - Matrix size: [d_model × vocab_size]
   - Example: 1024 × 128K = 131M parameters
   - Computed for every token generation

3. **Solutions**:
   - Sparse attention patterns
   - Flash Attention (optimized CUDA kernels)
   - Smaller vocabulary with byte-pair encoding

**Q3: How would you optimize transformer inference for production?**

**Answer:**
1. **Model Optimization**:
   - Quantization (FP16, INT8)
   - Pruning unnecessary weights
   - Knowledge distillation (smaller student model)

2. **Inference Optimization**:
   - KV-cache (cache key-value pairs)
   - Batch processing
   - Speculative decoding

3. **Infrastructure**:
   - GPU optimization (TensorRT, ONNX)
   - Model parallelism for large models
   - Efficient serving frameworks (vLLM, TGI)

**Q4: What is the difference between pre-training and fine-tuning?**

**Answer:**
- **Pre-training**:
  - Train on massive unlabeled data
  - Learn general language understanding
  - Objective: Next token prediction
  - Very expensive (millions of dollars)

- **Fine-tuning**:
  - Train on specific task data
  - Adapt pre-trained model to task
  - Much cheaper and faster
  - Examples: Instruction tuning, RLHF

**Q5: Explain the role of temperature in text generation.**

**Answer:**
- **Temperature** controls randomness in sampling
- **Low temperature (0.1-0.5)**:
  - More deterministic
  - Picks highest probability tokens
  - Good for factual tasks
  
- **High temperature (0.8-1.5)**:
  - More creative/random
  - Samples from broader distribution
  - Good for creative writing

- **Formula**: `softmax(logits / temperature)`

### Coding/Implementation Questions

**Q6: How would you implement masked self-attention?**

**Answer:**
```python
import torch
import torch.nn.functional as F

def masked_self_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, seq_len, d_model]
    mask: [seq_len, seq_len] - causal mask
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (set future positions to -inf)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

**Q7: What are common issues when training transformers?**

**Answer:**
1. **Vanishing/Exploding Gradients**:
   - Use residual connections
   - Apply gradient clipping
   - Use proper initialization

2. **Overfitting**:
   - Dropout in attention and FFN
   - Weight decay
   - Data augmentation

3. **Training Instability**:
   - Layer normalization
   - Warmup learning rate schedule
   - Mixed precision training

4. **Memory Issues**:
   - Gradient checkpointing
   - Reduce batch size
   - Use gradient accumulation

### System Design Questions

**Q8: Design a system for serving a large language model in production.**

**Answer:**
**Components**:
1. **Load Balancer**: Distribute requests
2. **API Gateway**: Authentication, rate limiting
3. **Model Serving**: 
   - Multiple GPU instances
   - Model parallelism for large models
   - KV-cache for efficiency
4. **Caching Layer**: Cache common queries
5. **Monitoring**: Latency, throughput, errors
6. **Logging**: Track usage, debug issues

**Considerations**:
- Latency requirements (real-time vs batch)
- Cost optimization (GPU utilization)
- Scalability (auto-scaling)
- Reliability (failover, retries)

**Q9: How would you evaluate a text generation model?**

**Answer:**
**Automatic Metrics**:
1. **Perplexity**: How well model predicts next token
2. **BLEU/ROUGE**: Compare to reference texts
3. **BERTScore**: Semantic similarity

**Human Evaluation**:
1. **Relevance**: Does output answer the question?
2. **Coherence**: Is output logical and consistent?
3. **Fluency**: Is language natural?
4. **Factuality**: Is information correct?

**A/B Testing**:
- Compare different models/prompts
- Measure user engagement
- Track task completion rates

**Q10: Explain the difference between SFT and IFT.**

**Answer:**
**Supervised Fine-Tuning (SFT)**:
- Show many question-answer pairs
- Model learns from demonstrations
- Hope model figures out the pattern
- Less sample-efficient
- Example: Show 1000 addition problems

**Instruction Fine-Tuning (IFT)**:
- Explicitly teach reasoning process
- Include step-by-step explanations
- More sample-efficient
- Example: "To add numbers: take first, take second, combine"

**When to use**:
- SFT: When you have lots of data
- IFT: When you want efficient learning

**Q11: Design a RAG system for a company knowledge base.**

**Answer:**
**Architecture**:
```
Documents → Chunking → Embedding → Vector Store
                                        ↓
User Query → Embed → Similarity Search → Top-K Docs
                                        ↓
                            Prompt Augmentation
                                        ↓
                                      LLM
                                        ↓
                                    Response
```

**Components**:
1. **Document Processing**:
   - Chunk documents (500-1000 tokens)
   - Overlap chunks (50-100 tokens)
   - Generate embeddings
   - Store in vector DB (Pinecone, Weaviate, Chroma)

2. **Retrieval**:
   - Embed user query
   - Similarity search (cosine)
   - Retrieve top-5 to top-10 chunks
   - Optional: Re-rank results

3. **Generation**:
   - Augment prompt with retrieved context
   - Send to LLM
   - Generate response

**Considerations**:
- Chunk size vs context window
- Embedding model quality
- Retrieval accuracy
- Latency requirements
- Cost optimization

**Q12: What is contrastive loss and why is it used for embeddings?**

**Answer:**
**Contrastive Loss**:
```
Loss = Similarity(Anchor, Positive) - Similarity(Anchor, Negative)
```

**Purpose**:
- Pull similar items together
- Push dissimilar items apart
- Learn meaningful representations

**Example**:
- Anchor: "cat"
- Positive: "kitten" (should be close)
- Negative: "car" (should be far)

**Training**:
- Minimize distance between anchor and positive
- Maximize distance between anchor and negative
- Results in semantic embeddings

**Applications**:
- Text embeddings (sentence transformers)
- Image embeddings (SimCLR)
- Multimodal embeddings (CLIP)

**Q13: Explain document chunking strategies and trade-offs.**

**Answer:**
**Strategies**:

1. **Fixed-Size Chunking**:
   - Split by character/token count
   - Pros: Simple, predictable
   - Cons: May break mid-sentence

2. **Semantic Chunking**:
   - Split by paragraphs/sections
   - Pros: Preserves meaning
   - Cons: Variable sizes

3. **Overlapping Chunks**:
   - Include context from previous chunk
   - Pros: Maintains continuity
   - Cons: More storage, redundancy

**Trade-offs**:
- **Chunk Size**:
  - Too small: Lose context
  - Too large: Wash out information
  - Sweet spot: 500-1000 tokens

- **Overlap**:
  - More overlap: Better context, more storage
  - Less overlap: Less redundancy, may lose info

**Best Practices**:
- Use semantic boundaries when possible
- 10-20% overlap recommended
- Test different sizes for your use case

**Q14: How do you handle hallucinations in production LLM systems?**

**Answer:**
**Understanding Hallucinations**:
- Probabilistic sampling from learned distribution
- Wrong probabilities + bad sampling = hallucination
- Cannot be completely eliminated

**Mitigation Strategies**:

1. **RAG (Retrieval Augmented Generation)**:
   - Ground responses in retrieved facts
   - Provide source citations
   - Verify against knowledge base

2. **Temperature Control**:
   - Lower temperature (0.1-0.3) for factual tasks
   - Reduces randomness
   - More deterministic outputs

3. **Prompt Engineering**:
   - Explicit instructions: "Only use provided context"
   - Ask for citations
   - Request confidence scores

4. **Verification Layer**:
   - Fact-checking against sources
   - Consistency checks
   - Human review for critical outputs

5. **Confidence Thresholds**:
   - Monitor output probabilities
   - Flag low-confidence responses
   - Fallback to "I don't know"

**Production Checklist**:
- Use RAG for factual queries
- Set appropriate temperature
- Implement verification
- Monitor and log outputs
- Human-in-the-loop for critical tasks

**Q15: Compare different prompting techniques with examples.**

**Answer:**

**Zero-Shot**:
```
Prompt: "Translate to French: Hello"
Output: "Bonjour"
```
- No examples
- Relies on pre-training
- Fast, simple

**Few-Shot**:
```
Prompt: 
"English: Hello → French: Bonjour
English: Goodbye → French: Au revoir
English: Thank you → French: ?"
Output: "Merci"
```
- Provide examples
- Model learns pattern
- Better accuracy

**Chain-of-Thought**:
```
Prompt: "Solve: 23 + 47
Let's think step by step:
1. Add ones place: 3 + 7 = 10
2. Carry 1 to tens place
3. Add tens place: 2 + 4 + 1 = 7
4. Answer: 70"
```
- Include reasoning
- Better for complex tasks
- Improves accuracy

**ReAct**:
```
Question: "What's the weather in Paris?"
Thought: Need current data
Action: search("Paris weather today")
Observation: "15°C, cloudy"
Answer: "It's 15°C and cloudy in Paris"
```
- Combines reasoning + actions
- Can use tools
- Best for dynamic information

**When to Use**:
- Zero-shot: Simple, well-known tasks
- Few-shot: New tasks, need examples
- CoT: Complex reasoning, math
- ReAct: Need external tools/data

---

## 📝 Summary {#summary}

### Key Takeaways

1. **Transformers revolutionized NLP** by solving RNN limitations through parallel processing and attention mechanisms

2. **Attention is the core innovation** - allows models to focus on relevant parts of input dynamically

3. **Modern LLMs use decoder-only architecture** - simpler and scales better than encoder-decoder

4. **Key components**:
   - Multi-head attention (multiple perspectives)
   - Positional encoding (word order)
   - Residual connections (gradient flow)
   - Layer normalization (training stability)

5. **LLMs are probability models** - they predict next token given context, output is a distribution over vocabulary

6. **Practical considerations**:
   - Computational cost (attention is O(n²))
   - Memory requirements (large models need multiple GPUs)
   - Inference optimization (caching, quantization)

### Additional Topics Covered in Class

#### 1. **Fine-Tuning Strategies**

**Supervised Fine-Tuning (SFT)**:
- Extension of pre-training with task-specific data
- Show many examples of question-answer pairs
- Model learns patterns from demonstrations
- Like showing "2+2=4, 3+5=8" and hoping model learns addition
- More data-intensive but less sample-efficient

**Instruction Fine-Tuning (IFT)**:
- Explicitly teach HOW to solve problems
- Include reasoning steps, not just answers
- Format: "Given two numbers, add them like this..."
- More sample-efficient than SFT
- Teaches the model to think, not just memorize

**Key Difference**:
- SFT: "Here are many examples, figure it out"
- IFT: "Here's how to think about this problem"

#### 2. **Retrieval Augmented Generation (RAG)**

**What is RAG?**
- Enriching context with external knowledge
- Formula: `Generation(X₀ + Retrieved Context)`
- Solves knowledge cutoff problem
- Provides domain-specific information

**RAG Pipeline**:
```
User Query → Embed Query → Vector Search → Retrieve Top-K Documents
→ Augment Prompt → LLM Generation → Response
```

**Why RAG?**
1. **Knowledge Cutoff**: Pre-trained models have outdated information
2. **Domain-Specific**: Add proprietary/private knowledge
3. **Factual Grounding**: Base answers on retrieved documents
4. **Dynamic Updates**: No need to retrain model

**RAG Components**:
- **Vector Store**: Stores document embeddings
- **Retriever**: Finds relevant documents
- **Ranker**: Orders documents by relevance (optional)
- **Generator**: LLM that uses retrieved context

#### 3. **Vector Databases and Embeddings**

**What are Embeddings?**
- Dense vector representations of text
- Capture semantic meaning
- Similar texts → Similar vectors
- Example: `"cat" → [0.2, 0.8, 0.1, ...]`

**Embedding Models**:
- Convert text to fixed-size vectors
- Trained using contrastive loss
- Pull similar items together, push dissimilar apart

**Contrastive Loss**:
```
Loss = Similarity(Anchor, Positive) - Similarity(Anchor, Negative)
```
- **Anchor**: Reference item (e.g., "cat")
- **Positive**: Similar item (e.g., "kitten")
- **Negative**: Dissimilar item (e.g., "plant")

**Vector Stores**:
- Specialized databases for high-dimensional vectors
- Implement Approximate Nearest Neighbor (ANN) algorithms
- Examples: HNSW, Product Quantization, Annoy
- Handle curse of dimensionality

**Similarity Search**:
- Cosine similarity (most common)
- Euclidean distance
- Dot product

#### 4. **Document Chunking**

**Why Chunk?**
1. **Context Window Limits**: Models have max token limits
2. **Granular Retrieval**: More precise context matching
3. **Better Embeddings**: Avoid washing out information

**Chunking Strategies**:
- **Fixed-size**: Split by character/token count
- **Semantic**: Split by paragraphs, sentences
- **Recursive**: Try paragraphs → sentences → words

**Overlapping Chunks**:
```
Chunk 1: [Sentence 1, 2, 3, 4, 5]
Chunk 2:          [Sentence 4, 5, 6, 7, 8]
Chunk 3:                   [Sentence 7, 8, 9, 10, 11]
```
- Maintains context continuity
- Prevents information loss at boundaries
- Generally more effective than non-overlapping

#### 5. **Prompting Techniques**

**Zero-Shot**:
```
Q: What is 3 + 4?
A: 7
```
- Direct question, no examples
- Relies on pre-trained knowledge

**Few-Shot**:
```
Q: 2 + 3 = ?
A: 5

Q: 5 + 7 = ?
A: 12

Q: 3 + 4 = ?
A: [Model generates]
```
- Provide examples in prompt
- Model learns pattern from demonstrations

**Chain-of-Thought (CoT)**:
```
Q: What is 3 + 4?
A: To add two numbers:
   1. Take the first number: 3
   2. Take the second number: 4
   3. Combine them: 3 + 4 = 7
```
- Include reasoning steps
- Teaches model HOW to think
- Better for complex problems

**ReAct (Reason + Act)**:
```
Question: What's the weather today?
Thought: I need current information
Action: Search[current weather]
Observation: 72°F, sunny
Answer: It's 72°F and sunny today
```
- Model thinks before acting
- Can use external tools
- Combines reasoning with actions

#### 6. **Prompt Engineering**

**Why Important?**
- LLMs are probabilistic: `P(Y_t | X, Y_0...Y_{t-1})`
- Bad prompt → Bad first token → Cascading errors
- Context quality directly affects output quality

**Good Prompt Structure**:
1. **System Instruction**: Define role and behavior
2. **Context**: Provide relevant information
3. **Task**: Clear, specific request
4. **Format**: Specify output structure
5. **Examples**: Few-shot demonstrations (optional)

**Example**:
```
System: You are an expert Python programmer.
Context: User wants to sort a list of numbers.
Task: Write a Python function to sort a list in ascending order.
Format: Provide code with comments.
```

#### 7. **Multitask Learning**

**Traditional Approach**:
- Separate model for each task
- Separate loss functions
- Example: Object detection = Bounding box loss + Classification loss

**LLM Approach**:
- Single model, multiple tasks
- Same loss function (next token prediction)
- Task specified in prompt

**T5 Model Example**:
```
"translate English to German: Hello" → "Hallo"
"summarize: [long text]" → [summary]
"question: What is AI?" → [answer]
```
- Task prefix in prompt
- Model learns task from context
- No separate loss functions needed

#### 8. **Emergent Abilities**

**What are Emergent Abilities?**
- Capabilities not explicitly programmed
- Arise from scale (data + parameters)
- Examples: reasoning, math, coding

**How Do They Emerge?**
- Training objective: Fill in the blanks
- To fill blanks correctly, model must understand
- Understanding math → Can solve math problems
- Understanding code → Can write code

**Key Insight**:
- Don't need separate models for each skill
- General language modeling → Multiple abilities
- "Fill in the blanks" at scale → Intelligence

#### 9. **Hallucinations**

**Why Do LLMs Hallucinate?**
1. **Probabilistic Nature**: Sampling from distribution
2. **Wrong Histogram**: Model's probabilities are off
3. **Bad Sampling**: Pick wrong token from distribution
4. **Cascading Errors**: One bad token → More bad tokens

**Cannot Be Completely Eliminated**:
- Inherent to probabilistic models
- Can reduce but not eliminate
- Always need verification for critical tasks

**Mitigation Strategies**:
- RAG: Ground in retrieved facts
- Temperature control: Lower = more deterministic
- Verification: Check outputs against sources
- Human-in-the-loop: Review critical outputs

#### 10. **Scaling Laws**

**What Are Scaling Laws?**
- Relationship between model size, data, and performance
- Larger models + More data = Better performance
- Predictable improvement patterns

**Why Decoder-Only Works**:
- Scales better than encoder-decoder
- Can work directly with raw input
- No compression bottleneck
- Empirically better at scale

**Key Findings**:
- Performance improves predictably with scale
- Compute-optimal training exists
- Bigger isn't always better (efficiency matters)

#### 11. **Sampling and Generation**

**Temperature**:
- Controls randomness in sampling
- Low (0.1-0.5): Deterministic, factual
- High (0.8-1.5): Creative, diverse
- Formula: `softmax(logits / temperature)`

**Top-p (Nucleus) Sampling**:
- Sample from smallest set with cumulative probability ≥ p
- More dynamic than top-k
- Adapts to confidence

**Top-k Sampling**:
- Sample from top k most likely tokens
- Fixed cutoff
- Simpler than top-p

**Greedy Decoding**:
- Always pick highest probability token
- Deterministic
- Can be repetitive

### What's Next?

In **Week 29 Part 2**, you'll learn:
- Advanced agentic AI patterns
- LangChain for building applications
- Tool use and function calling
- Production deployment strategies
- Advanced RAG techniques

### Resources for Further Learning

1. **Papers**:
   - "Attention Is All You Need" (Original Transformer)
   - "BERT: Pre-training of Deep Bidirectional Transformers"
   - "Language Models are Few-Shot Learners" (GPT-3)

2. **Courses**:
   - Stanford CS224N (NLP with Deep Learning)
   - Fast.ai NLP Course
   - Hugging Face Course

3. **Tools**:
   - Hugging Face Transformers
   - LangChain
   - OpenAI API

---

## 🎓 Study Tips

1. **Understand the fundamentals** - Don't just memorize, understand WHY each component exists
2. **Implement from scratch** - Build a simple transformer to truly understand it
3. **Experiment with code** - Use Hugging Face to try different models
4. **Practice explaining** - Can you explain attention to a 12-year-old? To a technical interviewer?
5. **Stay updated** - Field moves fast, follow papers and blogs

---

*This study guide was created from Week 29 - Generative AI Part 1 class materials. For questions or clarifications, refer to the class recording and coding notebooks.*
