# Week 30 - Generative AI Part 2: Complete Learning Resources

## 📚 Overview

This folder contains comprehensive study materials, coding guides, and interview preparation resources for **Week 30: Generative AI Part 2**. The materials cover decoder-only models, sampling techniques, KV caching, knowledge distillation, and LLM optimization strategies.

---

## 📂 File Structure and Contents

### 1. **Week30_GenAI_Part2_Study_Guide.md** ⭐ START HERE
**Comprehensive Study Guide** - Your primary learning resource

**Contents:**
- Core concepts explained simply (like teaching a 12-year-old)
- Technical deep dives with mathematical foundations
- LLM optimization techniques (vocabulary, architecture, attention, FFN, distributed training)
- Model compression strategies (distillation, pruning, quantization)
- Advanced inference optimizations (speculative decoding, linear attention, beam search)
- 35+ interview questions with detailed answers
- Key points from class sessions
- Summary of critical concepts

**How to Use:**
1. Start with "Core Concepts Explained Simply" section
2. Move to "Technical Deep Dive" for mathematical details
3. Review "Key Points from Class" for important takeaways
4. Practice with "Interview Questions & Answers"
5. Use "Summary" as a quick reference

**Time to Complete:** 4-6 hours for thorough understanding

---

### 2. **Coding Guides** (4 Comprehensive Guides)

#### a) **KV_Caching_Coding_Guide.md**
**Focus:** Key-Value caching for efficient inference

**Key Topics:**
- Why KV caching is essential (36x speedup)
- Cache initialization and management
- Forward pass with caching
- Performance comparison (with vs without cache)
- Memory vs computation trade-offs
- Multi-Query Attention (MQA) optimization

**Code Examples:**
- Sampler base class
- GreedySampler implementation
- SelfAttention with KV cache
- Performance benchmarking

**Interview Questions:** 8 questions on KV caching concepts

**Time to Complete:** 2-3 hours

---

#### b) **Sampling_Coding_Guide.md**
**Focus:** Text generation sampling strategies

**Key Topics:**
- From logits to tokens pipeline
- Greedy search implementation
- Temperature scaling effects
- Top-K and Top-P sampling (extensions)
- Visualization of sampling decisions
- Probability distribution manipulation

**Code Examples:**
- Basic text generation
- Manual model inference
- Utility functions for sampling
- Graph visualization of decisions

**Interview Questions:** 5 questions on sampling strategies

**Time to Complete:** 2-3 hours

---

#### c) **KnowledgeDistillation_Coding_Guide.md**
**Focus:** Compressing large models into smaller ones

**Key Topics:**
- Knowledge distillation concept
- Temperature scaling in softmax
- KL divergence loss
- Teacher-student training
- Multi-teacher distillation
- Feature-level distillation

**Code Examples:**
- Loss function setup
- KD training step
- Model mode management
- Backpropagation and optimization

**Interview Questions:** 6 questions on distillation

**Time to Complete:** 2-3 hours

---

#### d) **Gen_AI_1_Assignment_Solution_Coding_Guide.md**
**Focus:** Practical T5 fine-tuning for toxicity detection

**Key Topics:**
- Multi-label classification with T5
- Custom PyTorch dataset implementation
- Fine-tuning pre-trained models
- Training loop implementation
- Model checkpointing
- Text-to-text paradigm

**Code Examples:**
- Data loading and exploration
- Dataset class implementation
- Training setup and loop
- Model evaluation

**Interview Questions:** 5 questions on fine-tuning

**Time to Complete:** 2-3 hours

---

### 3. **Jupyter Notebooks** (4 Implementation Notebooks)

#### a) **KV_Caching.ipynb**
- Practical implementation of KV caching
- Performance benchmarking code
- Comparison with/without caching
- Real-world speedup measurements

#### b) **Sampling.ipynb**
- Various sampling strategies
- Visualization of sampling decisions
- Interactive examples
- Performance comparisons

#### c) **KnowledgeDistillation.ipynb**
- Teacher-student training loop
- Temperature scaling experiments
- Loss computation
- Model compression results

#### d) **Gen_AI_1_Assignment_Solution.ipynb**
- Complete toxicity detection pipeline
- T5 fine-tuning example
- Multi-label classification
- Training and evaluation

---

### 4. **Class Materials**

#### Transcripts
- `meeting_saved_closed_caption.txt` - Main class session
- `meeting_saved_closed_caption copy.txt` - Assignment review
- `meeting_saved_closed_caption copy 2.txt` - Q&A session
- `meeting_saved_closed_caption copy 3.txt` - Additional session

#### Slides
- `GenerativeAI-Text2Text Part2_1stMar26.pdf` - Official presentation slides

---

## 🎯 Learning Paths

### Path 1: Quick Overview (2-3 hours)
1. Read "Core Concepts Explained Simply" in Study Guide
2. Skim KV Caching and Sampling guides
3. Review interview questions
4. **Outcome:** Understand main concepts, ready for basic interviews

### Path 2: Comprehensive Learning (8-10 hours)
1. Complete Study Guide (all sections)
2. Read all 4 coding guides
3. Run through Jupyter notebooks
4. Practice interview questions
5. **Outcome:** Deep understanding, ready for technical interviews

### Path 3: Implementation Focus (6-8 hours)
1. Focus on coding guides
2. Run and modify Jupyter notebooks
3. Implement concepts from scratch
4. Debug and optimize code
5. **Outcome:** Practical coding skills, ready for implementation interviews

### Path 4: Interview Preparation (4-6 hours)
1. Review Study Guide summary
2. Practice all 35+ interview questions
3. Prepare examples and use cases
4. Mock interview practice
5. **Outcome:** Interview-ready, confident answers

---

## 🔑 Key Concepts at a Glance

### Architecture
- **Decoder-Only**: GPT, Llama, Claude (autoregressive generation)
- **Encoder-Only**: BERT, RoBERTa (classification, understanding)
- **Encoder-Decoder**: T5, BART (sequence-to-sequence)

### Sampling Methods
- **Greedy**: Fastest, deterministic, can be repetitive
- **Top-K**: Fixed vocabulary, balanced
- **Top-P**: Adaptive vocabulary, natural
- **Beam Search**: Best quality, slowest

### Optimizations
- **KV Caching**: 36x speedup for inference
- **Quantization**: 4-8x memory reduction
- **Distillation**: Compress large models
- **LoRA**: Parameter-efficient fine-tuning

### Training Concepts
- **Pre-training**: Next token prediction on massive data
- **Instruction Tuning**: Learn to follow instructions
- **Fine-tuning**: Adapt to specific tasks
- **Teacher Forcing**: Use ground truth during training

---

## 📊 Topics Covered

### Fundamental Concepts
- ✅ Transformer architecture (encoder, decoder, attention)
- ✅ Positional encoding and layer normalization
- ✅ Causal masking and autoregressive generation
- ✅ Attention mechanisms (self, cross, multi-head)

### Sampling and Generation
- ✅ Logits to probabilities conversion
- ✅ Greedy, top-k, top-p, beam search
- ✅ Temperature scaling
- ✅ Sampling vs deterministic generation

### Efficiency and Optimization
- ✅ KV caching (36x speedup)
- ✅ Vocabulary and dimension reduction
- ✅ Multi-Query Attention (MQA)
- ✅ Grouped Query Attention (GQA)
- ✅ Window attention and latent attention

### Model Compression
- ✅ Knowledge distillation
- ✅ Model pruning
- ✅ Quantization (PTQ, QAT)
- ✅ LoRA (Low-Rank Adaptation)

### Training and Fine-tuning
- ✅ Pre-training vs fine-tuning
- ✅ Instruction tuning
- ✅ Multi-task learning
- ✅ Loss functions (cross-entropy, KL divergence)

### Advanced Topics
- ✅ Distributed training (data, model, pipeline parallelism)
- ✅ Speculative decoding
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Production deployment challenges

---

## 🎓 Interview Preparation

### Must-Know Topics (Absolutely Essential)
1. Transformer architecture and attention mechanism
2. Decoder-only models and causal masking
3. Sampling strategies and temperature
4. KV caching and why it matters
5. Training vs inference differences

### Should-Know Topics (Very Important)
1. Model compression techniques
2. Attention optimizations (MQA, GQA)
3. Distributed training approaches
4. Fine-tuning and instruction tuning
5. Loss functions and training objectives

### Nice-to-Know Topics (Good to Mention)
1. LoRA and parameter-efficient fine-tuning
2. RAG and knowledge augmentation
3. Speculative decoding
4. Multi-task learning
5. Production deployment strategies

### Interview Strategy
- Start with simple explanations
- Provide concrete examples
- Discuss mathematical foundations
- Mention trade-offs
- Reference key papers
- Ask clarifying questions

---

## 💡 Tips for Success

### Study Tips
1. **Understand, Don't Memorize**: Focus on understanding concepts
2. **Use Examples**: Concrete examples are more memorable
3. **Connect Concepts**: See how different topics relate
4. **Practice Coding**: Implement concepts from scratch
5. **Teach Others**: Explaining helps solidify understanding

### Interview Tips
1. **Listen Carefully**: Understand what's being asked
2. **Think Out Loud**: Show your reasoning process
3. **Use Diagrams**: Draw attention mechanisms, architectures
4. **Discuss Trade-offs**: Every technique has pros and cons
5. **Be Honest**: It's okay to say "I don't know"

### Coding Tips
1. **Start Simple**: Implement basic versions first
2. **Test Thoroughly**: Verify your implementations
3. **Optimize Later**: Get it working before optimizing
4. **Document Code**: Clear comments help understanding
5. **Use Existing Libraries**: Don't reinvent the wheel

---

## 📈 Expected Learning Outcomes

After completing these materials, you should be able to:

### Knowledge
- ✅ Explain transformer architecture in detail
- ✅ Understand different model types and their use cases
- ✅ Describe sampling strategies and their trade-offs
- ✅ Explain optimization techniques and their benefits
- ✅ Discuss production deployment challenges

### Skills
- ✅ Implement sampling strategies from scratch
- ✅ Optimize models using KV caching
- ✅ Fine-tune pre-trained models
- ✅ Implement knowledge distillation
- ✅ Debug and optimize LLM code

### Interview Readiness
- ✅ Answer 35+ interview questions confidently
- ✅ Explain concepts clearly and concisely
- ✅ Provide concrete examples and use cases
- ✅ Discuss trade-offs and design decisions
- ✅ Reference relevant papers and research

---

## 🔗 Additional Resources

### Official Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

### Key Papers
- "Attention is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2020)
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

### Online Courses
- Stanford CS224N: NLP with Deep Learning
- Fast.ai Deep Learning Course
- DeepLearning.AI Short Courses

---

## ✅ Checklist for Mastery

- [ ] Read Study Guide completely
- [ ] Understand all core concepts
- [ ] Review all 4 coding guides
- [ ] Run all Jupyter notebooks
- [ ] Answer all 35+ interview questions
- [ ] Implement concepts from scratch
- [ ] Explain concepts to someone else
- [ ] Practice mock interviews
- [ ] Review class transcripts
- [ ] Study official slides

---

## 📞 Quick Reference

### When to Use Each Sampling Method
- **Greedy**: Factual tasks, translation, summarization
- **Top-K**: Balanced creativity and coherence
- **Top-P**: Creative tasks, story generation
- **Beam Search**: High-quality output needed

### When to Use Each Optimization
- **KV Caching**: Always for inference
- **Quantization**: When memory is limited
- **Distillation**: When deploying to edge devices
- **LoRA**: When fine-tuning with limited resources

### Key Formulas
- Attention: softmax(QK^T / √d) @ V
- Temperature: softmax(logits / T)
- KL Divergence: Σ P(x) * log(P(x) / Q(x))
- LoRA: y = Wx + BAx

---

## 🎯 Final Notes

This comprehensive resource package is designed to take you from beginner to expert in Generative AI Part 2 concepts. The materials are structured to support multiple learning styles:

- **Visual Learners**: Diagrams, flowcharts, and visualizations
- **Conceptual Learners**: Detailed explanations and examples
- **Practical Learners**: Coding guides and Jupyter notebooks
- **Interview Learners**: 35+ practice questions with answers

Use this guide as your reference throughout your learning journey and interview preparation. Good luck!

---

**Last Updated:** March 2, 2026
**Difficulty Level:** Intermediate to Advanced
**Estimated Time:** 8-15 hours for complete mastery
**Target Audience:** ML Engineers, SDE transitioning to MLE, AI/ML Researchers
