# NLP_3_ARS.ipynb - Coding Guide

## Overview
This notebook demonstrates an **Automatic Research Article Similarity (ARS)** system that compares research papers using advanced NLP techniques. The system extracts text from PDF files and uses both traditional and modern approaches to measure document similarity.

## Learning Objectives
- Understand PDF text extraction techniques
- Learn about document similarity measurement methods
- Explore the difference between lexical and semantic similarity
- Implement BERT-based embeddings for document comparison
- Handle text preprocessing and chunking for large documents

---

## Section 1: Environment Setup and Library Installation

### Code Block 1: Installing Required Libraries
```python
!pip install tika
!pip install pdfminer.six
!pip install transformers torch
```

**Purpose**: Install essential libraries for PDF processing and NLP tasks.

**Key Libraries Explained**:
- **tika**: Apache Tika Python library for document parsing and text extraction from various file formats
- **pdfminer.six**: Pure Python PDF parser that can extract text, images, and metadata from PDF files
- **transformers**: Hugging Face library providing pre-trained transformer models like BERT, GPT, etc.
- **torch**: PyTorch deep learning framework required for running transformer models

**Why These Libraries?**:
- **tika** vs **pdfminer.six**: Both can extract PDF text, but pdfminer.six is more lightweight and doesn't require Java
- **transformers**: Provides easy access to state-of-the-art pre-trained language models
- **torch**: Backend for running neural network computations efficiently

---

## Section 2: PDF Text Extraction

### Code Block 2: Extracting Text from PDF Files
```python
from pdfminer.high_level import extract_text

# Extract text from a PDF file
text0 = extract_text("/content/291.pdf")
text1 = extract_text("/content/292.pdf")

texts = [text0, text1]
```

**Purpose**: Extract raw text content from research paper PDFs for analysis.

**Function Breakdown**:
- **`extract_text()`**: High-level function from pdfminer that handles the complex PDF parsing internally
- **Parameters**: Takes file path as input and returns extracted text as string
- **Return Value**: Complete text content of the PDF as a single string

**Key Concepts**:
- **PDF Structure**: PDFs contain complex formatting, fonts, and layout information
- **Text Extraction Challenges**: Handling multi-column layouts, headers, footers, and special characters
- **Data Storage**: Storing extracted texts in a list for batch processing

**Best Practices**:
- Always handle potential file path errors
- Consider memory usage for large PDF files
- Validate extracted text quality before processing

---

## Section 3: Document Similarity Analysis Setup

### Code Block 3: Installing and Importing Similarity Libraries
```python
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
```

**Purpose**: Import libraries for both traditional (TF-IDF) and modern (BERT) similarity analysis.

**Library Functions**:
- **AutoTokenizer**: Automatically loads the appropriate tokenizer for a given model
- **AutoModel**: Loads pre-trained transformer models without task-specific heads
- **cosine_distances**: Calculates cosine distance (1 - cosine similarity) between vectors
- **TfidfVectorizer**: Converts text to TF-IDF (Term Frequency-Inverse Document Frequency) vectors

---

## Section 4: BERT-Based Semantic Similarity

### Code Block 4: Loading SciBERT Model
```python
# Load SciBERT
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
```

**Purpose**: Load a specialized BERT model trained on scientific literature.

**Model Choice Explanation**:
- **SciBERT**: BERT model specifically trained on scientific papers
- **"allenai/scibert_scivocab_uncased"**: Model identifier on Hugging Face Hub
- **Uncased**: Model treats uppercase and lowercase letters the same
- **Scientific Vocabulary**: Contains domain-specific terms common in research papers

**Why SciBERT over Regular BERT?**:
- Better understanding of scientific terminology
- Trained on scientific corpus (papers from Semantic Scholar)
- More accurate embeddings for research document comparison

### Code Block 5: Basic Text Embedding Generation
```python
embeddings = []
for each_text in texts:
  # Tokenize and get embeddings
  inputs = tokenizer(each_text[:500], return_tensors='pt', truncation=True, padding=True)
  with torch.no_grad():
      outputs = model(**inputs)

  # Mean pooling for sentence embedding
  token_embeddings = outputs.last_hidden_state
  attention_mask = inputs['attention_mask']
  mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
  sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
  sentence_embedding = sum_embeddings / sum_mask
  embeddings.append(sentence_embedding)
```

**Purpose**: Convert text documents into dense vector representations using BERT.

**Step-by-Step Breakdown**:

1. **Text Truncation**: `each_text[:500]` - Limits input to first 500 characters
   - **Why?**: BERT has token limits (usually 512 tokens)
   - **Trade-off**: May lose information from longer documents

2. **Tokenization**: `tokenizer(text, return_tensors='pt', truncation=True, padding=True)`
   - **return_tensors='pt'**: Returns PyTorch tensors
   - **truncation=True**: Automatically cuts text if too long
   - **padding=True**: Adds padding tokens to make all sequences same length

3. **Model Inference**: `model(**inputs)`
   - **torch.no_grad()**: Disables gradient computation for faster inference
   - **outputs.last_hidden_state**: Gets final layer embeddings for all tokens

4. **Mean Pooling**: Converts token-level embeddings to document-level embedding
   - **Purpose**: BERT outputs one vector per token; we need one vector per document
   - **Attention Mask**: Ensures padding tokens don't affect the average
   - **mask_expanded**: Broadcasts attention mask to match embedding dimensions
   - **sum_embeddings**: Sums all token embeddings (excluding padding)
   - **sum_mask**: Counts non-padding tokens
   - **Final Result**: Average of all meaningful token embeddings

**Key Concepts**:
- **Dense Embeddings**: Each document becomes a 768-dimensional vector
- **Contextual Understanding**: BERT considers word context, not just individual words
- **Semantic Similarity**: Similar documents will have similar embedding vectors

---

## Section 6: Similarity Calculation

### Code Block 6: Computing Cosine Similarity
```python
from sklearn.metrics.pairwise import cosine_distances
1 - cosine_distances(embeddings[0],embeddings[1])[0] # Cosine dissimilarity(distance) -->  [1 - cosine similarity]
```

**Purpose**: Calculate how similar the two research papers are semantically.

**Mathematical Explanation**:
- **Cosine Distance**: Measures angle between two vectors (0 = identical, 1 = orthogonal)
- **Cosine Similarity**: 1 - cosine_distance (1 = identical, 0 = orthogonal)
- **Range**: Similarity values between 0 (completely different) and 1 (identical)

**Interpretation**:
- **High Similarity (>0.8)**: Papers likely discuss very similar topics
- **Medium Similarity (0.5-0.8)**: Papers share some common themes
- **Low Similarity (<0.5)**: Papers discuss different topics

---

## Section 7: Traditional TF-IDF Approach

### Code Block 7: TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
tv.fit_transform(texts).toarray().shape
```

**Purpose**: Compare BERT embeddings with traditional bag-of-words approach.

**TF-IDF Explanation**:
- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: How rare a word is across all documents
- **TF-IDF Score**: TF × IDF - gives higher weight to important, distinctive words

**Output Shape Analysis**:
- **(2, 2511)**: 2 documents, 2511 unique words in vocabulary
- **Sparse Representation**: Most values are 0 (words don't appear in most documents)
- **High Dimensionality**: Much higher than BERT's 768 dimensions

**BERT vs TF-IDF Comparison**:
- **BERT**: Dense 768-dimensional vectors, captures semantic meaning
- **TF-IDF**: Sparse 2511-dimensional vectors, captures word frequency patterns
- **Context**: BERT understands context; TF-IDF treats words independently

---

## Section 8: Robust Text Processing Pipeline

### Code Block 8: Production-Ready Text Processing
```python
# Code with chunking
from transformers import AutoTokenizer, AutoModel
import torch

# Load SciBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embeddings = []

for i, text in enumerate(texts):
    try:
        # Tokenize safely with truncation to 512 tokens
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,        # Ensures no sequence exceeds model limit
            max_length=512,         # BERT models have a hard cap at 512
            padding="max_length"    # Ensures uniform input shape
        )

        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Disable gradient computation (inference mode)
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling over the token embeddings (excluding padding tokens)
        attention_mask = inputs['attention_mask']
        last_hidden_state = outputs.last_hidden_state

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        chunk_embedding = sum_embeddings / sum_mask

        embeddings.append(chunk_embedding.cpu())
        print(f"Processed chunk {i+1}/{len(texts)} successfully.")

    except Exception as e:
        print(f"⚠️ Skipping chunk {i+1} due to error: {e}")
        continue

# Stack all embeddings
embeddings = torch.cat(embeddings, dim=0)
print("✅ Shape of all embeddings:", embeddings.shape)
```

**Purpose**: Create a robust, production-ready pipeline for processing multiple documents.

**Key Improvements**:

1. **GPU Utilization**:
   - **`torch.device()`**: Automatically detects and uses GPU if available
   - **`.to(device)`**: Moves model and tensors to appropriate device
   - **Performance**: GPU processing is much faster for neural networks

2. **Proper Token Limits**:
   - **`max_length=512`**: Explicit token limit (BERT's maximum)
   - **`padding="max_length"`**: Ensures consistent tensor shapes
   - **Memory Efficiency**: Prevents out-of-memory errors

3. **Error Handling**:
   - **try-except blocks**: Gracefully handles processing errors
   - **Continue processing**: Doesn't stop entire pipeline if one document fails
   - **Progress tracking**: Shows processing status

4. **Memory Management**:
   - **`.cpu()`**: Moves results back to CPU to free GPU memory
   - **`torch.cat()`**: Efficiently combines all embeddings
   - **Batch processing**: Processes documents one at a time to manage memory

5. **Tensor Operations**:
   - **`dim=1`**: Specifies dimension for sum operations
   - **`unsqueeze(-1)`**: Adds dimension for broadcasting
   - **`torch.clamp()`**: Prevents division by zero

**Production Considerations**:
- **Scalability**: Can handle large document collections
- **Reliability**: Robust error handling and recovery
- **Performance**: Optimized for GPU usage and memory efficiency
- **Monitoring**: Progress tracking and error reporting

---

## Key Concepts Summary

### 1. Document Similarity Methods
- **Lexical Similarity**: Based on word overlap (TF-IDF, Jaccard)
- **Semantic Similarity**: Based on meaning (BERT embeddings)
- **Dense vs Sparse**: BERT creates dense vectors; TF-IDF creates sparse vectors

### 2. BERT Embeddings
- **Contextual**: Word meaning depends on surrounding context
- **Pre-trained**: Leverages knowledge from large text corpora
- **Transfer Learning**: Applies general language understanding to specific tasks

### 3. Text Processing Pipeline
- **Tokenization**: Converting text to model-readable format
- **Truncation**: Handling text longer than model limits
- **Padding**: Making all inputs the same length
- **Pooling**: Converting token-level to document-level representations

### 4. Similarity Metrics
- **Cosine Similarity**: Measures angle between vectors
- **Range**: 0 (different) to 1 (identical)
- **Interpretation**: Higher values indicate more similar documents

---

## Practical Applications

1. **Academic Research**: Finding related papers and avoiding duplication
2. **Literature Review**: Automatically grouping similar studies
3. **Plagiarism Detection**: Identifying potentially copied content
4. **Recommendation Systems**: Suggesting relevant documents to users
5. **Document Clustering**: Organizing large document collections

---

## Best Practices

1. **Model Selection**: Choose domain-specific models (SciBERT for scientific texts)
2. **Text Preprocessing**: Clean and normalize text before processing
3. **Chunking Strategy**: Handle long documents by splitting into manageable pieces
4. **Similarity Thresholds**: Establish meaningful similarity cutoffs for your use case
5. **Performance Optimization**: Use GPU acceleration and batch processing for large datasets

---

## Common Pitfalls to Avoid

1. **Token Limit Exceeded**: Always truncate or chunk long texts
2. **Memory Issues**: Process large datasets in batches
3. **Model Mismatch**: Use the same tokenizer as the model
4. **Similarity Interpretation**: Understand that similarity scores are relative, not absolute
5. **Context Loss**: Be careful when truncating important information

This coding guide provides a comprehensive understanding of document similarity analysis using modern NLP techniques, preparing you for real-world text analysis applications.