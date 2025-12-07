# vector_db.ipynb - Comprehensive Coding Guide

## 📋 Overview
This notebook demonstrates **Vector Databases** and **Semantic Search** using ChromaDB with LangChain. You'll learn how to store documents as vector embeddings and perform similarity searches to find relevant content.

**Target Audience**: Python programmers new to vector databases and semantic search

**Key Concepts**:
- Vector embeddings for text
- Document storage in vector databases
- Similarity search
- ChromaDB integration with LangChain

---

## 🔧 Cell 1: Installing Required Libraries

```python
!pip install langchain langchain-chroma langchain-openai
```

### What's Being Installed?

#### 1. `langchain`
**Purpose**: Core framework for building LLM applications

**What it provides**:
- Document handling abstractions
- Chain composition tools
- Integration with various AI services
- Utilities for text processing

**Why we need it**: Provides the `Document` class and orchestration tools

#### 2. `langchain-chroma`
**Purpose**: ChromaDB integration for LangChain

**What is ChromaDB?**
- Open-source vector database
- Stores high-dimensional vectors (embeddings)
- Optimized for similarity search
- Lightweight and easy to use

**Key Features**:
- In-memory and persistent storage
- Fast similarity search algorithms
- Metadata filtering
- Simple API

**Why we need it**: Stores and searches document embeddings efficiently

#### 3. `langchain-openai`
**Purpose**: OpenAI integration for LangChain

**What it provides**:
- Access to OpenAI's embedding models
- Text-to-vector conversion
- API handling and authentication

**Embedding Model Used**: `text-embedding-ada-002`
- **Dimensions**: 1536
- **Max Input**: 8,191 tokens
- **Quality**: High-quality semantic representations
- **Cost**: ~$0.0001 per 1K tokens

**Why we need it**: Converts text into vector embeddings

---

## 🔑 Cell 2: Environment Configuration

```python
import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
os.environ["OPENAI_API_KEY"] = getpass.getpass()
```

### Import Breakdown

#### `getpass`
**Purpose**: Securely input sensitive information

**Why use it?**
- Hides input while typing (shows `····`)
- Prevents API keys from appearing in code
- Better security than hardcoding keys

**Usage**: `getpass.getpass()` prompts for password/key input

#### `os`
**Purpose**: Interact with operating system

**What we use it for**: Setting environment variables

### Environment Variables

#### 1. `LANGCHAIN_TRACING_V2`
**Value**: `"true"`

**Purpose**: Enable LangSmith tracing

**What is LangSmith?**
- Debugging and monitoring platform for LangChain
- Tracks all LangChain operations
- Logs inputs, outputs, and performance

**Benefits**:
- Debug vector operations
- Monitor embedding generation
- Track search performance
- Analyze costs

**Note**: Optional - works without LangSmith account

#### 2. `OPENAI_API_KEY`
**Value**: Your OpenAI API key (entered securely)

**Purpose**: Authenticate with OpenAI API

**Required**: Yes - needed for generating embeddings

**How to get it**:
1. Go to platform.openai.com
2. Create account / Sign in
3. Navigate to API keys section
4. Create new secret key
5. Copy and paste when prompted

**Security Best Practice**: Never hardcode API keys in code!

---

## 📄 Cell 3: Creating Sample Documents

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="LangChain helps building LLM-based applications",
        metadata={"source": "lang-chain-doc"},
    ),
    Document(
        page_content="LLMs have changed the scene in AI and ML",
        metadata={"source": "llm-doc"},
    ),
]
```

### Understanding the Document Class

#### What is a `Document`?
**Definition**: LangChain's standard container for text and metadata

**Structure**:
```python
Document(
    page_content="The actual text content",
    metadata={"key": "value"}
)
```

**Why use it?**
- Standardized format across LangChain
- Combines content with context
- Enables metadata filtering
- Tracks document sources

### Document Components

#### 1. `page_content` (Required)
**Type**: String

**Purpose**: The actual text to be embedded and searched

**Examples in our code**:
- `"Dogs are great companions..."`
- `"Cats are independent pets..."`
- `"LangChain helps building..."`

**What happens to it?**
1. Sent to OpenAI embedding model
2. Converted to 1536-dimensional vector
3. Stored in ChromaDB
4. Used for similarity comparisons

#### 2. `metadata` (Optional)
**Type**: Dictionary

**Purpose**: Additional information about the document

**Common use cases**:
- **Source tracking**: Where did this come from?
- **Categorization**: What type of document?
- **Timestamps**: When was it created?
- **Authors**: Who wrote it?
- **Tags**: What topics does it cover?

**Examples in our code**:
```python
{"source": "mammal-pets-doc"}  # Category: mammal pets
{"source": "fish-pets-doc"}     # Category: fish pets
{"source": "bird-pets-doc"}     # Category: bird pets
{"source": "lang-chain-doc"}    # Category: LangChain info
{"source": "llm-doc"}           # Category: LLM info
```

**Why metadata matters**:
- **Filtering**: Search only mammal documents
- **Traceability**: Know where information came from
- **Organization**: Group related documents
- **Debugging**: Track which documents are retrieved

### Our Document Collection

**Total Documents**: 7

**Categories**:
1. **Mammal Pets** (3 docs): Dogs, Cats, Rabbits
2. **Fish Pets** (1 doc): Goldfish
3. **Bird Pets** (1 doc): Parrots
4. **Tech** (2 docs): LangChain, LLMs

**Purpose of variety**: Demonstrates how vector search finds semantically similar content

---

## 🗄️ Cell 6: Creating the Vector Store

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)
```

### Import Breakdown

#### `Chroma`
**From**: `langchain_chroma`

**What it is**: Vector database implementation

**What it does**:
- Stores vector embeddings
- Performs similarity searches
- Manages document metadata
- Handles indexing

#### `OpenAIEmbeddings`
**From**: `langchain_openai`

**What it is**: Embedding model wrapper

**What it does**:
- Connects to OpenAI API
- Converts text to vectors
- Handles batching and rate limits
- Manages authentication

### The `from_documents()` Method

#### Signature
```python
Chroma.from_documents(
    documents,           # List of Document objects
    embedding,           # Embedding model instance
    collection_name,     # Optional: name for this collection
    persist_directory    # Optional: save to disk
)
```

#### What Happens Step-by-Step

**Step 1: Extract Text**
```python
texts = [
    "Dogs are great companions...",
    "Cats are independent pets...",
    "Goldfish are popular pets...",
    # ... etc
]
```

**Step 2: Generate Embeddings**
```python
# For each text, call OpenAI API
embedding_1 = OpenAI.embed("Dogs are great companions...")
# Returns: [0.123, -0.456, 0.789, ..., 0.321]  # 1536 numbers

embedding_2 = OpenAI.embed("Cats are independent pets...")
# Returns: [0.234, -0.345, 0.678, ..., 0.432]  # 1536 numbers

# ... for all 7 documents
```

**Step 3: Store in ChromaDB**
```python
# ChromaDB stores:
{
    "id": "doc_1",
    "embedding": [0.123, -0.456, ..., 0.321],
    "text": "Dogs are great companions...",
    "metadata": {"source": "mammal-pets-doc"}
}
# ... for all documents
```

**Step 4: Create Index**
- ChromaDB builds search index
- Optimizes for fast similarity lookups
- Uses HNSW algorithm (Hierarchical Navigable Small World)

**Step 5: Return Vector Store Object**
```python
vectorstore = <Chroma object>
# Ready to perform searches!
```

### What is a Vector Embedding?

**Simple Explanation**: A list of numbers that represents the meaning of text

**Example**:
```python
"cat" → [0.2, 0.8, 0.1, 0.9, -0.3, ..., 0.5]  # 1536 numbers
"dog" → [0.3, 0.7, 0.2, 0.8, -0.2, ..., 0.6]  # Similar to cat!
"car" → [0.9, 0.1, 0.8, 0.2, -0.9, ..., 0.1]  # Very different!
```

**Key Property**: Similar meanings → Similar vectors

**How similarity is measured**:
- **Cosine Similarity**: Angle between vectors
- Range: -1 to 1 (1 = identical, 0 = unrelated, -1 = opposite)
- Formula: `cos(θ) = (A·B) / (||A|| ||B||)`

### Memory vs Persistent Storage

**Current Code** (In-Memory):
```python
vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
```
- Stored in RAM
- Lost when program ends
- Fast
- Good for testing

**Persistent Storage**:
```python
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"  # Save to disk
)
```
- Stored on disk
- Survives program restarts
- Slightly slower
- Good for production

---

## 🔍 Cell 7: Basic Similarity Search

```python
results = vectorstore.similarity_search("cat", 2)
for doc in results:
    print(doc.page_content)
    print("=================")
```

### Method: `similarity_search()`

#### Signature
```python
vectorstore.similarity_search(
    query,              # String: search query
    k=4,               # Int: number of results (default: 4)
    filter=None        # Dict: metadata filter (optional)
)
```

#### Parameters Explained

**1. `query` (Required)**
- **Type**: String
- **Value**: `"cat"`
- **Purpose**: What you're searching for

**2. `k` (Optional)**
- **Type**: Integer
- **Value**: `2`
- **Purpose**: How many results to return
- **Default**: 4

**3. `filter` (Optional)**
- **Type**: Dictionary
- **Example**: `{"source": "mammal-pets-doc"}`
- **Purpose**: Search only specific documents

### What Happens Under the Hood

**Step 1: Embed the Query**
```python
query = "cat"
query_embedding = OpenAI.embed("cat")
# Returns: [0.25, 0.75, 0.15, ..., 0.55]  # 1536 numbers
```

**Step 2: Calculate Similarities**
```python
# Compare query embedding with all stored embeddings
similarity_1 = cosine_similarity(query_embedding, embedding_1)  # Dogs: 0.72
similarity_2 = cosine_similarity(query_embedding, embedding_2)  # Cats: 0.95
similarity_3 = cosine_similarity(query_embedding, embedding_3)  # Goldfish: 0.45
similarity_4 = cosine_similarity(query_embedding, embedding_4)  # Parrots: 0.38
similarity_5 = cosine_similarity(query_embedding, embedding_5)  # Rabbits: 0.68
similarity_6 = cosine_similarity(query_embedding, embedding_6)  # LangChain: 0.12
similarity_7 = cosine_similarity(query_embedding, embedding_7)  # LLMs: 0.10
```

**Step 3: Sort by Similarity**
```python
sorted_results = [
    (0.95, "Cats are independent pets..."),      # Highest!
    (0.72, "Dogs are great companions..."),
    (0.68, "Rabbits are social animals..."),
    (0.45, "Goldfish are popular pets..."),
    (0.38, "Parrots are intelligent birds..."),
    (0.12, "LangChain helps building..."),
    (0.10, "LLMs have changed the scene...")
]
```

**Step 4: Return Top K**
```python
# k=2, so return top 2
results = [
    Document(page_content="Cats are independent pets...", metadata={...}),
    Document(page_content="Dogs are great companions...", metadata={...})
]
```

### Expected Output

```
Cats are independent pets that often enjoy their own space.
=================
Dogs are great companions, known for their loyalty and friendliness.
=================
```

### Why These Results?

**1. Cats Document** (Highest similarity)
- Direct match: query is "cat"
- Semantic relevance: about cats
- Score: ~0.95

**2. Dogs Document** (Second highest)
- Semantic similarity: both are pets
- Both are mammals
- Both are companions
- Score: ~0.72

**Not Returned**:
- Rabbits: Similar but lower score
- Goldfish: Different type of pet
- Parrots: Different type of pet
- LangChain/LLMs: Completely different topic

---

## 📊 Cell 8: Similarity Search with Scores

```python
results = vectorstore.similarity_search_with_relevance_scores("cat")
for doc, score in results:
    print(score)
    print(doc.page_content)
    print("=================")
```

### Method: `similarity_search_with_relevance_scores()`

#### Signature
```python
vectorstore.similarity_search_with_relevance_scores(
    query,              # String: search query
    k=4,               # Int: number of results (default: 4)
    filter=None        # Dict: metadata filter (optional)
)
```

#### What's Different from `similarity_search()`?

**Returns**: List of tuples `(Document, score)` instead of just `Document`

**Example**:
```python
# similarity_search() returns:
[Document(...), Document(...)]

# similarity_search_with_relevance_scores() returns:
[(Document(...), 0.95), (Document(...), 0.72)]
```

### Understanding Relevance Scores

#### Score Range
- **0.0 to 1.0** (normalized)
- **1.0**: Perfect match (identical)
- **0.8-0.9**: Very high similarity
- **0.6-0.7**: Good similarity
- **0.4-0.5**: Moderate similarity
- **< 0.4**: Low similarity

#### What the Score Means

**High Score (0.8+)**:
- Semantically very similar
- Likely relevant to query
- High confidence

**Medium Score (0.5-0.7)**:
- Some semantic overlap
- Possibly relevant
- Medium confidence

**Low Score (< 0.5)**:
- Weak semantic connection
- Likely not relevant
- Low confidence

### Expected Output

```
0.9523847
Cats are independent pets that often enjoy their own space.
=================
0.7234561
Dogs are great companions, known for their loyalty and friendliness.
=================
0.6812345
Rabbits are social animals that need plenty of space to hop around.
=================
0.4523456
Goldfish are popular pets for beginners, requiring relatively simple care.
=================
```

**Note**: Actual scores will vary slightly based on OpenAI's embedding model

### Why Use Scores?

#### 1. Quality Control
```python
# Filter out low-confidence results
good_results = [(doc, score) for doc, score in results if score > 0.7]
```

#### 2. Confidence Thresholds
```python
if score > 0.8:
    print("High confidence match")
elif score > 0.6:
    print("Medium confidence match")
else:
    print("Low confidence - may not be relevant")
```

#### 3. Debugging
```python
# Understand why certain results were returned
for doc, score in results:
    print(f"Score: {score:.2f} - {doc.page_content[:50]}...")
```

#### 4. User Feedback
```python
# Show confidence to users
print(f"Found with {score*100:.1f}% confidence")
```

---

## 🎯 Key Concepts Summary

### 1. Vector Embeddings
**What**: Numerical representations of text meaning

**How**: Text → OpenAI API → 1536 numbers

**Why**: Enable semantic search (meaning-based, not keyword-based)

### 2. Similarity Search
**What**: Find documents similar to a query

**How**: Compare vector embeddings using cosine similarity

**Why**: Better than keyword search - understands meaning

### 3. ChromaDB
**What**: Vector database for storing embeddings

**How**: Stores vectors + metadata, optimized for similarity search

**Why**: Fast, efficient, easy to use

### 4. Metadata
**What**: Additional information about documents

**How**: Dictionary attached to each document

**Why**: Filtering, organization, traceability

---

## 💡 Common Use Cases

### 1. Document Search
```python
# Find relevant documents for a question
query = "What pets are good for beginners?"
results = vectorstore.similarity_search(query, k=3)
```

### 2. Recommendation System
```python
# Find similar items
item = "Dogs are loyal companions"
similar = vectorstore.similarity_search(item, k=5)
```

### 3. Question Answering (RAG)
```python
# Retrieve context for LLM
question = "Tell me about independent pets"
context_docs = vectorstore.similarity_search(question, k=2)
context = "\n".join([doc.page_content for doc in context_docs])
# Send context + question to LLM
```

### 4. Content Classification
```python
# Find category by similarity
content = "Small furry pet that hops"
results = vectorstore.similarity_search_with_relevance_scores(content, k=1)
category = results[0][0].metadata["source"]
```

---

## 🔧 Advanced Features

### Metadata Filtering

```python
# Search only mammal documents
results = vectorstore.similarity_search(
    "friendly pet",
    k=3,
    filter={"source": "mammal-pets-doc"}
)
```

### Persistent Storage

```python
# Save to disk
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="./my_vector_db"
)

# Load later
vectorstore = Chroma(
    persist_directory="./my_vector_db",
    embedding_function=OpenAIEmbeddings()
)
```

### Adding Documents Later

```python
# Add new documents to existing store
new_docs = [
    Document(
        page_content="Hamsters are small, easy-to-care-for pets.",
        metadata={"source": "mammal-pets-doc"}
    )
]
vectorstore.add_documents(new_docs)
```

### Deleting Documents

```python
# Delete by ID
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])
```

---

## 🐛 Troubleshooting

### Issue: "OpenAI API key not found"
**Solution**: Make sure you ran Cell 2 and entered your API key

### Issue: Low similarity scores
**Possible causes**:
- Query too vague
- Documents don't match query topic
- Need more documents

**Solution**: Try more specific queries or add more relevant documents

### Issue: Unexpected results
**Debug approach**:
```python
# Check scores to understand ranking
results = vectorstore.similarity_search_with_relevance_scores(query)
for doc, score in results:
    print(f"{score:.3f}: {doc.page_content[:50]}")
```

### Issue: Slow performance
**Solutions**:
- Use smaller embedding models
- Reduce number of documents
- Use persistent storage
- Implement caching

---

## 📚 Key Takeaways

1. **Vector databases enable semantic search** - Find by meaning, not just keywords

2. **Embeddings capture meaning** - Text → Numbers that represent semantics

3. **ChromaDB is simple and powerful** - Easy to use, fast similarity search

4. **Metadata adds context** - Filter, organize, and track documents

5. **Scores indicate confidence** - Use them for quality control

6. **LangChain simplifies integration** - Standard interfaces for documents and vector stores

---

## 🚀 Next Steps

1. **Try different queries** - Experiment with various search terms

2. **Add your own documents** - Use real data from your domain

3. **Implement filtering** - Use metadata to narrow searches

4. **Build a RAG system** - Combine with LLMs for question answering

5. **Explore other vector databases** - Try Pinecone, Weaviate, or FAISS

6. **Optimize performance** - Experiment with chunk sizes and embedding models

---

## 🎓 Additional Resources

**Vector Databases**:
- ChromaDB Documentation: https://docs.trychroma.com/
- Vector Database Comparison: https://www.pinecone.io/learn/vector-database/

**Embeddings**:
- OpenAI Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
- Understanding Embeddings: https://www.deeplearning.ai/short-courses/

**LangChain**:
- LangChain Documentation: https://python.langchain.com/
- Vector Stores Guide: https://python.langchain.com/docs/modules/data_connection/vectorstores/

---

*This guide covers all major concepts in vector_db.ipynb. Practice by experimenting with different documents and queries to build intuition for semantic search!*
