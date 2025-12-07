# LangChain.ipynb - Comprehensive Coding Guide

## 📋 Overview
This notebook introduces **LangChain**, a framework for building applications with Large Language Models (LLMs). It covers basic LLM calls, prompt templating, chaining components, and implementing Retrieval Augmented Generation (RAG).

**Target Audience**: Python programmers new to LangChain and LLM application development

**Key Concepts Covered**:
- LLM initialization and basic calls
- Prompt templates and message formatting
- Chain composition using the pipe operator
- Output parsing
- Document loading and text splitting
- Vector embeddings and similarity search
- RAG (Retrieval Augmented Generation)

---

## 🔧 Setup and Installation

### Cell 1: Installing Required Libraries

```python
!pip install langchain langchain-chroma langchain-openai
!pip install beautifulsoup4
!pip install langchain-community
!pip install faiss-cpu
```

**What's Being Installed:**

1. **`langchain`**: Core LangChain library
   - Framework for building LLM applications
   - Provides abstractions for chains, prompts, and agents

2. **`langchain-chroma`**: ChromaDB integration
   - Vector database for storing embeddings
   - Enables semantic search capabilities

3. **`langchain-openai`**: OpenAI integration
   - Wrapper for OpenAI's GPT models
   - Handles API calls and response formatting

4. **`beautifulsoup4`**: HTML parsing library
   - Used by WebBaseLoader to extract text from web pages
   - Cleans and structures HTML content

5. **`langchain-community`**: Community-contributed integrations
   - Additional loaders, vector stores, and utilities
   - Includes WebBaseLoader for loading web content

6. **`faiss-cpu`**: Facebook AI Similarity Search
   - Efficient vector similarity search library
   - CPU-optimized version (no GPU required)
   - Used for finding similar documents based on embeddings

**Why These Libraries?**
- LangChain provides high-level abstractions for common LLM patterns
- Vector databases enable semantic search (finding similar content)
- FAISS is fast and memory-efficient for similarity search

---

### Cell 2: Environment Configuration

```python
import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
os.environ["OPENAI_API_KEY"] = getpass.getpass()
```

**Import Breakdown:**

1. **`getpass`**: Secure password input
   - Hides input when typing (doesn't show on screen)
   - Best practice for entering API keys

2. **`os`**: Operating system interface
   - Used to set environment variables
   - Environment variables are accessible throughout the program

**Environment Variables:**

1. **`LANGCHAIN_TRACING_V2`**: Enables LangSmith tracing
   - **Purpose**: Debug and monitor LLM calls
   - **Value**: "true" activates tracing
   - **What it does**: Logs all LLM interactions to LangSmith platform
   - **Use case**: Track costs, latency, and debug issues

2. **`LANGCHAIN_API_KEY`** (commented out): LangSmith API key
   - Required if you want to use LangSmith tracing
   - Commented out in this notebook

3. **`OPENAI_API_KEY`**: OpenAI API authentication
   - **Required**: Yes, for making API calls
   - **Security**: Never hardcode keys in code!
   - **getpass.getpass()**: Prompts user to enter key securely

**Best Practices:**
- Always use environment variables for API keys
- Never commit API keys to version control
- Use `.env` files with `python-dotenv` in production

---

## 🤖 Simple LLM Call

### Cell 4: Initialize LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

**What's Happening:**

1. **`ChatOpenAI`**: LangChain wrapper for OpenAI's chat models
   - Default model: `gpt-3.5-turbo` (can be changed)
   - Handles API communication
   - Formats requests and responses

2. **`api_key` parameter**: Authentication
   - Retrieves key from environment variable
   - Passed to OpenAI API for authentication

**Additional Parameters (not shown but available):**
```python
llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4",              # Specify model
    temperature=0.7,            # Creativity (0-2)
    max_tokens=500,             # Max response length
    timeout=30,                 # Request timeout
    max_retries=2               # Retry failed requests
)
```

**Key Concepts:**
- **Temperature**: Controls randomness
  - 0 = deterministic (same output each time)
  - 1 = balanced
  - 2 = very creative/random
  
- **Max Tokens**: Limits response length
  - 1 token ≈ 4 characters
  - Helps control costs

---

### Cell 5-6: Basic LLM Invocation

```python
answer = llm.invoke("how can langsmith help with testing?")
print(answer)
```

**Method: `invoke()`**
- **Purpose**: Send a message to the LLM and get a response
- **Input**: String (user message)
- **Output**: `AIMessage` object containing response

**Response Structure:**
```python
AIMessage(
    content="LangSmith can help with testing by...",  # The actual response
    response_metadata={
        'token_usage': {...},      # Tokens used
        'model_name': 'gpt-3.5-turbo',
        'finish_reason': 'stop'    # Why generation stopped
    }
)
```

**Why This Isn't Interesting:**
- No context or conversation history
- No structured prompts
- No chaining with other components
- Just a simple question-answer

---

## 🔗 Prompt Templating and Chaining

### Cell 10: Creating Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

prompt_value = template.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?"
    }
)
```

**Key Components:**

1. **`ChatPromptTemplate`**: Template for chat conversations
   - Structures multi-turn conversations
   - Supports variable substitution
   - Maintains message roles (system, human, ai)

2. **Message Roles:**
   - **`system`**: Instructions for the AI's behavior
     - Sets personality, constraints, and guidelines
     - Example: "You are a helpful assistant"
   
   - **`human`**: User messages
     - Questions or inputs from the user
     - Can contain variables like `{user_input}`
   
   - **`ai`**: AI responses (for few-shot examples)
     - Shows the AI how to respond
     - Provides context and examples

3. **Variable Substitution:**
   - **`{name}`**: Replaced with "Bob"
   - **`{user_input}`**: Replaced with "What is your name?"
   - Uses Python string formatting syntax

**Why Use Templates?**
- **Reusability**: Define once, use many times
- **Consistency**: Same structure for all requests
- **Maintainability**: Easy to update prompts
- **Few-shot learning**: Include examples in the template

**Output Structure:**
```python
ChatPromptValue(
   messages=[
       SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
       HumanMessage(content='Hello, how are you doing?'),
       AIMessage(content="I'm doing well, thanks!"),
       HumanMessage(content='What is your name?')
   ]
)
```

---

### Cell 11: Inspecting Messages

```python
for msg in prompt_value.messages:
  print(type(msg).__name__, ":", msg.content)
```

**What This Does:**
- Iterates through all messages in the prompt
- Prints message type and content
- Useful for debugging and understanding prompt structure

**Output:**
```
SystemMessage : You are a helpful AI bot. Your name is Bob.
HumanMessage : Hello, how are you doing?
AIMessage : I'm doing well, thanks!
HumanMessage : What is your name?
```

---

### Cell 12-13: Simpler Template

```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])
```

**Simplified Structure:**
- Only system message and user input
- No conversation history
- Single variable: `{input}`

**Use Case:**
- Single-turn interactions
- Task-specific prompts
- When conversation history isn't needed

---

### Cell 15-18: Creating and Using Chains

```python
chain = prompt | llm

chain_result = chain.invoke({"input": "how can langsmith help with testing?"})
print(chain_result.content)
```

**The Pipe Operator (`|`):**
- **Purpose**: Compose components into a chain
- **Syntax**: `component1 | component2 | component3`
- **Flow**: Output of component1 → Input of component2

**Chain Execution:**
1. `prompt` receives `{"input": "..."}`
2. `prompt` formats the message
3. Formatted message → `llm`
4. `llm` generates response
5. Response returned as `AIMessage`

**Why Chaining?**
- **Modularity**: Each component has one job
- **Reusability**: Mix and match components
- **Readability**: Clear data flow
- **Extensibility**: Easy to add more steps

**Accessing Response:**
- **`.content`**: The actual text response
- **`.response_metadata`**: Token usage, model info, etc.

---

### Cell 23-26: Adding Output Parser

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

chain_result = chain.invoke({"input": "how can langsmith help with testing?"})
print(chain_result)  # Now it's a string, not AIMessage
```

**`StrOutputParser`:**
- **Purpose**: Extract string content from AIMessage
- **Input**: `AIMessage` object
- **Output**: Plain string

**Before Parser:**
```python
AIMessage(content="LangSmith helps with...", response_metadata={...})
```

**After Parser:**
```python
"LangSmith helps with..."  # Just the string
```

**Why Use Parsers?**
- Simplify output handling
- Extract specific fields
- Convert to desired format (JSON, list, etc.)
- Chain with other components that expect strings

**Other Parsers Available:**
- `JsonOutputParser`: Parse JSON responses
- `PydanticOutputParser`: Parse into Pydantic models
- `ListOutputParser`: Parse comma-separated lists
- `DatetimeOutputParser`: Parse dates and times

---

## 📚 Retrieval Augmented Generation (RAG)

### Cell 28: Loading Documents

```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()
```

**`WebBaseLoader`:**
- **Purpose**: Load content from web pages
- **Input**: URL(s)
- **Output**: List of `Document` objects
- **Under the hood**: Uses BeautifulSoup to parse HTML

**Document Structure:**
```python
Document(
    page_content="The actual text content...",
    metadata={
        'source': 'https://docs.smith.langchain.com/user_guide',
        'title': 'User Guide',
        ...
    }
)
```

**Why Load Documents?**
- Provide context to the LLM
- Ground responses in factual information
- Reduce hallucinations
- Enable question-answering over specific content

**Other Loaders Available:**
- `PyPDFLoader`: Load PDF files
- `TextLoader`: Load text files
- `CSVLoader`: Load CSV files
- `DirectoryLoader`: Load all files in a directory
- `GitHubLoader`: Load from GitHub repositories

---

### Cell 29: Creating Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

**What are Embeddings?**
- **Definition**: Numerical representations of text
- **Format**: Vector of numbers (e.g., [0.2, -0.5, 0.8, ...])
- **Dimension**: Typically 1536 for OpenAI embeddings
- **Property**: Similar texts have similar vectors

**`OpenAIEmbeddings`:**
- Uses OpenAI's `text-embedding-ada-002` model
- Converts text to 1536-dimensional vectors
- Captures semantic meaning

**Example:**
```python
text1 = "The cat sat on the mat"
text2 = "A feline rested on the rug"
text3 = "Python is a programming language"

# text1 and text2 will have similar embeddings
# text3 will have very different embeddings
```

**Why Embeddings?**
- Enable semantic search (meaning-based, not keyword-based)
- Find similar documents
- Cluster related content
- Power RAG systems

---

### Cell 30: Creating Vector Store

```python
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
```

**Step-by-Step Breakdown:**

1. **`RecursiveCharacterTextSplitter`:**
   - **Purpose**: Split long documents into smaller chunks
   - **Why**: LLMs have token limits, embeddings work better on smaller chunks
   - **How**: Recursively splits on separators (\n\n, \n, space)
   
   **Default Parameters:**
   ```python
   RecursiveCharacterTextSplitter(
       chunk_size=1000,        # Characters per chunk
       chunk_overlap=200,      # Overlap between chunks
       separators=["\n\n", "\n", " ", ""]
   )
   ```
   
   **Why Overlap?**
   - Prevents splitting mid-sentence
   - Maintains context across chunks
   - Improves retrieval quality

2. **`split_documents(docs)`:**
   - Takes list of Document objects
   - Returns list of smaller Document chunks
   - Preserves metadata

3. **`FAISS.from_documents()`:**
   - **Purpose**: Create vector database from documents
   - **Process**:
     1. Generate embeddings for each chunk
     2. Store embeddings in FAISS index
     3. Enable fast similarity search
   
   **Arguments:**
   - `documents`: List of Document chunks
   - `embeddings`: Embedding model to use

**What is FAISS?**
- **Full Name**: Facebook AI Similarity Search
- **Purpose**: Efficient similarity search in high-dimensional spaces
- **Speed**: Can search millions of vectors in milliseconds
- **Memory**: Optimized for large-scale applications

**Vector Store Capabilities:**
- Store document embeddings
- Perform similarity search
- Retrieve top-k most similar documents
- Filter by metadata

---

### Cell 31: Creating Document Chain

```python
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""", output_parser = output_parser)

document_chain = create_stuff_documents_chain(llm, prompt)
```

**`create_stuff_documents_chain`:**
- **Purpose**: Create a chain that passes documents to LLM
- **"Stuff" Strategy**: Put all documents into the prompt
- **Input**: List of documents + user question
- **Output**: LLM response based on documents

**Prompt Structure:**
- **`{context}`**: Placeholder for retrieved documents
- **`{input}`**: User's question
- **Instruction**: "based only on the provided context"
  - Reduces hallucinations
  - Grounds response in facts

**How It Works:**
1. Receives documents and question
2. Formats documents into context
3. Inserts into prompt template
4. Sends to LLM
5. Returns response

**Other Document Combination Strategies:**
- **Map-Reduce**: Process each document separately, then combine
- **Refine**: Iteratively refine answer with each document
- **Map-Rerank**: Score each document's answer, return best

---

### Cell 32: Creating Retrieval Chain

```python
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

**`as_retriever()`:**
- Converts vector store to retriever interface
- **Default**: Returns top 4 most similar documents
- **Customizable**:
  ```python
  retriever = vector.as_retriever(
      search_type="similarity",
      search_kwargs={"k": 6}  # Return top 6 documents
  )
  ```

**`create_retrieval_chain`:**
- **Purpose**: Combine retrieval and generation
- **Flow**:
  1. User asks question
  2. Retriever finds relevant documents
  3. Documents + question → document_chain
  4. LLM generates answer based on documents

**Complete RAG Pipeline:**
```
User Question
     ↓
Retriever (finds relevant docs)
     ↓
Document Chain (LLM + docs)
     ↓
Final Answer
```

---

### Cell 33-34: Using the RAG Chain

```python
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
```

**Response Structure:**
```python
{
    "input": "how can langsmith help with testing?",
    "context": [Document(...), Document(...), ...],  # Retrieved docs
    "answer": "LangSmith can help with testing by..."  # LLM response
}
```

**Key Fields:**
- **`input`**: Original question
- **`context`**: Documents used for answering
- **`answer`**: Final response from LLM

**Why RAG is Powerful:**
1. **Factual Accuracy**: Answers based on real documents
2. **Up-to-date**: Can use latest information
3. **Transparency**: Can see which documents were used
4. **Reduced Hallucinations**: LLM constrained to provided context
5. **Domain-Specific**: Works with your own data

---

## 🎯 Key Concepts Summary

### 1. LangChain Components

**Prompts:**
- Templates for structuring LLM inputs
- Support variables and message roles
- Enable few-shot learning

**LLMs:**
- Wrappers for language models
- Handle API communication
- Provide consistent interface

**Chains:**
- Compose multiple components
- Use pipe operator (`|`)
- Enable complex workflows

**Output Parsers:**
- Extract and format LLM responses
- Convert to desired types
- Enable downstream processing

### 2. RAG Architecture

**Components:**
1. **Document Loader**: Get content from sources
2. **Text Splitter**: Break into manageable chunks
3. **Embeddings**: Convert text to vectors
4. **Vector Store**: Store and search embeddings
5. **Retriever**: Find relevant documents
6. **LLM**: Generate answers from context

**Benefits:**
- Factual responses
- Custom knowledge bases
- Reduced hallucinations
- Transparent sourcing

### 3. Best Practices

**Security:**
- Use environment variables for API keys
- Never hardcode secrets
- Use `getpass` for interactive input

**Prompt Engineering:**
- Be specific in instructions
- Provide examples (few-shot)
- Constrain output format
- Use system messages for behavior

**RAG Optimization:**
- Tune chunk size and overlap
- Adjust number of retrieved documents
- Use metadata filtering
- Experiment with different embeddings

**Error Handling:**
- Set timeouts for API calls
- Implement retries
- Handle rate limits
- Validate inputs

---

## 🔍 Common Patterns and Use Cases

### Pattern 1: Simple Q&A
```python
chain = prompt | llm | output_parser
answer = chain.invoke({"input": "Your question"})
```

### Pattern 2: RAG Q&A
```python
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "Your question"})
```

### Pattern 3: Conversational
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
# Add to chain for conversation history
```

### Pattern 4: Multi-Step Reasoning
```python
chain = (
    prompt1 | llm | parser1 |
    prompt2 | llm | parser2
)
```

---

## 🐛 Troubleshooting

**Issue: API Key Error**
```
Solution: Ensure OPENAI_API_KEY is set correctly
Check: os.environ.get("OPENAI_API_KEY")
```

**Issue: Rate Limit Exceeded**
```
Solution: Add delays between requests or upgrade API plan
Use: time.sleep() or implement exponential backoff
```

**Issue: Token Limit Exceeded**
```
Solution: Reduce chunk_size or max_tokens
Adjust: text_splitter parameters
```

**Issue: Poor Retrieval Quality**
```
Solution: Tune chunk size, overlap, and k value
Experiment: Different embedding models
```

---

## 📚 Further Learning

**Next Steps:**
1. Explore LangChain agents (see LangChain-agent.ipynb)
2. Learn about different vector databases
3. Experiment with prompt engineering
4. Build a complete RAG application

**Resources:**
- LangChain Documentation: https://python.langchain.com/
- OpenAI API Reference: https://platform.openai.com/docs
- FAISS Documentation: https://github.com/facebookresearch/faiss

---

*This guide covers the essential concepts in LangChain.ipynb. Practice by modifying the code and experimenting with different prompts and documents!*
