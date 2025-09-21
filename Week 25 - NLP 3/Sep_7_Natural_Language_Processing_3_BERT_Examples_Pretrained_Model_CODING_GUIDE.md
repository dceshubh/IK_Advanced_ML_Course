# Natural Language Processing 3: BERT Examples with Pretrained Models - Coding Guide

## Overview
This notebook demonstrates practical applications of BERT (Bidirectional Encoder Representations from Transformers) using pretrained models for various NLP tasks including sentiment analysis, text summarization, and question answering. It showcases how to leverage the Hugging Face Transformers library to implement state-of-the-art NLP solutions with minimal code.

## Learning Objectives
- Understand how to use pretrained BERT models for different NLP tasks
- Learn sentiment analysis using ELECTRA models
- Implement text summarization with BERT-based extractive summarizers
- Build question-answering systems using fine-tuned BERT models
- Master the Hugging Face Transformers pipeline API
- Understand tokenization and model inference processes

## Key Libraries and Their Purpose

### 1. **Transformers** - Hugging Face Transformers Library
```python
import transformers
from transformers import AutoTokenizer, BertForQuestionAnswering, BertTokenizer, pipeline
```
- **Purpose**: State-of-the-art NLP library providing pretrained models
- **Key Features**:
  - Access to thousands of pretrained models
  - Easy-to-use pipeline API for common NLP tasks
  - Model-specific tokenizers and architectures
  - Support for PyTorch and TensorFlow backends

### 2. **PyTorch** - Deep Learning Framework
```python
import torch
```
- **Purpose**: Tensor operations and neural network computations
- **Key Functions Used**:
  - `torch.no_grad()`: Disables gradient computation for inference
  - `torch.argmax()`: Finds indices of maximum values
  - `torch.tensor()`: Creates tensors from data

### 3. **ELECTRA Classifier** - Specialized Sentiment Analysis
```python
from electra_classifier import ElectraClassifier
```
- **Purpose**: ELECTRA-based sentiment classification model
- **Why ELECTRA**: More efficient than BERT for many tasks, uses replaced token detection

### 4. **BERT Extractive Summarizer** - Text Summarization
```python
from summarizer import Summarizer, TransformerSummarizer
```
- **Purpose**: Extractive text summarization using BERT embeddings
- **Approach**: Selects most important sentences from original text

## Code Analysis by Section

### Section 1: Environment Setup and Library Installation

#### Cell 1-2: Basic Setup
```bash
!pip install transformers
```
```python
import transformers
print(transformers.__version__)
```

**Installation and Version Check**:
- **transformers**: Core library for pretrained NLP models
- **Version checking**: Ensures compatibility and debugging support
- **Best Practice**: Always verify library versions in production environments

### Section 2: Sentiment Analysis with ELECTRA

#### Cell 3-4: ELECTRA Model Setup
```bash
!pip install electra-classifier
```
```python
import torch
from transformers import AutoTokenizer
from electra_classifier import ElectraClassifier

# Load tokenizer and model
model_name = "jbeno/electra-large-classifier-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ElectraClassifier.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Run inference
text = "I love this restaurant!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs)
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = model.config.id2label[predicted_class_id]
    print(f"Predicted label: {predicted_label}")
```

**Detailed Component Analysis**:

1. **Model Selection**:
   - `"jbeno/electra-large-classifier-sentiment"`: Pretrained ELECTRA model for sentiment analysis
   - **ELECTRA Architecture**: Uses replaced token detection instead of masked language modeling
   - **Large Model**: More parameters = better performance but slower inference

2. **AutoTokenizer**:
   - **Purpose**: Automatically selects appropriate tokenizer for the model
   - **Flexibility**: Works with any model architecture without manual specification
   - **return_tensors="pt"**: Returns PyTorch tensors instead of lists

3. **Model Evaluation Mode**:
   ```python
   model.eval()
   ```
   - **Purpose**: Disables dropout and batch normalization updates
   - **Critical**: Ensures consistent inference results
   - **Memory**: Reduces memory usage during inference

4. **Inference Process**:
   ```python
   with torch.no_grad():
       logits = model(**inputs)
   ```
   - **torch.no_grad()**: Disables gradient computation for faster inference
   - **logits**: Raw model outputs before softmax activation
   - **Unpacking**: `**inputs` passes tokenizer outputs as keyword arguments

5. **Prediction Extraction**:
   ```python
   predicted_class_id = torch.argmax(logits, dim=1).item()
   predicted_label = model.config.id2label[predicted_class_id]
   ```
   - **argmax**: Finds index of highest probability class
   - **dim=1**: Operates along class dimension
   - **item()**: Converts single-element tensor to Python scalar
   - **id2label**: Maps class indices to human-readable labels

#### Cell 5-6: Pipeline API Approach
```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="jbeno/electra-large-classifier-sentiment")

# Predict sentiment
result = pipe("I love this restaurant!")
print(result)
```

**Pipeline API Benefits**:

1. **Simplicity**: One-line model loading and inference
2. **Abstraction**: Handles tokenization, inference, and post-processing automatically
3. **Consistency**: Standardized interface across different tasks
4. **Efficiency**: Optimized for common use cases

**Pipeline Output Format**:
```python
[{'label': 'POSITIVE', 'score': 0.9998}]
```
- **label**: Predicted class name
- **score**: Confidence probability (0-1 range)

### Section 3: Text Summarization with BERT

#### Cell 12: Library Installation
```bash
!pip install bert-extractive-summarizer torch
```

**bert-extractive-summarizer**:
- **Approach**: Extractive summarization (selects existing sentences)
- **vs. Abstractive**: Doesn't generate new text, only selects important parts
- **BERT Integration**: Uses BERT embeddings to identify key sentences

#### Cell 15-17: Summarizer Setup and Function Definition
```python
# Importing the library
from summarizer import Summarizer, TransformerSummarizer

#1. Load the model
bert_model = Summarizer()

#2. Function to generate and print the summary
def gen_text_summary(Model, Text, min_length=60, num_sentences=1):
    bert_summary = bert_model(text, min_length=min_length, num_sentences=num_sentences)
    print(bert_summary)
```

**Summarizer Components**:

1. **Summarizer Class**:
   - **Default Model**: Uses DistilBERT for efficiency
   - **Clustering**: Groups sentences by semantic similarity
   - **Selection**: Picks representative sentences from each cluster

2. **Function Parameters**:
   - **Model**: Summarizer instance (for flexibility)
   - **Text**: Input text to summarize
   - **min_length**: Minimum character count for summary
   - **num_sentences**: Number of sentences in output summary

3. **Algorithm Overview**:
   ```mermaid
   flowchart TD
       A[Input Text] --> B[Sentence Tokenization]
       B --> C[BERT Embeddings for Each Sentence]
       C --> D[K-Means Clustering]
       D --> E[Select Centroid Sentences]
       E --> F[Rank by Importance]
       F --> G[Return Top N Sentences]
   ```

#### Cell 18-20: Summarization Example
```python
# initializing the original text
text = '''
Natural language processing (NLP) is an area of computer science and artificial intelligence concerned with the interaction between computers and humans in natural language. The ultimate goal of NLP is to help computers understand language as well as we do. It is the driving force behind things like virtual assistants, speech recognition, sentiment analysis, automatic text summarization, machine translation and much more...
'''

# function call
gen_text_summary(bert_model, text, min_length=60, num_sentences=3)
gen_text_summary(bert_model, text, min_length=60, num_sentences=1)
```

**Parameter Impact**:
- **num_sentences=3**: Returns 3 most important sentences
- **num_sentences=1**: Returns single most representative sentence
- **min_length=60**: Ensures summary has at least 60 characters

### Section 4: Question Answering with BERT

#### Cell 22-24: QA Model Setup
```bash
!pip install transformers
!pip install spacy
```
```python
# importing the libraries
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
```

**Library Requirements**:
- **BertForQuestionAnswering**: BERT model fine-tuned for QA tasks
- **spacy**: Advanced NLP library (imported but not used in this example)

#### Cell 26-27: Model and Tokenizer Loading
```python
#4. loading the model bert-large-uncased-whole-word-masking-finetuned-squad
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#5. loading the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
```

**Model Specifications**:

1. **bert-large-uncased-whole-word-masking-finetuned-squad**:
   - **Architecture**: 24 layers, 1024 hidden dimensions, 16 attention heads
   - **Parameters**: 336M parameters
   - **Training**: Whole Word Masking during pretraining
   - **Fine-tuning**: SQuAD (Stanford Question Answering Dataset)

2. **Whole Word Masking**:
   - **Innovation**: Masks entire words instead of individual tokens
   - **Benefit**: Better understanding of word-level semantics
   - **Example**: "playing" → mask all subword tokens [play, ##ing]

#### Cell 28: Question Answering Function
```python
def bert_ans_questions_from_text(Model, Tokenizer, Text, Question):
    # Encode the sequence
    encoding = Tokenizer.encode_plus(text=Question, text_pair=Text)
    inputs = encoding['input_ids']
    sentence_embedding = encoding['token_type_ids']
    tokens = Tokenizer.convert_ids_to_tokens(inputs)
    
    # Model inference
    start_scores, end_scores = Model(
        input_ids=torch.tensor([inputs]), 
        token_type_ids=torch.tensor([sentence_embedding]), 
        return_dict=False
    )
    
    # Find answer span
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = tokens[start_index:end_index+1]
    
    # Reconstruct answer text
    corrected_answer = ''
    for word in answer:
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word
    
    print(corrected_answer)
```

**Function Breakdown**:

1. **Input Encoding**:
   ```python
   encoding = Tokenizer.encode_plus(text=Question, text_pair=Text)
   ```
   - **encode_plus**: Advanced encoding with special tokens
   - **text_pair**: Combines question and context with [SEP] token
   - **Format**: [CLS] question [SEP] context [SEP]

2. **Encoding Components**:
   - **input_ids**: Token IDs for the input sequence
   - **token_type_ids**: Segment embeddings (0 for question, 1 for context)
   - **tokens**: Human-readable tokens for debugging

3. **Model Inference**:
   ```python
   start_scores, end_scores = Model(input_ids=..., token_type_ids=...)
   ```
   - **start_scores**: Probability distribution over start positions
   - **end_scores**: Probability distribution over end positions
   - **return_dict=False**: Returns tuple instead of dictionary

4. **Answer Extraction**:
   ```python
   start_index = torch.argmax(start_scores)
   end_index = torch.argmax(end_scores)
   ```
   - **Logic**: Find most likely start and end positions
   - **Span**: Extract tokens between start and end indices

5. **Text Reconstruction**:
   ```python
   if word[0:2] == '##':
       corrected_answer += word[2:]
   else:
       corrected_answer += ' ' + word
   ```
   - **Subword Handling**: Removes '##' prefix from subword tokens
   - **Space Management**: Adds spaces between complete words
   - **Result**: Reconstructed human-readable answer

#### Cell 29-30: QA Example
```python
# initializing the question and paragraph variable
question = '''What is Machine Learning?'''
paragraph = ''' Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task...'''

# function call
bert_ans_questions_from_text(Model=model, Tokenizer=tokenizer, Text=paragraph, Question=question)
```

**QA Process Flow**:

```mermaid
flowchart TD
    A[Question + Context] --> B[Tokenization]
    B --> C[Add Special Tokens<br/>[CLS] Q [SEP] C [SEP]]
    C --> D[Create Token Type IDs<br/>0s for Q, 1s for C]
    D --> E[BERT Model Inference]
    E --> F[Start/End Score Predictions]
    F --> G[Find Argmax Positions]
    G --> H[Extract Token Span]
    H --> I[Reconstruct Answer Text]
    I --> J[Return Final Answer]
```

## Advanced Concepts and Best Practices

### 1. **Model Selection Guidelines**

| Task | Recommended Model | Reasoning |
|------|------------------|-----------|
| Sentiment Analysis | ELECTRA, RoBERTa | Efficient, good performance |
| Text Summarization | BERT, DistilBERT | Good sentence representations |
| Question Answering | BERT-Large-SQuAD | Specifically fine-tuned for QA |
| General NLP | BERT-Base | Good balance of performance/speed |

### 2. **Performance Optimization**

```python
# Batch processing for multiple inputs
def batch_sentiment_analysis(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = pipe(batch)
        results.extend(batch_results)
    return results

# GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 3. **Error Handling and Validation**

```python
def safe_qa_inference(model, tokenizer, question, context, max_length=512):
    try:
        # Check input lengths
        encoding = tokenizer.encode_plus(
            text=question, 
            text_pair=context,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Validate encoding
        if len(encoding['input_ids'][0]) >= max_length:
            print("Warning: Input truncated due to length")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**encoding)
            
        return extract_answer(outputs, encoding, tokenizer)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None
```

### 4. **Memory Management**

```python
# Clear cache for large models
import gc
torch.cuda.empty_cache()
gc.collect()

# Use model quantization for deployment
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModel.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

## Common Issues and Solutions

### 1. **Token Length Limitations**
```python
# Problem: Input too long for model
# Solution: Implement sliding window approach
def sliding_window_qa(question, long_context, window_size=400, stride=200):
    answers = []
    for i in range(0, len(long_context), stride):
        window = long_context[i:i+window_size]
        answer = qa_function(question, window)
        if answer:
            answers.append((answer, i))
    
    # Return best answer based on confidence
    return max(answers, key=lambda x: x[1]) if answers else None
```

### 2. **Model Loading Issues**
```python
# Handle network/cache issues
try:
    model = AutoModel.from_pretrained(model_name)
except Exception as e:
    print(f"Loading from cache failed: {e}")
    model = AutoModel.from_pretrained(model_name, force_download=True)
```

### 3. **Inconsistent Results**
```python
# Set random seeds for reproducibility
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```

## Extension Opportunities

### 1. **Multi-task Pipeline**
```python
class MultiTaskNLP:
    def __init__(self):
        self.sentiment_pipe = pipeline("sentiment-analysis")
        self.summarizer = Summarizer()
        self.qa_model = BertForQuestionAnswering.from_pretrained(...)
    
    def analyze_document(self, text, questions=None):
        results = {
            'sentiment': self.sentiment_pipe(text),
            'summary': self.summarizer(text),
            'qa_results': []
        }
        
        if questions:
            for q in questions:
                answer = self.answer_question(q, text)
                results['qa_results'].append({'question': q, 'answer': answer})
        
        return results
```

### 2. **Custom Fine-tuning**
```python
from transformers import Trainer, TrainingArguments

def fine_tune_model(model, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    return trainer
```

### 3. **Real-time API Service**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models once at startup
sentiment_pipe = pipeline("sentiment-analysis")
qa_model = BertForQuestionAnswering.from_pretrained(...)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    
    result = {
        'sentiment': sentiment_pipe(text)[0],
        'length': len(text.split()),
        'summary': summarizer(text) if len(text) > 100 else text
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

This notebook provides a comprehensive introduction to using pretrained BERT models for practical NLP applications, demonstrating the power and simplicity of modern transformer-based approaches to natural language understanding.