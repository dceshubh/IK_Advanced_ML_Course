# Student Version NLP3 BERT Fine-tuning - Coding Guide

## Overview
This comprehensive notebook demonstrates **text classification using transformer models**, specifically focusing on emotion detection in tweets. The notebook covers the complete machine learning pipeline from data exploration to model fine-tuning using BERT/DistilBERT.

## Learning Objectives
- Understand the Hugging Face Datasets and Transformers ecosystem
- Learn text tokenization and preprocessing techniques
- Explore feature extraction vs fine-tuning approaches
- Implement BERT-based text classification
- Understand contextual embeddings and their advantages
- Master the Trainer API for model training

---

## Section 1: Environment Setup and Data Loading

### Code Block 1: Installing Required Libraries
```python
!pip install datasets
!pip install transformers[torch]
```

**Purpose**: Install essential libraries for NLP tasks and model training.

**Key Libraries Explained**:
- **datasets**: Hugging Face library for easy access to NLP datasets and data processing
- **transformers[torch]**: Hugging Face transformers library with PyTorch backend for pre-trained models
- **[torch] suffix**: Ensures PyTorch dependencies are installed alongside transformers

### Code Block 2: Loading the Emotions Dataset
```python
from datasets import load_dataset

emotions = load_dataset("emotion")
```

**Purpose**: Load a pre-labeled emotion classification dataset from Hugging Face Hub.

**Dataset Details**:
- **Source**: Hugging Face Hub's emotion dataset
- **Task**: Multi-class emotion classification
- **Classes**: 6 emotions (anger, disgust, fear, joy, sadness, surprise)
- **Format**: Automatically downloads and caches the dataset locally

**Key Concepts**:
- **Hugging Face Hub**: Central repository for datasets and models
- **Automatic Caching**: Downloaded datasets are stored locally for future use
- **DatasetDict Structure**: Contains train/validation/test splits

---

## Section 2: Data Exploration and Analysis

### Code Block 3: Exploring Dataset Structure
```python
emotions
train_ds = emotions["train"]
len(train_ds)
train_ds[0]
train_ds.column_names
print(train_ds.features)
```

**Purpose**: Understand the dataset structure and data types.

**Key Insights**:
- **DatasetDict**: Dictionary-like object containing different splits
- **Dataset Object**: Array-like structure for individual splits
- **Features**: Metadata about column types and label mappings
- **ClassLabel**: Special data type that maps integers to class names

**Data Structure Analysis**:
- **Text Column**: Contains tweet text as strings
- **Label Column**: Integer labels (0-5) representing emotions
- **Indexing**: Can access individual examples or slices like Python lists

### Code Block 4: Converting to Pandas for Analysis
```python
import pandas as pd

emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)
```

**Purpose**: Convert dataset to Pandas DataFrame for easier data analysis and visualization.

**Key Functions**:
- **`set_format(type="pandas")`**: Changes output format without modifying underlying data
- **`int2str()`**: Converts integer labels to human-readable strings
- **`apply()`**: Applies function to each row/column in DataFrame

**Why Pandas?**:
- **Familiar Interface**: Easy data manipulation and analysis
- **Visualization**: Better integration with matplotlib/seaborn
- **Temporary Conversion**: Can switch back to original format later

### Code Block 5: Class Distribution Analysis
```python
import matplotlib.pyplot as plt

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()
```

**Purpose**: Visualize class distribution to identify potential imbalances.

**Analysis Insights**:
- **Imbalanced Dataset**: Joy and sadness are most frequent
- **Rare Classes**: Love and surprise appear much less frequently
- **Impact**: Class imbalance affects model training and evaluation strategies

**Best Practices**:
- Always check class distribution before training
- Consider stratified sampling for train/validation splits
- May need class weighting or resampling techniques

### Code Block 6: Text Length Analysis
```python
import seaborn as sns
df["Words Per Tweet"] = df["text"].str.split().apply(len)
sns.violinplot(data=df, x ='label_name', y='Words Per Tweet')
plt.show()
```

**Purpose**: Analyze text length distribution across different emotions.

**Key Insights**:
- **Average Length**: Most tweets around 15 words
- **Model Compatibility**: All texts well below BERT's 512 token limit
- **Emotion Patterns**: Different emotions may have different typical lengths

**Why This Matters**:
- **Truncation Strategy**: Determines if we need to handle long texts
- **Padding Requirements**: Influences batch processing decisions
- **Model Selection**: Helps choose appropriate model architecture

---

## Section 3: Text Tokenization Fundamentals

### Code Block 7: Character Tokenization Example
```python
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
input_ids = [token2idx[token] for token in tokenized_text]
```

**Purpose**: Demonstrate the simplest tokenization approach and its limitations.

**Character Tokenization**:
- **Pros**: Handles any text, no out-of-vocabulary issues
- **Cons**: Loses linguistic structure, requires learning word boundaries
- **Use Cases**: Rarely used in practice due to inefficiency

**Numericalization Process**:
1. **Vocabulary Creation**: Map each unique character to integer
2. **Encoding**: Convert text to sequence of integers
3. **Model Input**: Neural networks require numerical inputs

### Code Block 8: Word Tokenization with NLTK
```python
import nltk
from nltk import word_tokenize, TweetTokenizer
nltk.download('punkt')

text = "Tokenizing text is a core task of NLP :D."
tokenized_text = TweetTokenizer().tokenize(text)
print(tokenized_text)
```

**Purpose**: Show word-level tokenization and its advantages for social media text.

**TweetTokenizer Advantages**:
- **Social Media Aware**: Handles hashtags, mentions, emoticons
- **Punctuation Handling**: Separates punctuation appropriately
- **URL Recognition**: Properly tokenizes web links and handles contractions

**Word vs Character Tokenization**:
- **Linguistic Structure**: Preserves word boundaries and meaning
- **Vocabulary Size**: Manageable vocabulary compared to character-level
- **OOV Problem**: Unknown words can't be handled effectively

---

## Section 4: Subword Tokenization with BERT

### Code Block 9: Loading DistilBERT Tokenizer
```python
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```

**Purpose**: Load the tokenizer that matches the pre-trained model.

**Model Selection**:
- **DistilBERT**: Smaller, faster version of BERT (66% size, 97% performance)
- **Uncased**: Treats uppercase and lowercase the same
- **AutoTokenizer**: Automatically loads correct tokenizer for the model

**Critical Principle**: Always use the same tokenizer that the model was trained with!

### Code Block 10: Understanding WordPiece Tokenization
```python
encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))
```

**Purpose**: Explore how WordPiece tokenization works and its benefits.

**WordPiece Algorithm**:
- **Subword Units**: Splits rare words into smaller, known pieces
- **Frequent Words**: Keeps common words as single tokens
- **## Prefix**: Indicates continuation of previous token
- **Special Tokens**: [CLS] (classification), [SEP] (separator)

**Key Advantages**:
- **OOV Handling**: Can represent any word using subword pieces
- **Efficiency**: Smaller vocabulary than word-level
- **Semantic Preservation**: Maintains more meaning than character-level

### Code Block 11: Tokenizer Properties
```python
tokenizer.vocab_size
tokenizer.model_max_length
tokenizer.model_input_names
```

**Purpose**: Understand tokenizer limitations and requirements.

**Important Properties**:
- **vocab_size**: Number of unique tokens (30,522 for DistilBERT)
- **model_max_length**: Maximum sequence length (512 tokens)
- **model_input_names**: Required input fields (['input_ids', 'attention_mask'])

---

## Section 5: Dataset Tokenization

### Code Block 12: Batch Tokenization Function
```python
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

print(tokenize(emotions["train"][:2]))
```

**Purpose**: Create a function to tokenize entire dataset efficiently.

**Function Parameters**:
- **padding=True**: Adds [PAD] tokens to make all sequences same length
- **truncation=True**: Cuts sequences longer than max_length
- **batch processing**: Processes multiple texts simultaneously for efficiency

**Output Analysis**:
- **input_ids**: Token IDs for model input
- **attention_mask**: Indicates which tokens are real vs padding
- **Consistent Length**: All sequences padded to same length

### Code Block 13: Special Tokens Overview
```python
tokens2ids = list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
data = sorted(tokens2ids, key=lambda x : x[-1])
df = pd.DataFrame(data, columns=["Special Token", "Special Token ID"])
```

**Purpose**: Understand the role of special tokens in BERT.

**Special Tokens Explained**:
- **[PAD] (0)**: Padding token for sequence alignment
- **[UNK] (100)**: Unknown token for out-of-vocabulary words
- **[CLS] (101)**: Classification token, used for sequence-level tasks
- **[SEP] (102)**: Separator token, marks end of sequence
- **[MASK] (103)**: Masking token for pre-training (not used in classification)

### Code Block 14: Full Dataset Tokenization
```python
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)
```

**Purpose**: Apply tokenization to entire dataset efficiently.

**Key Parameters**:
- **batched=True**: Process multiple examples at once
- **batch_size=None**: Process entire dataset as single batch
- **map() method**: Applies function to all examples in dataset

**Result**: Adds 'input_ids' and 'attention_mask' columns to dataset

---

## Section 6: Feature Extraction Approach

### Code Block 15: Loading Pre-trained Model
```python
import torch
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
```

**Purpose**: Load pre-trained BERT model for feature extraction.

**Key Concepts**:
- **AutoModel**: Loads model without task-specific head
- **Device Management**: Automatically uses GPU if available
- **Pre-trained Weights**: Model already understands language from pre-training

### Code Block 16: Extracting Hidden States
```python
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
outputs.last_hidden_state.size()
```

**Purpose**: Extract contextual embeddings from BERT for a single text.

**Step-by-Step Process**:
1. **Tokenization**: Convert text to token IDs
2. **Tensor Conversion**: return_tensors="pt" creates PyTorch tensors
3. **Device Transfer**: Move tensors to GPU/CPU as needed
4. **Model Inference**: Get hidden states without computing gradients
5. **Output Shape**: [batch_size, sequence_length, hidden_size]

**Hidden States Explained**:
- **Contextual**: Each token's representation depends on surrounding context
- **768 Dimensions**: Each token represented by 768-dimensional vector
- **Layer Output**: Using final layer representations

### Code Block 17: BERT Contextual Embeddings Demo
```python
text1 = 'I went to the bank and took out money.'
text2 = 'I went to the river bank and got in the water.'

# Process both texts and extract embeddings for "bank"
# Compare embeddings to show contextual differences
```

**Purpose**: Demonstrate how BERT creates different embeddings for the same word in different contexts.

**Key Insight**: Unlike Word2Vec, BERT embeddings change based on context:
- **Financial Context**: "bank" near "money" gets financial embedding
- **Geographic Context**: "bank" near "river" gets geographic embedding
- **Contextual Understanding**: This is BERT's key advantage over static embeddings

### Code Block 18: Batch Hidden State Extraction
```python
def extract_hidden_states(batch):
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
```

**Purpose**: Extract [CLS] token embeddings for entire dataset.

**Function Breakdown**:
- **Input Filtering**: Only pass required inputs to model
- **GPU Processing**: Move tensors to device for computation
- **[CLS] Extraction**: Use first token ([:,0]) for classification
- **CPU Return**: Move results back to CPU as NumPy arrays

**Why [CLS] Token?**:
- **Sequence Representation**: Designed to represent entire sequence
- **Classification Tasks**: Standard practice for sentence-level classification
- **Pre-training**: Trained to aggregate sequence information

### Code Block 19: Creating Feature Matrix
```python
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

X_train = np.array(emotions_hidden["train"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
```

**Purpose**: Convert dataset to scikit-learn compatible format.

**Data Preparation**:
- **Format Setting**: Convert specific columns to PyTorch tensors
- **Feature Extraction**: Apply hidden state extraction to all examples
- **NumPy Conversion**: Create standard ML arrays for training

**Result**: X_train contains 768-dimensional vectors, y_train contains labels

---

## Section 7: Traditional ML Classification

### Code Block 20: Logistic Regression Training
```python
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=300)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)
```

**Purpose**: Train a simple classifier on BERT features.

**Approach Benefits**:
- **Fast Training**: No need to update BERT weights
- **Low Resource**: Can run on CPU without GPU
- **Interpretable**: Logistic regression coefficients are interpretable

**Performance**: Achieves ~63% accuracy using frozen BERT features

### Code Block 21: Baseline Comparison
```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)
```

**Purpose**: Establish baseline performance for comparison.

**Baseline Strategy**: Always predict most frequent class
**Result**: ~35% accuracy (much lower than BERT features)
**Conclusion**: BERT features provide significant improvement over naive baseline

### Code Block 22: Confusion Matrix Analysis
```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
```

**Purpose**: Visualize model performance across different classes.

**Analysis Insights**:
- **Diagonal Elements**: Correct predictions (higher is better)
- **Off-diagonal**: Confusion between classes
- **Pattern Recognition**: Anger/fear often confused with sadness

---

## Section 8: Fine-tuning Approach

### Code Block 23: Loading Model for Classification
```python
from transformers import AutoModelForSequenceClassification

num_labels = 6
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))
```

**Purpose**: Load BERT model with classification head for fine-tuning.

**Key Differences from Feature Extraction**:
- **Classification Head**: Adds linear layer on top of BERT
- **End-to-End Training**: All parameters can be updated
- **Task-Specific**: Model adapts specifically to emotion classification

**Warning Message**: "Some weights randomly initialized" is normal - the classification head is new!

### Code Block 24: Metrics Definition
```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
```

**Purpose**: Define evaluation metrics for training monitoring.

**Metrics Explained**:
- **Accuracy**: Percentage of correct predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Weighted Average**: Accounts for class imbalance
- **argmax(-1)**: Converts logits to predicted class labels

### Code Block 25: Training Configuration
```python
from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    eval_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error",
    report_to="none"
)
```

**Purpose**: Configure training hyperparameters and logging.

**Key Hyperparameters**:
- **Learning Rate (2e-5)**: Small rate for fine-tuning pre-trained models
- **Epochs (2)**: Few epochs prevent overfitting
- **Batch Size (64)**: Balance between memory and training stability
- **Weight Decay (0.01)**: L2 regularization to prevent overfitting
- **Evaluation Strategy**: Evaluate after each epoch

**Best Practices**:
- **Small Learning Rates**: Pre-trained models need gentle updates
- **Few Epochs**: Avoid catastrophic forgetting of pre-trained knowledge
- **Regular Evaluation**: Monitor performance during training

### Code Block 26: Model Training
```python
trainer = Trainer(
    model=model, 
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer
)
trainer.train()
```

**Purpose**: Execute the fine-tuning process using Hugging Face Trainer.

**Trainer Benefits**:
- **Automatic Optimization**: Handles gradient computation and updates
- **Evaluation**: Automatically runs evaluation at specified intervals
- **Logging**: Tracks metrics and saves checkpoints
- **Device Management**: Handles GPU/CPU operations

**Training Process**:
1. **Forward Pass**: Compute predictions and loss
2. **Backward Pass**: Compute gradients
3. **Parameter Update**: Update model weights
4. **Evaluation**: Assess performance on validation set

**Expected Results**: ~92% F1-score (significant improvement over feature extraction)

---

## Section 9: Model Evaluation and Inference

### Code Block 27: Model Prediction and Analysis
```python
preds_output = trainer.predict(emotions_encoded["validation"])
y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels)
```

**Purpose**: Evaluate fine-tuned model performance in detail.

**Prediction Process**:
- **Raw Logits**: Model outputs probability scores for each class
- **Argmax**: Convert to predicted class labels
- **Confusion Matrix**: Visualize performance across all classes

**Performance Analysis**:
- **Improved Accuracy**: Much better diagonal in confusion matrix
- **Remaining Confusions**: Love still confused with joy (semantically similar)
- **Overall**: Significant improvement over feature extraction approach

### Code Block 28: Real-time Inference
```python
from torch.nn.functional import softmax

# Create label mapping
ind2label = {i:name for i,name in enumerate(emotions["train"].features["label"].names)}

# Test on new tweets
tweet = 'I am very sad about this product. I was expecting better'
tweet_tokens = tokenizer(tweet, return_tensors='pt').to(device='cuda')
predicted_num_label = softmax(trainer.model(tweet_tokens.input_ids).logits, dim=1).argmax(dim=1)
pred_num_label = predicted_num_label.item()
print("predicted_label: ", ind2label[pred_num_label])
```

**Purpose**: Demonstrate how to use the trained model for new predictions.

**Inference Pipeline**:
1. **Tokenization**: Convert text to model input format
2. **Model Forward**: Get raw logits from model
3. **Softmax**: Convert logits to probabilities
4. **Argmax**: Get most likely class
5. **Label Mapping**: Convert number back to emotion name

**Production Considerations**:
- **Batch Processing**: Can process multiple texts simultaneously
- **Confidence Scores**: Softmax provides probability distributions
- **Error Handling**: Should validate inputs and handle edge cases

---

## Key Concepts Summary

### 1. Feature Extraction vs Fine-tuning
- **Feature Extraction**: Use BERT as fixed feature extractor + simple classifier
  - **Pros**: Fast, low resource, interpretable
  - **Cons**: Limited performance, doesn't adapt to task
- **Fine-tuning**: Update all model parameters for specific task
  - **Pros**: Higher performance, task-specific adaptation
  - **Cons**: More resources, longer training time

### 2. Tokenization Strategies
- **Character**: Simple but inefficient
- **Word**: Intuitive but OOV problems
- **Subword (WordPiece)**: Best balance of efficiency and coverage

### 3. BERT Architecture Benefits
- **Contextual Embeddings**: Word meaning depends on context
- **Bidirectional**: Considers both left and right context
- **Transfer Learning**: Leverages pre-trained language understanding

### 4. Training Best Practices
- **Small Learning Rates**: Preserve pre-trained knowledge
- **Few Epochs**: Prevent overfitting and catastrophic forgetting
- **Proper Evaluation**: Use appropriate metrics for imbalanced data
- **Device Management**: Utilize GPU for faster training

---

## Practical Applications

1. **Sentiment Analysis**: Customer feedback, social media monitoring
2. **Content Moderation**: Detecting harmful or inappropriate content
3. **Customer Support**: Automatic ticket routing based on emotion
4. **Market Research**: Understanding consumer emotions about products
5. **Mental Health**: Monitoring emotional states in text communications

---

## Advanced Concepts Covered

### 1. Attention Mechanisms
- **Attention Masks**: Prevent model from attending to padding tokens
- **Self-Attention**: How BERT relates different positions in sequence
- **Multi-Head Attention**: Multiple attention patterns for different aspects

### 2. Transfer Learning
- **Pre-training**: Learning general language understanding
- **Fine-tuning**: Adapting to specific downstream tasks
- **Domain Adaptation**: Using models trained on different but related data

### 3. Evaluation Strategies
- **Confusion Matrix**: Understanding per-class performance
- **F1-Score**: Balancing precision and recall
- **Weighted Metrics**: Handling class imbalance appropriately

This comprehensive guide provides deep understanding of modern NLP techniques, preparing you for real-world text classification challenges and advanced transformer applications.