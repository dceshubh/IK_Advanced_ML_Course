# Gen_AI_1_Assignment_Solution - Coding Guide

## 📋 Overview
This notebook demonstrates how to build a **toxicity detector using T5 (Text-to-Text Transfer Transformer)**, a powerful language model. The project involves fine-tuning T5 for multi-label toxic comment classification.

---

## 🎯 Learning Objectives
- Understand how to fine-tune pre-trained language models
- Learn to work with multi-label classification problems
- Master the T5 text-to-text approach for classification
- Implement custom PyTorch datasets and training loops

---

## 📚 Key Libraries and Imports

### Core Data Science Libraries
```python
import pandas as pd
from google.colab import drive
from sklearn.model_selection import train_test_split
```

**Why these imports?**
- `pandas`: Essential for data manipulation and CSV file handling
- `google.colab.drive`: Allows mounting Google Drive to access datasets stored in the cloud
- `sklearn.model_selection.train_test_split`: Creates training/testing splits for model evaluation

### Deep Learning and NLP Libraries
```python
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq
)
import warnings
```

**Why these imports?**
- `torch`: PyTorch framework for deep learning operations
- `T5Tokenizer`: Converts text to tokens that T5 can understand
- `T5ForConditionalGeneration`: The main T5 model for text generation tasks
- `DataCollatorForSeq2Seq`: Handles batching and padding for sequence-to-sequence models
- `warnings`: Suppresses non-critical warnings for cleaner output

---

## 🔍 Data Exploration and Preparation

### 1. Loading and Examining the Dataset

```python
# Mount Google Drive to access stored datasets
drive.mount("/content/gdrive")

# Load the toxic comments dataset
df_train_file = pd.read_csv("/content/gdrive/MyDrive/T5/train.csv")
```

**Key Concepts:**
- **Google Drive Integration**: Colab allows seamless access to files stored in Google Drive
- **CSV Loading**: Using pandas to read structured data files

### 2. Understanding Multi-Label Classification

```python
# Examine the label structure
cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df_train_file[df_train_file["toxic"] == 1][cols].sample(10)
```

**Important Insight:**
- This is a **multi-label classification** problem
- Each comment can have multiple labels simultaneously (e.g., both "toxic" and "insult")
- Unlike multi-class classification where each sample has exactly one label

### 3. Creating Training and Test Sets

```python
# Split data into training and testing sets
df_train, df_test = train_test_split(df_train_file, test_size=0.2)

# Save the splits for later use
df_train.to_csv("/content/gdrive/MyDrive/T5/assignment_train.csv", index=False)
df_test.to_csv("/content/gdrive/MyDrive/T5/assignment_test.csv", index=False)
```

**Why this approach?**
- The original test set doesn't have labels (typical for Kaggle competitions)
- We need labeled data for evaluation, so we create our own test split
- 80/20 split is a common practice for training/testing

---

## 🤖 Model Setup and Baseline Testing

### 1. Installing Required Libraries

```python
!pip install transformers
!pip install sentencepiece
```

**Why these libraries?**
- `transformers`: Hugging Face library providing pre-trained models
- `sentencepiece`: Tokenization library used by T5

### 2. Loading the Pre-trained T5 Model

```python
# Load T5-small model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Move model to GPU for faster computation
model = model.to("cuda")
```

**Key Points:**
- **T5-small**: ~60M parameters, good balance between performance and computational requirements
- **GPU Usage**: `.to("cuda")` moves the model to GPU memory for faster training/inference
- **Pre-trained Models**: Starting with a model already trained on diverse tasks

### 3. Testing T5's Text-to-Text Approach

```python
# Test sentiment analysis (a task T5 was already fine-tuned for)
input_ids = tokenizer(
    "sst2 sentence: The movie was interesting.",
    return_tensors="pt"
).input_ids
output = model.generate(input_ids=input_ids.to("cuda"))
print(tokenizer.decode(output[0]))  # Output: <pad> positive</s>
```

**Understanding T5's Approach:**
- **Task Prefixes**: T5 uses prefixes like "sst2 sentence:" to indicate the task
- **Text-to-Text**: All tasks are framed as text generation problems
- **Tokenization**: Text is converted to numerical tokens for model processing

---

## 🏗️ Custom Dataset Implementation

### Creating a PyTorch Dataset Class

```python
class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer):
        """Initialize dataset with CSV file and tokenizer"""
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Return a single item from the dataset"""
        # Define all possible toxic categories
        categories = [
            "toxic", "severe_toxic", "obscene", 
            "threat", "insult", "identity_hate"
        ]
        
        # Get the row at the specified index
        row = self.df.iloc[idx]
        
        # Build list of applicable labels
        labels = []
        toxic = False
        for cat in categories:
            if row[cat] == 1:
                toxic = True
                labels.append(cat)
        
        # If no toxic labels, mark as "not_toxic"
        if toxic is False:
            labels.append("not_toxic")
        
        # Tokenize input with task prefix
        input_ids = self.tokenizer.encode(
            "toxic comment classification: " + row["comment_text"],
            return_tensors="pt"
        )
        
        # Tokenize output labels
        labels = self.tokenizer.encode(
            " ".join(labels),
            return_tensors="pt"
        )
        
        return {"input_ids": input_ids[0], "labels": labels[0]}
```

**Key Design Decisions:**

1. **Task Prefix**: "toxic comment classification:" tells T5 what task to perform
2. **Multi-Label Handling**: Multiple labels are joined with spaces (e.g., "toxic insult")
3. **Default Label**: Comments with no toxic labels get "not_toxic"
4. **Tokenization**: Both input and output are converted to token IDs

**PyTorch Dataset Requirements:**
- `__init__`: Initialize the dataset
- `__len__`: Return dataset size
- `__getitem__`: Return a single sample by index

---

## 🚀 Training Setup and Implementation

### 1. Creating DataLoader and Optimizer

```python
# Create dataset instance
dataset = ToxicityDataset("/content/gdrive/MyDrive/T5/assignment_train.csv", tokenizer)

# Create data collator for batching
collator = DataCollatorForSeq2Seq(tokenizer)

# Create DataLoader with batch size of 1 (due to memory constraints)
training_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collator
)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

**Important Concepts:**

- **DataCollator**: Handles padding and masking for variable-length sequences
- **Batch Size**: Set to 1 due to GPU memory limitations on free Colab
- **AdamW Optimizer**: Advanced version of Adam with weight decay regularization
- **Learning Rate**: 1e-4 is a common starting point for fine-tuning

### 2. Training Loop Implementation

```python
# Training loop for one epoch
n_batches = dataset.__len__()
current_batch = 0
total_loss = 0

for batch in training_loader:
    # Move batch to GPU
    batch = batch.to("cuda")
    
    # Forward pass and loss calculation
    output = model(**batch)
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()
    optimizer.step()
    
    # Track loss for monitoring
    total_loss += output.loss.item()
    current_batch += 1
    
    # Print progress every 100 batches
    if current_batch % 100 == 0:
        status = "Mean loss: {}. Batch {} of {}".format(
            total_loss/100,
            current_batch,
            n_batches
        )
        print(status)
        total_loss = 0
    
    # Save checkpoint every 5000 batches
    if current_batch % 5000 == 0:
        model.save_pretrained("/content/gdrive/MyDrive/T5/t5_checkpoint")
        print("Checkpointing model...")
```

**Training Loop Breakdown:**

1. **Batch Processing**: Each batch is moved to GPU for computation
2. **Forward Pass**: `model(**batch)` computes predictions and loss
3. **Backpropagation**: 
   - `optimizer.zero_grad()`: Clear previous gradients
   - `output.loss.backward()`: Compute gradients
   - `optimizer.step()`: Update model parameters
4. **Monitoring**: Track and print loss every 100 batches
5. **Checkpointing**: Save model every 5000 batches to prevent data loss

---

## 🔧 Advanced Concepts and Best Practices

### 1. Memory Management
- **Batch Size**: Limited to 1 due to GPU memory constraints
- **Gradient Clearing**: `set_to_none=True` for memory efficiency
- **GPU Usage**: Moving tensors to CUDA for faster computation

### 2. Model Checkpointing
```python
# Save model checkpoint
model.save_pretrained("/content/gdrive/MyDrive/T5/t5_checkpoint")
```
- Prevents loss of progress during long training sessions
- Essential for Colab environments that can disconnect

### 3. Text-to-Text Paradigm
- **Unified Framework**: All NLP tasks become text generation problems
- **Flexible Output**: Can generate any text sequence as output
- **Task Specification**: Prefixes tell the model what task to perform

---

## 🎯 Key Takeaways for Beginners

### 1. **Fine-tuning vs Training from Scratch**
- Fine-tuning adapts a pre-trained model to your specific task
- Much faster and requires less data than training from scratch
- Leverages knowledge learned from large-scale pre-training

### 2. **Multi-label Classification Strategy**
- Convert multiple labels into a single text string
- Use spaces to separate multiple labels
- Handle the "no labels" case with a default label

### 3. **PyTorch Dataset Pattern**
- Implement `__init__`, `__len__`, and `__getitem__` methods
- Return dictionaries with consistent key names
- Handle tokenization within the dataset class

### 4. **Training Loop Essentials**
- Always clear gradients before backpropagation
- Move data to the same device as your model
- Monitor loss to track training progress
- Save checkpoints regularly

### 5. **Practical Considerations**
- GPU memory is often the limiting factor
- Start with small batch sizes and increase if possible
- Use appropriate learning rates for fine-tuning (typically smaller than training from scratch)

---

## 🔍 Debugging Tips

1. **Memory Issues**: Reduce batch size or sequence length
2. **Slow Training**: Ensure model and data are on GPU
3. **Poor Performance**: Check data preprocessing and label formatting
4. **Tokenization Errors**: Verify input format matches model expectations

---

## 📈 Next Steps

After completing this implementation, consider:
1. **Evaluation**: Implement metrics to assess model performance
2. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes
3. **Model Variants**: Try larger T5 models (t5-base, t5-large)
4. **Data Augmentation**: Techniques to improve model robustness
5. **Deployment**: Convert the model for production use

This coding guide provides a comprehensive foundation for understanding transformer fine-tuning and multi-label text classification using the T5 architecture.