# Week 28 - Quick Reference Cheat Sheet

## 🎯 Two Projects at a Glance

### Project 1: Spoiler Shield (NLP)
**Goal**: Detect spoilers in movie reviews  
**Type**: Binary Text Classification  
**Dataset**: 573K IMDB reviews  
**Key Feature**: `is_spoiler` (target label)  
**Models to Try**: Logistic Regression, Neural Network, LSTM, BERT  
**Metrics**: Accuracy, Precision, Recall, F1-Score, AUC  

### Project 2: EmoSense (Computer Vision)
**Goal**: Recognize group emotions from images  
**Type**: Face Detection + Emotion Classification  
**Dataset**: ~3,000 group images (manual labeling required)  
**Pipeline**: Face Detection → Emotion Recognition → Aggregation  
**Models**: YOLO/SSD/Haar Cascade + DeepFace  
**Metrics**: IoU, Accuracy, Precision, Recall  

---

## 🔑 Key Concepts

### Computer Vision Tasks
| Task | Description | Output |
|------|-------------|--------|
| Classification | One label per image | "Cat" |
| Detection | Boxes + labels | Box at (x,y,w,h) = "Cat" |
| Segmentation | Pixel-level labels | Every pixel labeled |
| Recognition | Identify individuals | "This is John" |

### NLP Tasks
- **Classification**: Categorize text (spam/not spam)
- **Sentiment**: Positive/Negative/Neutral
- **Q&A**: Answer questions from context
- **Translation**: Convert between languages
- **Summarization**: Condense long text

### Transfer Learning
```
Pre-trained Model (200M examples)
    ↓
Fine-tune on Your Data (1M examples)
    ↓
Better Results + Less Time + Less Cost
```



---

## 📊 Model Comparison

### BERT Variants
| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| RoBERTa | Large | Slow | Highest | High accuracy needs |
| ALBERT | Small | Fast | Good | Resource-constrained |
| ELECTRA | Medium | Fast | Good | Balance |

### Face Detection Models
| Model | Type | Accuracy | Speed | Use Case |
|-------|------|----------|-------|----------|
| Haar Cascade | Traditional | Low | Fast | Simple/lightweight |
| YOLO | Deep Learning | High | Fast | Production |
| SSD | Deep Learning | Medium | Medium | Alternative |

---

## 💻 Essential Code Snippets

### Spoiler Shield - BERT Fine-tuning
```python
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

# Train
outputs = model(**inputs, labels=labels)
loss = outputs.loss
```

### EmoSense - Face Detection + Emotion
```python
from ultralytics import YOLO
from deepface import DeepFace

# Detect faces
model = YOLO('yolov8n.pt')
results = model(image)

# For each face
for box in results[0].boxes:
    face_crop = image[y1:y2, x1:x2]
    
    # Recognize emotion
    emotion = DeepFace.analyze(face_crop, actions=['emotion'])
    print(emotion[0]['dominant_emotion'])
```



### IoU Calculation
```python
def calculate_iou(box1, box2):
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    
    # Union
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union
```

---

## 📈 Evaluation Metrics

### Classification Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix

# Get predictions
y_pred = model.predict(X_test)

# Metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Individual metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
```

### Confusion Matrix
```
                Predicted
              Pos    Neg
Actual  Pos   TP     FN
        Neg   FP     TN
```

---

## 🚀 Deployment Quick Start

### Streamlit App
```python
import streamlit as st

st.title("Your Project Name")

# Input
uploaded_file = st.file_uploader("Upload")

# Process
if uploaded_file:
    result = model.predict(uploaded_file)
    st.write(f"Result: {result}")
```

### Deploy to Hugging Face
```bash
# Install
pip install gradio

# Create app
import gradio as gr
demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()
```



---

## ✅ Project Checklist

### Spoiler Shield
- [ ] Download IMDB dataset
- [ ] EDA (6-8 visualizations)
- [ ] Text preprocessing
- [ ] Try 3+ models
- [ ] Hyperparameter tuning
- [ ] Evaluate with multiple metrics
- [ ] Error analysis
- [ ] Document results
- [ ] Create web interface
- [ ] Submit notebook + PDF

### EmoSense
- [ ] Download group images
- [ ] Manual labeling (50-100 images)
- [ ] Implement face detection
- [ ] Implement emotion recognition
- [ ] Aggregation logic
- [ ] Calculate IoU
- [ ] Latency analysis
- [ ] Innovation (multimodal?)
- [ ] Create web interface
- [ ] Submit notebook + PDF

---

## 🎯 Grading Focus Areas

### Both Projects
1. **Understanding** (25%): Clear problem comprehension
2. **EDA** (20%): Comprehensive analysis with visualizations
3. **Modeling** (25%): Multiple algorithms, tuning
4. **Evaluation** (20%): Appropriate metrics, analysis
5. **Documentation** (10%): Clear, organized, complete

### Spoiler Shield Specific
- 3+ algorithms tested
- Cross-validation
- Result interpretation
- Future work scoped

### EmoSense Specific
- Accurate face detection
- Emotion prediction per face
- Manual labeling quality
- Innovation demonstrated
- Latency analysis



---

## 💡 Pro Tips

### General
- ✅ Start small (1K samples), then scale
- ✅ Use Kaggle for free GPU (35 hrs/week)
- ✅ Don't train from scratch - use pre-trained
- ✅ Document everything as you go
- ✅ Create web demo for portfolio

### Spoiler Shield
- Compare review with plot synopsis
- Use semantic similarity
- Try ensemble methods
- Analyze false positives/negatives

### EmoSense
- Ensure diverse labeling (emotions, settings)
- Consider image quality enhancement
- Test different aggregation methods
- Maintain consistent labeling order

---

## 🔗 Essential Links

### Tools
- Kaggle: https://kaggle.com
- Hugging Face: https://huggingface.co
- DeepFace: https://github.com/serengil/deepface
- YOLO: https://github.com/ultralytics/ultralytics
- Label Studio: https://labelstud.io
- Streamlit: https://streamlit.io

### Learning
- Hugging Face Course (free)
- Fast.ai Practical DL
- Kaggle Learn

---

## 📅 Timeline

**Week 1**: Setup + EDA  
**Week 2**: Model Development  
**Week 3**: Enhancement + Documentation  
**Week 4**: Deployment + Submission  

---

## 🎓 Key Quotes from Instructor

> "Start small, see if pipeline works, then scale up"

> "You're not expected to train state-of-art models"

> "Focus on understanding basics and underlying concepts"

> "Make sure you have UI where people can interact"

---

*Quick Reference for Week 28 DL Mini Projects*
