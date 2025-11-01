# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is an **Interview Kickstart Advanced Machine Learning** course repository containing educational materials for a comprehensive 26+ week ML curriculum. The repository contains Jupyter notebooks, assignments, solutions, detailed coding guides, study materials, and hands-on projects ranging from Python fundamentals to modern ML architectures.

## Development Environment

### Python Environment
- **Python Version**: 3.12.1 (managed via pyenv)
- **Primary Virtual Environment**: `ik_env/` (Python venv) - **NEW**: Dedicated environment with 120+ ML packages
- **Legacy Environment**: `.venv/` (Basic setup)
- **Activation**: `source ik_env/bin/activate` (recommended) or `source .venv/bin/activate`

### Key Dependencies
The course uses comprehensive ML/DS ecosystem:
- **Deep Learning**: TensorFlow/Keras, PyTorch, Transformers (Hugging Face)
- **Classical ML**: scikit-learn, joblib
- **Data Science**: pandas, numpy, matplotlib, seaborn, plotly
- **NLP**: NLTK, spaCy, transformers, tokenizers
- **Computer Vision**: OpenCV, PIL, torchvision
- **Jupyter**: Full Jupyter ecosystem (notebook, lab, console)
- **Scientific Computing**: scipy, sympy
- **Visualization**: matplotlib, seaborn, plotly

## Repository Structure

### Content Organization
The repository is organized by weeks, each focusing on specific ML topics:

- **Weeks 1-4**: Python fundamentals and ML libraries
- **Weeks 5-9**: Statistics, probability, and exploratory data analysis
- **Weeks 10-17**: Classical ML algorithms (regression, classification, clustering, ensemble methods)
- **Weeks 18-22**: Deep learning and computer vision (Neural Networks, CNNs, AlexNet)
- **Weeks 23-25**: Natural Language Processing (NLP fundamentals, transformers, advanced NLP)
- **Week 26**: Modern ML Architectures (Autoencoders, VAEs, modern architectures)
- **Special Projects**: Kaggle competitions and mini-projects

### File Types and Patterns
- **`.ipynb`**: Main learning notebooks and assignments
- **`*_Solution.ipynb`**: Complete assignment solutions
- **`*_Question.ipynb`** or **`*_Assignment.ipynb`**: Assignment templates
- **`*_CODING_GUIDE.md`**: Comprehensive line-by-line coding explanations
- **`*_STUDY_GUIDE.md`**: Conceptual study materials from meeting transcripts
- **`README.md`**: Week-specific overview and guide navigation
- **`meeting_saved_closed_caption.txt`**: Raw meeting transcripts
- **`.pdf`**: Course materials, cheat sheets, and reference materials

## Common Development Tasks

### Environment Setup
```bash
# Use the main ML environment (recommended)
source ik_env/bin/activate

# Check installed packages
pip list | wc -l  # Should show ~120+ packages

# Verify key ML libraries
python -c "import pandas, numpy, sklearn, matplotlib, torch, tensorflow; print('All libraries available')"
```

### Working with Notebooks
```bash
# Activate environment (use ik_env for full ML stack)
source ik_env/bin/activate

# Launch Jupyter Notebook
jupyter notebook

# Or use JupyterLab (recommended for advanced features)
jupyter lab

# Launch Jupyter Console for interactive work
jupyter console
```

### Working with Kaggle Projects
```bash
# Navigate to Kaggle project
cd "Kaggle-Hands-On"

# Activate ML environment
source ../ik_env/bin/activate

# Run the spaceship pipeline
jupyter notebook spaceship_pipeline_notebook.ipynb
```

### Installing Additional Packages
```bash
# Always activate environment first
source ik_env/bin/activate

# Install new ML packages
pip install package_name

# For deep learning packages
pip install transformers accelerate

# For computer vision
pip install opencv-python pillow
```

### Working with Specific Topics
```bash
# Navigate to specific week
cd "Week 23 - NLP 1"

# Activate environment and launch
source ../ik_env/bin/activate
jupyter lab

# For weeks with README files
cat README.md  # Check week-specific guidance
```

## Special Directories and Projects

### Kaggle-Hands-On/
**Purpose**: Real-world machine learning projects and competitions
**Contents**:
- `spaceship_pipeline_notebook.ipynb`: Complete ML pipeline for Spaceship Titanic competition
- `spaceship_pipeline_notebook_New.ipynb`: Updated version with improvements
- `train.csv`: Training dataset for the competition
- `val_predictions_simple.csv`: Model validation results

**Key Features**:
- End-to-end ML pipeline implementation
- Feature engineering with custom transformers
- Model training with RandomForest
- Hyperparameter tuning with RandomizedSearchCV
- Model persistence with joblib
- Production-ready code structure

### ML - Mini Project/
**Purpose**: Small-scale ML projects and experiments
**Contents**:
- `AMA_ML_Mini_Project.txt`: Project specifications and requirements

### Environment Files
- `ik_env/`: Primary virtual environment with 120+ ML packages
- `.venv/`: Legacy environment (basic setup)
- `CourseCurriculum.png`: Visual course overview
- `*.pdf`: Comprehensive cheat sheets for major topics

## Architecture and Learning Path

### Course Progression
The course follows a structured learning path:

1. **Foundation Phase** (Weeks 1-9)
   - Python programming fundamentals
   - Data manipulation with pandas/numpy
   - Statistical concepts and probability theory
   - Exploratory data analysis techniques

2. **Classical ML Phase** (Weeks 10-17)
   - Supervised learning (regression, classification)
   - Unsupervised learning (clustering, dimensionality reduction)
   - Model evaluation and hyperparameter tuning
   - Ensemble methods (bagging, boosting)

3. **Deep Learning Phase** (Weeks 18-22)
   - Neural network fundamentals
   - Computer vision with CNNs
   - Advanced architectures (AlexNet, ResNet concepts)
   - Image processing and classification

4. **Advanced Topics** (Week 23+)
   - Natural Language Processing
   - Text preprocessing and analysis
   - Tokenization, stemming, lemmatization

### Implementation Approaches
The course teaches multiple implementation strategies:
- **From Scratch**: Understanding mathematical foundations
- **Library-Based**: Using scikit-learn, TensorFlow, PyTorch
- **Modular Design**: Building reusable components and classes

## Key Learning Patterns

### Assignment Structure
Most weeks follow this pattern:
- **Main Notebook**: Comprehensive topic coverage with examples
- **Assignment Question**: Template with TODO sections
- **Assignment Solution**: Complete implementation
- **Coding Guide**: Line-by-line explanation of concepts

### Code Organization
The materials emphasize:
- **Modular Programming**: Classes and functions for reusability
- **Documentation**: Extensive comments and docstrings
- **Best Practices**: Proper imports, error handling, and code structure
- **Multiple Approaches**: Comparing different implementation strategies
### Working with Specific Topics

### Neural Networks (Week 18+)
```python
# Common imports for deep learning
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers, models
```

### Computer Vision (Week 19-22)
```python
# CV-specific imports
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
```

### NLP (Week 23-25)
```python
# NLP toolkit imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Advanced NLP with transformers
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
```

### Modern ML Architectures (Week 26)
```python
# Autoencoders and advanced architectures
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
```
```

## Important Notes

### Git Configuration
- The repository tracks `.ipynb` and `.md` files
- Most other files are ignored via `.gitignore`
- Focus is on educational content, not production code

### Environment Management
- Always activate the virtual environment before working
- **Primary**: The `ik_env/` directory contains a comprehensive Python 3.12.1 setup with 120+ ML packages
- **Legacy**: The `.venv/` directory contains a minimal Python 3.12.1 setup
- Use `ik_env` for all ML work - it has the full ecosystem pre-installed
- Install additional packages as needed for specific assignments

### File Navigation
- Week directories contain multiple related notebooks
- Look for `*_Guide.md` files for detailed explanations
- Assignment solutions provide complete implementations

## Learning Resources

The repository includes comprehensive reference materials:
- **Course Curriculum**: Visual overview in `CourseCurriculum.png`
- **Cheat Sheets**: PDF guides for major topics
- **Coding Guides**: Detailed markdown explanations
- **Multiple Examples**: Different approaches to same problems

This structure supports both individual learning and reference lookup during development work.
