# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is an **Interview Kickstart Advanced Machine Learning** course repository containing educational materials for a comprehensive 23+ week ML curriculum. The repository contains Jupyter notebooks, assignments, solutions, and detailed coding guides for each week's topics ranging from Python fundamentals to advanced deep learning concepts.

## Development Environment

### Python Environment
- **Python Version**: 3.12.1 (managed via pyenv)
- **Virtual Environment**: `.venv/` (Python venv)
- **Activation**: `source .venv/bin/activate`

### Key Dependencies
The course uses several major ML/DS libraries:
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Classical ML**: scikit-learn
- **Data Science**: pandas, numpy, matplotlib, seaborn
- **NLP**: NLTK, spaCy
- **Computer Vision**: OpenCV, PIL
- **Jupyter**: For all notebook-based learning

## Repository Structure

### Content Organization
The repository is organized by weeks, each focusing on specific ML topics:

- **Weeks 1-4**: Python fundamentals and ML libraries
- **Weeks 5-9**: Statistics, probability, and exploratory data analysis
- **Weeks 10-17**: Classical ML algorithms (regression, classification, clustering)
- **Weeks 18-22**: Deep learning and computer vision
- **Week 23+**: Natural Language Processing

### File Types
- **`.ipynb`**: Main learning notebooks and assignments
- **`*_Solution.ipynb`**: Assignment solutions
- **`*_Question.ipynb`**: Assignment templates
- **`*_Guide.md`**: Detailed line-by-line coding explanations
- **`.pdf`**: Course materials and cheat sheets

## Common Development Tasks

### Working with Notebooks
```bash
# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

### Installing Packages
```bash
# Activate environment first
source .venv/bin/activate

# Install new packages
pip install package_name

# Install from requirements (if available)
pip install -r requirements.txt
```

### Running Specific Notebooks
```bash
# Navigate to specific week
cd "Week 18 - Intro to Neural Networks"

# Launch Jupyter in that directory
jupyter notebook
```

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

## Working with Specific Topics

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

### NLP (Week 23+)
```python
# NLP toolkit imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
```

## Important Notes

### Git Configuration
- The repository tracks `.ipynb` and `.md` files
- Most other files are ignored via `.gitignore`
- Focus is on educational content, not production code

### Environment Management
- Always activate the virtual environment before working
- The `.venv` directory contains a minimal Python 3.12.1 setup
- Install packages as needed for specific assignments

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
