#  Fake News Detection System

## Overview

This project is an end-to-end NLP-based machine learning system that classifies news articles as **Real or Fake**.

The model processes raw news text, converts it into numerical features using TF-IDF vectorization, and trains a Passive-Aggressive classifier for binary classification.

---

##  Machine Learning Pipeline

1. Data preprocessing
   - Lowercasing
   - Removing punctuation and numbers
   - Stopword removal using NLTK

2. Feature Engineering
   - TF-IDF Vectorization

3. Model Training
   - Passive-Aggressive Classifier
   - Train-test split (80-20)

4. Evaluation
   - Accuracy: **~99.5%**
   - Confusion Matrix
   - Classification Report

5. Deployment
   - Streamlit-based web interface
   - Model serialized using Pickle

---

##  Tech Stack

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- Matplotlib & Seaborn

--- 

##  How to Run Locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
