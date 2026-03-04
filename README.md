# Fake Job Posting Detection Using NLP and Machine Learning

> Binary text classification pipeline that detects fraudulent job postings using TF-IDF feature extraction and a Passive-Aggressive Classifier, achieving 98% accuracy and 0.77 F1-score.

---

## Overview

Online job platforms are increasingly targeted by fraudulent postings that exploit job seekers. This project builds a machine learning pipeline to automatically classify job postings as **genuine** or **fraudulent** based on textual features (title, location, description, requirements).

**Problem type**: Binary text classification (fraudulent = 1, genuine = 0)

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 98% |
| **F1-Score** (fraudulent class) | 0.77 |

## Methodology

```
Raw CSV → Text Preprocessing → TF-IDF Vectorization → Passive-Aggressive Classifier → Evaluation
```

### 1. Text Preprocessing (`src/model.py`)
- Combine title, location, description, and requirements into a single text field
- Convert to lowercase
- Remove stopwords (via Gensim), URLs, punctuation, numbers, and non-English characters
- Handle missing values with placeholder strings

### 2. Feature Engineering
- **TF-IDF Vectorization** with a vocabulary of 6,500 features
- Encoding: UTF-8 with error replacement

### 3. Class Balancing (optional)
- Minority class up-sampling using `sklearn.utils.resample`
- Configurable sampling ratio (disabled by default)

### 4. Model Training
- **Passive-Aggressive Classifier** with squared hinge loss
- Well-suited for large-scale text classification problems
- Trained on 80% of the data, tested on the remaining 20%

### 5. Evaluation (`src/evaluation.py`)
Custom evaluation class implementing from scratch:
- Confusion matrix computation
- Precision, Recall, F1-Score (macro, micro, weighted averages)
- AUC-ROC curve computation
- Also validated using `sklearn.metrics.classification_report`

## Project Structure

```
├── data/
│   └── job_train.csv          # Job posting dataset with labels
│
├── src/
│   ├── model.py               # ML pipeline: preprocessing, TF-IDF, PAC classifier
│   ├── evaluation.py          # Custom metrics: confusion matrix, precision, recall, F1, AUC
│   └── train.py               # Entry point: data loading, train/test split, evaluation
│
├── .gitignore
└── README.md
```

## Tech Stack

- **Language**: Python 3
- **ML**: scikit-learn (TF-IDF, Passive-Aggressive Classifier)
- **NLP**: Gensim (stopword removal)
- **Data**: pandas, NumPy

## Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/johnmelwin/Fake-Job-Prediction.git
   cd Fake-Job-Prediction
   ```

2. Install dependencies
   ```bash
   pip install pandas scikit-learn gensim numpy
   ```

3. Run the pipeline
   ```bash
   cd src
   python train.py
   ```
   This loads the dataset, trains the model, and prints the classification report with F1 score and runtime.
