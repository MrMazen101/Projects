# 🎭 Product Review Sentiment Analysis Suite

A comprehensive End-to-End NLP project designed to analyze product reviews and classify them as **Positive** or **Negative**. This project covers everything from classical Machine Learning to state-of-the-art Deep Learning (Transformers).

---

## 🚀 Overview

This suite provides a full pipeline for Sentiment Analysis, including:

- **Data Preprocessing:** Cleaning HTML tags, URLs, Emojis, and performing Lemmatization using `spaCy`.
- **Baseline Models:** Comparison between Logistic Regression, Naive Bayes, and SVM.
- **Advanced Experiments:** Word Embeddings (Word2Vec) and Contextual Embeddings (BERT).
- **Deployment:** A professional Web App using `Streamlit` and a production-ready API using `FastAPI`.

---

## 📁 Project Architecture

```text
sentiment_analysis_project/
│
├── data/
│   ├── raw/                # Original IMDB Dataset.csv
│   └── processed/          # Preprocessed clean_reviews.csv
│
├── src/
│   ├── config.py           # Paths and global configurations
│   ├── preprocessing.py    # Text cleaning logic (spaCy & Emoji handling)
│   ├── features.py         # TF-IDF Vectorization logic
│   ├── train.py            # Model training scripts
│   └── evaluate.py         # Evaluation metrics and visualizations
│
├── models/                 # Saved serialized models (.pkl)
├── api/                    # FastAPI source code
├── streamlit_app.py        # Streamlit Web Application
├── main.py                 # Main execution pipeline
└── requirements.txt        # Python dependencies
```

---

## 🛠️ Installation & Setup

Clone the project and install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Run the Training Pipeline:**

Place your `IMDB Dataset.csv` in `data/raw/` and execute:

```bash
python main.py
```

**Launch the Web Dashboard (Streamlit):**

```bash
streamlit run streamlit_app.py
```

**Run the API (FastAPI):**

```bash
uvicorn api.app:app --reload
```

---

## 📊 Performance Report

Our models were evaluated on **50,000 IMDB reviews** with the following results:

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 88.80% | 88.99% | 95.74% |
| SVM (LinearSVC) | 88.40% | 88.50% | 95.42% |
| Naive Bayes | 85.47% | 85.78% | 93.10% |

> **Note:** BERT (DistilBERT) was also integrated, providing superior context understanding for complex sentences.

---

## 🧠 Key Features

- **Context-Aware Cleaning:** Negations (like "not", "never") are preserved to maintain sentiment accuracy.
- **Emoji Support:** Emojis are converted into text tokens to leverage emotional cues.
- **Scalability:** Modular code structure allows for easy swapping of models and feature extractors.
- **Interactive UI:** Users can test their own reviews through a modern, responsive interface.

---

## 🧪 Tech Stack

| Category | Tools |
|---|---|
| NLP | spaCy, Transformers (Hugging Face) |
| ML Core | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Deployment | FastAPI, Streamlit, Uvicorn |

---

*Developed with ❤️ as a comprehensive NLP Portfolio Project.*
