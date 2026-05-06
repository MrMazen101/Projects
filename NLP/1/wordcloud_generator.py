import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src import config

def main():
    print("1. Loading Clean Data...")
    df = pd.read_csv(config.CLEAN_DATA_PATH).dropna(subset=['clean_text', 'target'])
    
    # Convert text to list of words for Word2Vec training
    sentences = [text.split() for text in df['clean_text']]
    
    print("2. Training Word2Vec Model (This might take a minute)...")
    # Training a Word2Vec model on the cleaned text data
    w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    
    def get_sentence_vector(words, model):
        """Convert a sentence to a vector by averaging the word vectors"""
        valid_words = [word for word in words if word in model.wv.key_to_index]
        if valid_words:
            return np.mean(model.wv[valid_words], axis=0)
        else:
            return np.zeros(model.vector_size)

    print("3. Converting text to Word2Vec Features...")
    X = np.array([get_sentence_vector(words, w2v_model) for words in sentences])
    y = df['target']
    
    print("4. Splitting and Training Logistic Regression...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n=== Word2Vec + Logistic Regression Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1-Score: {f1_score(y_test, y_pred) * 100:.2f}%")
    print("==============================================")

if __name__ == "__main__":
    main()