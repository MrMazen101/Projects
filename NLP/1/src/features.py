import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src import config

class FeatureExtractor:
    def __init__(self, max_features=config.MAX_FEATURES, ngram_range=config.NGRAM_RANGE):
        # استخدام TF-IDF لتحويل النصوص إلى أرقام
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

    def fit_transform(self, texts):
        """تدريب المُحوّل وتحويل النصوص إلى أرقام (يُستخدم مع بيانات التدريب)"""
        print("Extracting features using TF-IDF...")
        features = self.vectorizer.fit_transform(texts)
        return features

    def transform(self, texts):
        """تحويل النصوص فقط بناءً على ما تم تدريبه (يُستخدم مع بيانات الاختبار والنشر)"""
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, filename="tfidf_vectorizer.pkl"):
        """حفظ الـ Vectorizer لاستخدامه لاحقاً في واجهة الـ API"""
        import os
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        path = os.path.join(config.MODEL_DIR, filename)
        joblib.dump(self.vectorizer, path)
        print(f"Vectorizer saved at: {path}")

    def load_vectorizer(self, filename="tfidf_vectorizer.pkl"):
        """استدعاء الـ Vectorizer المحفوظ"""
        import os
        path = os.path.join(config.MODEL_DIR, filename)
        self.vectorizer = joblib.load(path)