import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from src import config

class ModelTrainer:
    def __init__(self):
        # Define the models to compare
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Naive Bayes": MultinomialNB(),
            "SVM": LinearSVC(max_iter=2000, random_state=42)
        }

    def train_all(self, X_train, y_train):
        """Train all defined models"""
        trained_models = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models

    def save_model(self, model, filename):
        """Save the model for later use in deployment"""
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        path = os.path.join(config.MODEL_DIR, filename)
        joblib.dump(model, path)
        print(f"Model saved successfully at: {path}")