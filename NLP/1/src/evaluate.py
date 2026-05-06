import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)

class ModelEvaluator:
    def __init__(self):
        self.results = [] # to store evaluation results for all models

    def evaluate(self, model_name, model, X_test, y_test):
        """Evaluate the model and store the results for reporting"""
        y_pred = model.predict(X_test)
        
        # some models (like SVM) don't have predict_proba, so we handle that case to still calculate ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }
        
        self.results.append(metrics)
        return y_pred, y_prob

    def generate_report(self):
        """create a DataFrame from the results and sort by F1-Score for ranking"""
        df_results = pd.DataFrame(self.results)
        # RANKING MODELS BY F1-Score 
        df_results = df_results.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
        return df_results

    def plot_confusion_matrix(self, y_test, y_pred, title="Confusion Matrix"):
        """DRAW CONFUSION MATRIX"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def plot_roc_curve(self, models_probs, y_test):
        """draw ROC curve for all models on the same plot for comparison"""
        plt.figure(figsize=(8, 6))
        for model_name, y_prob in models_probs.items():
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name}')
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.show()