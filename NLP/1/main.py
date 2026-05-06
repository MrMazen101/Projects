import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.preprocessing import TextPreprocessor
from src.features import FeatureExtractor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src import config

def main():
    # 0. CONFIGURATION: Ensure necessary directories exist
    os.makedirs(os.path.dirname(config.CLEAN_DATA_PATH), exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. CONFIGURATION: Check if clean data exists
    if not os.path.exists(config.CLEAN_DATA_PATH):
        print("Clean data not found! Starting preprocessing pipeline...")
        # Read raw data
        df_raw = pd.read_csv(config.RAW_DATA_PATH)
        
        # Clean the data using our TextPreprocessor
        preprocessor = TextPreprocessor()
        df = preprocessor.process_dataframe(df_raw, text_column="review")
        
        # Convert sentiments (positive/negative) to numbers (1/0)
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['sentiment'])
        
        # Save the cleaned data
        df.to_csv(config.CLEAN_DATA_PATH, index=False)
        print(f"Cleaned data saved successfully at: {config.CLEAN_DATA_PATH}")
    else:
        print("1. Loading existing Cleaned Data...")
        df = pd.read_csv(config.CLEAN_DATA_PATH)

    # 2. Cleaned data might have some missing values after preprocessing, so we drop those rows
    df = df.dropna(subset=['clean_text', 'target'])

    # لضمان السرعة في التجربة الحالية (يمكنك مسح السطر القادم لتدريب الـ 50,000 كاملة)
    # df = df.sample(10000, random_state=42).reset_index(drop=True)

    print("2. Splitting Data (80% Train, 20% Test)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
    )

    print("3. Extracting Features (TF-IDF)...")
    extractor = FeatureExtractor()
    X_train = extractor.fit_transform(X_train_text)
    X_test = extractor.transform(X_test_text)
    extractor.save_vectorizer() # Save the vectorizer for later use in the API

    print("4. Training Models...")
    trainer = ModelTrainer()
    trained_models = trainer.train_all(X_train, y_train)

    print("5. Evaluating Models...")
    evaluator = ModelEvaluator()
    best_f1 = 0
    best_model_name = ""
    best_model = None

    for name, model in trained_models.items():
        y_pred, y_prob = evaluator.evaluate(name, model, X_test, y_test)
        
        # Track the best model based on F1-Score
        current_f1 = evaluator.results[-1]["F1-Score"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_name = name
            best_model = model

    print("\n================ MODEL COMPARISON REPORT ================")
    report_df = evaluator.generate_report()
    print(report_df.to_string(index=False))
    print("=========================================================\n")

    print(f"6. Saving the Best Model: {best_model_name}...")
    trainer.save_model(best_model, filename="best_sentiment_model.pkl")

    print("Pipeline finished successfully! \U0001F389")

if __name__ == "__main__":
    main()