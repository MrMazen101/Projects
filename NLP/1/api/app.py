from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import sys

# adding the src directory to sys.path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.preprocessing import TextPreprocessor

# 1. initiate the FastAPI app
app = FastAPI(title="Sentiment Analysis API", description="API to predict if a review is Positive or Negative")

# 2. define the request model
class ReviewRequest(BaseModel):
    text: str

# 3. load the models and tools into memory (so they're ready and fast)
print("Loading Models and Tools...")
try:
    # load the model and vectorizer
    model_path = os.path.join(config.MODEL_DIR, "best_sentiment_model.pkl")
    vectorizer_path = os.path.join(config.MODEL_DIR, "tfidf_vectorizer.pkl")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # load the text preprocessor
    preprocessor = TextPreprocessor()
    print("Models Loaded Successfully! 🚀")
except Exception as e:
    print(f"Error loading models: {e}")

# 4. build the prediction endpoint
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty!")

    # clean the input text using the same preprocessing steps we used during training
    clean_text = preprocessor.clean_text(request.text)
    
    # convert the cleaned text to features using the same vectorizer we trained with
    features = vectorizer.transform([clean_text])
    
    # prediction
    prediction = model.predict(features)[0]
    
    # calculate confidence score if the model supports it
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = round(max(probabilities) * 100, 2)
    elif hasattr(model, "decision_function"):
        # convert decision function output to probability using sigmoid (for SVM)
        score = model.decision_function(features)[0]
        import math
        prob = 1 / (1 + math.exp(-score))
        confidence = round((prob if prediction == 1 else (1 - prob)) * 100, 2)

    # turn prediction into human-readable sentiment
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return {
        "original_text": request.text,
        "clean_text": clean_text,
        "sentiment": sentiment,
        "confidence": f"{confidence}%" if confidence else "N/A"
    }