from transformers import pipeline

def main():
    print("Loading Pre-trained BERT Model... (This will download a ~250MB model for the first time)")
    # Load the pre-trained BERT model for sentiment analysis
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    print("\nModel Loaded! Let's test it.\n")
    
    # Some example reviews to test the BERT model on
    reviews_to_test = [
        "in the first scene i fall in love with the movie after the middle of i hate it because it has more sexual scene",
        "This is the worst experience I have ever had, totally disappointed and wasting my money.",
        "It's okay, not bad but also not great. I might buy it again.",
        "I was skeptical at first, but honestly, it is absolutely amazing 😡 -> 😍"
    ]
    
    # Sending the reviews to the model
    results = sentiment_analyzer(reviews_to_test)
    
    print("=== BERT Predictions ===")
    for review, result in zip(reviews_to_test, results):
        print(f"Review: '{review}'")
        print(f"Prediction: {result['label']} (Confidence: {result['score'] * 100:.2f}%)")
        print("-" * 50)

if __name__ == "__main__":
    main()