import streamlit as st
from transformers import pipeline
import os

# 1. Streamlit page configuration
st.set_page_config(page_title="BERT Sentiment Pro", page_icon="🤖", layout="centered")

# 2. Utility function to load the BERT model (with caching to speed up subsequent loads)
# Using Streamlit's caching to avoid reloading the model on every interaction
@st.cache_resource
def load_bert_model():
    st.info("Loading BERT model... please wait (First time may take a minute)")
    # Load the pre-trained BERT model for sentiment analysis
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

analyzer = load_bert_model()

# 3.user interface
st.title("🤖 BERT Sentiment Analyzer")
st.markdown("""
Welcome to the BERT Sentiment Analyzer! This app uses a powerful pre-trained BERT model to analyze the sentiment of your reviews. Just type in your review and see if it's positive or negative, along with the confidence score!
""")

user_input = st.text_area("Enter your review:", placeholder="Type something like: 'I was skeptical, but now I love it!'")

if st.button("Analyze with BERT 🚀", type="primary"):
    if user_input.strip():
        with st.spinner("BERT is thinking..."):
            # analyze the input text using BERT
            result = analyzer(user_input)[0]
            
            label = result['label']
            score = result['score']
            
            st.markdown("---")
            
            # Show the result with confidence score
            if label == "POSITIVE":
                st.success(f"### Result: Positive 😊 (Confidence: {score*100:.2f}%)")
                st.balloons()
            else:
                st.error(f"### Result: Negative 😞 (Confidence: {score*100:.2f}%)")
            
            # Explain to the user why BERT is more accurate
            st.info("💡 **Why is this accurate?** BERT reads the whole sentence at once to understand context, unlike traditional models.")
    else:
        st.warning("Please enter some text first.")