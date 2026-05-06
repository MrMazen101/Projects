import re
import emoji
from bs4 import BeautifulSoup
import spacy
import pandas as pd
from tqdm import tqdm  

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = self.nlp.Defaults.stop_words
        self.stopwords -= {"not", "no", "never", "n't", "cannot", "nothing"}

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        doc = self.nlp(text)
        clean_tokens = [
            token.lemma_ for token in doc 
            if token.text.strip() and token.text not in self.stopwords
        ]
        return " ".join(clean_tokens)

    def process_dataframe(self, df, text_column):
        print("Starting data preprocessing...")
        
        # 1. remove duplicates based on the text column to avoid redundant processing
        df = df.drop_duplicates(subset=[text_column]).copy()
        
        # 2. remove short reviews (less than 3 words) 
        df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
        df = df[df['word_count'] >= 3].copy()
        df = df.drop(columns=['word_count'])
        
        # 3. Clean the text and create a new column for it
        print("Applying Text Cleaning & Lemmatization...")
        tqdm.pandas(desc="Cleaning Reviews") # verfy tqdm is working with pandas
        df['clean_text'] = df[text_column].progress_apply(self.clean_text) # use progress_apply instead of apply to see the progress bar
        
        print("Preprocessing completed successfully!")
        return df