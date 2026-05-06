import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "IMDB Dataset.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "processed", "clean_reviews.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MAX_FEATURES = 5000  
NGRAM_RANGE = (1, 2) 