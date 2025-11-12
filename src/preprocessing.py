# ==============================================
# preprocessing.py
# Preprocesses Tweets.csv for Sentiment Analysis
# ==============================================

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ============================
# CONFIGURATION
# ============================

DATA_PATH = os.path.join("data", "Tweets.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MAX_WORDS = 20000       # Vocabulary size
MAX_LEN = 60            # Sequence length

# ============================
# CLEANING FUNCTION
# ============================

def clean_text(text):
    """Clean tweets by removing mentions, links, and special chars."""
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove links
    text = re.sub(r"@\w+", "", text)            # Remove mentions
    text = re.sub(r"[^A-Za-z\s]", "", text)     # Remove non-alphabetic
    text = re.sub(r"\s+", " ", text)            # Normalize whitespace
    return text.strip().lower()

# ============================
# MAIN PREPROCESSING FUNCTION
# ============================

def preprocess_dataset():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Expecting columns: "text", "airline_sentiment"
    if "text" not in df.columns or "airline_sentiment" not in df.columns:
        raise ValueError("Tweets.csv must contain 'text' and 'airline_sentiment' columns.")

    print(f"✅ Loaded {len(df)} samples.")

    # Clean text
    print("🧹 Cleaning text...")
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    # Tokenize text
    print("🔤 Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean_text"])
    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    # Encode labels
    print("🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["airline_sentiment"])
    y = to_categorical(y_encoded)

    print(f"🧠 Data shapes → X: {X.shape}, y: {y.shape}")

    # ============================
    # SAVE PROCESSED ARTIFACTS
    # ============================
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save tokenizer
    tokenizer_path = os.path.join(PROCESSED_DIR, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved → {tokenizer_path}")

    # Save dataset
    dataset_path = os.path.join(PROCESSED_DIR, "dataset.pkl")
    with open(dataset_path, "wb") as f:
        pickle.dump((X, y), f)
    print(f"✅ Dataset saved → {dataset_path}")

    print("\n🎯 Preprocessing complete!\n")
    return X, y, tokenizer


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    X, y, tokenizer = preprocess_dataset()
