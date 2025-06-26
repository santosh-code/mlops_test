import pandas as pd
import re
import os

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def preprocess():
    print("📍 Running from:", os.getcwd())
    df = pd.read_csv("data/raw_tweets.csv")
    print("🧾 Columns in CSV:", df.columns.tolist())

    df["clean_tweet"] = df["text"].apply(clean_text)
    df.to_csv("data/processed_tweets.csv", index=False)
    print("✅ Preprocessed data saved at: data/processed_tweets.csv")

if __name__ == "__main__":
    preprocess()
