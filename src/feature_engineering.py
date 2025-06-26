import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse

def extract_features():
    df = pd.read_csv("data/processed_tweets.csv")
    vectorizer = TfidfVectorizer(max_features=5000)  # Optional: cap vocab size
    X = vectorizer.fit_transform(df["clean_tweet"])

    # Save features in sparse format
    sparse.save_npz("data/features.npz", X)

    # Save labels
    df["label"].to_csv("data/labels.csv", index=False)

    # Save vectorizer
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Features and labels saved (sparse matrix).")

if __name__ == "__main__":
    extract_features()
