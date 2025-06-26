import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def extract_features():
    df = pd.read_csv("data/processed_tweets.csv")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_tweet"])

    # Save TF-IDF features
    pd.DataFrame(X.toarray()).to_csv("data/features.csv", index=False)

    # Save labels
    df["emotion"].to_csv("data/labels.csv", index=False)

    # Save vectorizer
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Features and labels saved.")

if __name__ == "__main__":
    extract_features()
