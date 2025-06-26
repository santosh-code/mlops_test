import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import json
from scipy import sparse

def train():
    # Load sparse features
    X = sparse.load_npz("data/features.npz")

    # Load labels
    y = pd.read_csv("data/labels.csv")["label"]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save accuracy as metric
    accuracy = model.score(X, y)
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

    print(f"✅ Model trained — Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    train()
