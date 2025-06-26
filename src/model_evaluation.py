import pandas as pd
import pickle
from scipy import sparse

def evaluate():
    # Load model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load features
    X = sparse.load_npz("data/features.npz")

    # Predict
    predictions = model.predict(X)

    print("✅ Predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"{i+1}. {pred}")

if __name__ == "__main__":
    evaluate()
