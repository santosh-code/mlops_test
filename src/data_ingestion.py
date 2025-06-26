import pandas as pd

def ingest():
    df = pd.read_csv("data/raw_tweets.csv")
    print("✅ Data ingested:", df.shape)
    print("This is new branch code")
    return df

if __name__ == "__main__":
    ingest()