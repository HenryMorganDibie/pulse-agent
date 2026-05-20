import pandas as pd


class AmazonLoader:
    def __init__(self, config):
        self.path = config["data"]["amazon_path"]   # amazon_reviews.jsonl

    def load(self, nrows=None):
        print(f"Loading Amazon data from: {self.path}")

        # Dataset is saved as JSONL via ds.to_json()
        df = pd.read_json(self.path, lines=True, nrows=nrows)

        print("Amazon columns:", df.columns.tolist())

        # Amazon Reviews 2023 actual field names
        # (McAuley-Lab/Amazon-Reviews-2023)
        df = df.rename(columns={
            "asin":      "item_id",
            "text":      "review_text",
            "rating":    "rating",
            "user_id":   "user_id",
            "timestamp": "timestamp",
        })

        df["source"] = "amazon"

        # Drop rows missing critical fields
        df = df.dropna(subset=["user_id", "item_id", "rating", "review_text"])

        # Timestamp is Unix milliseconds in this dataset — convert to datetime
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        print(f"Amazon loaded: {len(df):,} reviews")

        return df[["user_id", "item_id", "rating", "review_text", "timestamp", "source"]]