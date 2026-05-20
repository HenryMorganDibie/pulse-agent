import json
import pandas as pd
from pathlib import Path


class YelpLoader:
    def __init__(self, config):
        self.review_path = config["data"]["yelp_review_path"]      # yelp_academic_dataset_review.json
        self.business_path = config["data"]["yelp_business_path"]  # yelp_academic_dataset_business.json

    def load(self, nrows=None):
        print(f"Loading Yelp reviews from: {self.review_path}")

        # Yelp dataset is newline-delimited JSON (one JSON object per line)
        records = []
        with open(self.review_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if nrows and i >= nrows:
                    break
                records.append(json.loads(line))

        df = pd.DataFrame(records)

        print("Yelp columns:", df.columns.tolist())

        # Yelp actual field names
        df = df.rename(columns={
            "business_id": "item_id",
            "stars":        "rating",
            "text":         "review_text",
            "date":         "timestamp",
        })

        df["source"] = "yelp"

        # Drop rows missing critical fields
        df = df.dropna(subset=["user_id", "item_id", "rating", "review_text"])

        print(f"Yelp loaded: {len(df):,} reviews")

        return df[["user_id", "item_id", "rating", "review_text", "timestamp", "source"]]

    def load_business_metadata(self, nrows=None):
        """Optional — loads business metadata for item attributes."""
        print(f"Loading Yelp business metadata from: {self.business_path}")

        records = []
        with open(self.business_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if nrows and i >= nrows:
                    break
                records.append(json.loads(line))

        biz_df = pd.DataFrame(records)
        biz_df = biz_df.rename(columns={"business_id": "item_id"})

        return biz_df[["item_id", "name", "categories", "stars", "review_count"]]