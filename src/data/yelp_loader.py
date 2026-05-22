import json
import pandas as pd


class YelpLoader:

    def __init__(self, config):

        self.review_path = config["data"]["yelp_review_path"]

        # Optional business metadata
        self.business_path = config["data"].get("yelp_business_path")

    def load(self, nrows=None):

        print(f"Loading Yelp reviews from: {self.review_path}")

        # Yelp dataset is newline-delimited JSON
        records = []

        with open(self.review_path, "r", encoding="utf-8") as f:

            for i, line in enumerate(f):

                if nrows and i >= nrows:
                    break

                records.append(json.loads(line))

        df = pd.DataFrame(records)

        print("Yelp columns:", df.columns.tolist())

        # Normalize schema
        df = df.rename(columns={
            "business_id": "item_id",
            "stars": "rating",
            "text": "review_text",
            "date": "timestamp",
        })

        df["source"] = "yelp"

        # Drop rows missing critical fields
        df = df.dropna(
            subset=[
                "user_id",
                "item_id",
                "rating",
                "review_text",
            ]
        )

        print(f"Yelp loaded: {len(df):,} reviews")

        return df[
            [
                "user_id",
                "item_id",
                "rating",
                "review_text",
                "timestamp",
                "source",
            ]
        ]

    def load_business_metadata(self, nrows=None):
        """
        Optional Yelp business metadata loader.
        """

        if not self.business_path:

            print(
                "[WARNING] No Yelp business metadata path provided."
            )

            return pd.DataFrame()

        print(
            f"Loading Yelp business metadata from: "
            f"{self.business_path}"
        )

        records = []

        with open(
            self.business_path,
            "r",
            encoding="utf-8"
        ) as f:

            for i, line in enumerate(f):

                if nrows and i >= nrows:
                    break

                records.append(json.loads(line))

        biz_df = pd.DataFrame(records)

        biz_df = biz_df.rename(
            columns={"business_id": "item_id"}
        )

        return biz_df[
            [
                "item_id",
                "name",
                "categories",
                "stars",
                "review_count",
            ]
        ]