import pandas as pd


class GoodreadsLoader:
    def __init__(self, config):
        self.path = config["data"]["goodreads_path"]

    def load(self, nrows=None):
        print(f"Loading Goodreads data from: {self.path}")

        df = pd.read_json(self.path, lines=True, nrows=nrows)

        print("Goodreads columns:", df.columns.tolist())

        # Rename into unified item schema
        df = df.rename(columns={
            "Book": "item_name",
            "Author": "author",
            "Description": "description",
            "Genres": "genres",
            "Avg_Rating": "avg_rating",
            "Num_Ratings": "num_ratings",
            "URL": "url"
        })

        df["source"] = "goodreads"
        df["category"] = "Books"

        # Create synthetic item_id
        df["item_id"] = df.index.astype(str)

        # Clean ratings count
        df["num_ratings"] = (
            df["num_ratings"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )

        df["num_ratings"] = pd.to_numeric(
            df["num_ratings"],
            errors="coerce"
        )

        print(f"Goodreads loaded: {len(df):,} books")

        return df[
            [
                "item_id",
                "item_name",
                "author",
                "description",
                "genres",
                "avg_rating",
                "num_ratings",
                "category",
                "source",
                "url"
            ]
        ]