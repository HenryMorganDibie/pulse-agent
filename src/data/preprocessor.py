import pandas as pd

from src.data.yelp_loader import YelpLoader
from src.data.amazon_loader import AmazonLoader
from src.data.goodreads_loader import GoodreadsLoader


class UnifiedLoader:
    """
    Unified multi-domain dataset loader.

    Loads:
    - Yelp reviews
    - Amazon reviews
    - Goodreads reviews

    Returns one normalized dataframe shared across the system.
    """

    def __init__(self, config):
        self.yelp_loader = YelpLoader(config)
        self.amazon_loader = AmazonLoader(config)
        self.goodreads_loader = GoodreadsLoader(config)

    def load_all(self, nrows=None):
        """
        Load and merge all datasets into one unified dataframe.

        Args:
            nrows (int | None):
                Optional limit for debugging / fast testing.

        Returns:
            pd.DataFrame
        """

        datasets = []

        loaders = {
            "Yelp": self.yelp_loader,
            "Amazon": self.amazon_loader,
            "Goodreads": self.goodreads_loader,
        }

        for name, loader in loaders.items():

            try:
                print(f"\nLoading {name} dataset...")

                df = loader.load(nrows=nrows)

                if df.empty:
                    print(f"[WARNING] {name} returned an empty dataframe.")
                    continue

                datasets.append(df)

                print(f"{name} loaded successfully")
                print(f"Rows: {len(df):,}")

            except Exception as e:
                print(f"[WARNING] Failed to load {name}: {e}")

        if not datasets:
            raise RuntimeError(
                "All dataset loaders failed. No data available."
            )

        # Merge datasets
        unified_df = pd.concat(datasets, ignore_index=True)

        # Remove duplicates
        unified_df = unified_df.drop_duplicates()

        # Reset index
        unified_df = unified_df.reset_index(drop=True)

        # Basic cleanup
        unified_df["review_text"] = (
            unified_df["review_text"]
            .astype(str)
            .str.strip()
        )

        print("\nUnified Dataset Summary")
        print("-" * 40)
        print(f"Total Rows: {len(unified_df):,}")
        print(f"Columns: {list(unified_df.columns)}")
        print("\nSource Distribution:")
        print(unified_df["source"].value_counts())

        return unified_df