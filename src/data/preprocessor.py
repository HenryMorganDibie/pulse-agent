import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.yelp_loader import YelpLoader
from src.data.amazon_loader import AmazonLoader
from src.data.goodreads_loader import GoodreadsLoader
from src.utils.config import load_config


class UnifiedLoader:
    """
    Unified multi-domain dataset loader (Option C).

    Loads:
    - Yelp reviews
    - Amazon reviews
    - Goodreads reviews

    Applies per-source caps to ensure balanced dataset.
    """

    def __init__(self, config):
        self.config = config
        self.yelp_loader = YelpLoader(config)
        self.amazon_loader = AmazonLoader(config)
        self.goodreads_loader = GoodreadsLoader(config)

    def load_all(
        self,
        amazon_cap=80000,
        yelp_cap=30000,
        goodreads_cap=30000,
        nrows=None,
    ):
        """
        Load, cap, and merge datasets (Option C balanced strategy).
        """

        datasets = []

        loaders = {
            "Yelp": (self.yelp_loader, yelp_cap),
            "Amazon": (self.amazon_loader, amazon_cap),
            "Goodreads": (self.goodreads_loader, goodreads_cap),
        }

        for name, (loader, cap) in loaders.items():

            try:
                print(f"\nLoading {name} dataset...")

                df = loader.load(nrows=nrows)

                if df is None or df.empty:
                    print(f"[WARNING] {name} returned empty dataframe.")
                    continue

                # OPTION C: cap dataset size
                if len(df) > cap:
                    df = df.sample(cap, random_state=42)

                datasets.append(df)

                print(f"{name} loaded successfully")
                print(f"Rows: {len(df):,}")

            except Exception as e:
                print(f"[WARNING] Failed to load {name}: {e}")

        if not datasets:
            raise RuntimeError("All dataset loaders failed. No data available.")

        # Merge datasets
        unified_df = pd.concat(datasets, ignore_index=True)

        # Remove duplicates
        unified_df = unified_df.drop_duplicates()

        # Reset index
        unified_df = unified_df.reset_index(drop=True)

        # Clean text
        if "review_text" in unified_df.columns:
            unified_df["review_text"] = (
                unified_df["review_text"]
                .astype(str)
                .str.strip()
            )

        print("\nUnified Dataset Summary")
        print("-" * 40)
        print(f"Total Rows: {len(unified_df):,}")

        print("\nSource Distribution:")
        print(unified_df["source"].value_counts())

        return unified_df


def build_processed_datasets(
    config,
    test_size=0.2,
    random_state=42,
    nrows=None,
):
    """
    Build train/test datasets for evaluation.
    """

    print("\nBuilding processed datasets (Option C)...")

    loader = UnifiedLoader(config)

    df = loader.load_all(nrows=nrows)

    # Stratified split to preserve domain balance
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=df["source"],
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nProcessed datasets saved successfully")
    print("-" * 40)
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")

    return train_df, test_df


if __name__ == "__main__":

    config = load_config()
    build_processed_datasets(config)