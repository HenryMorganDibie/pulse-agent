from datasets import load_dataset

ds = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_All_Beauty",
    split="full",
    trust_remote_code=True
)

ds.to_json("data/amazon_reviews.jsonl")
print("Saved!")