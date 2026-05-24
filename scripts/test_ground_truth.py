"""
Quick smoke test: verify Task B category-level ground truth
produces meaningful metrics instead of zeros/nulls.

Run:  python scripts/test_ground_truth.py
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import ndcg_at_k, hit_rate_at_k


# --- Fake test set: user reviewed items in Food (high) and Books (low) ---
test_df = pd.DataFrame([
    {"user_id": "u1", "item_id": "yelp_abc", "source": "Food",  "rating": 5.0, "review_text": "great"},
    {"user_id": "u1", "item_id": "yelp_def", "source": "Food",  "rating": 4.0, "review_text": "good"},
    {"user_id": "u1", "item_id": "amz_123",  "source": "Books", "rating": 2.0, "review_text": "meh"},
])

# --- Simulated recommendations (catalog IDs that DON'T exist in test set) ---
recs = [
    {"item_id": "gr_0",   "category": "Food",  "predicted_rating": 4.5},
    {"item_id": "i_201",  "category": "Books", "predicted_rating": 2.5},
    {"item_id": "ng_f01", "category": "Food",  "predicted_rating": 4.0},
]

# --- Apply the same category-level ground truth logic from runner.py ---
user_test = test_df[test_df["user_id"] == "u1"]
source_col = "source" if "source" in user_test.columns else "category"
cat_truth = user_test.groupby(source_col)["rating"].mean().to_dict()
user_avg = float(user_test["rating"].mean())

all_true = []
all_pred = []
rows = []

for rec in recs:
    rec_cat = rec["category"]
    true_rating = cat_truth.get(rec_cat, user_avg)
    all_true.append(float(true_rating))
    all_pred.append(rec["predicted_rating"])
    rows.append({"user_id": "u1", "item_id": rec["item_id"]})

y_true = np.array(all_true)
y_pred = np.array(all_pred)
df = pd.DataFrame(rows)

# --- Results ---
print("Category ground truth map:", cat_truth)
print()
print("Ground truth ratings:", y_true.tolist())
print("Predicted ratings:   ", y_pred.tolist())
print()

ndcg = ndcg_at_k(df, y_true, y_pred, k=10)
hit  = hit_rate_at_k(df, y_true, y_pred, k=10, threshold=3.5)

print(f"NDCG@10:    {ndcg}")
print(f"HitRate@10: {hit}")
print()

if ndcg > 0 and ndcg is not None:
    print("PASS - Ground truth mismatch is fixed. NDCG is non-zero.")
else:
    print("FAIL - Something is still wrong.")
