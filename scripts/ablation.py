"""
PulseAgent — Ablation Study
Team HOKM · DSN × BCT Hackathon 3.0
"""

from __future__ import annotations

import asyncio
import json
import time
import numpy as np
import pandas as pd

from pathlib import Path
from unittest.mock import patch

from src.agents.graph import run_task_a, run_task_b
from src.evaluation.metrics import (
    rmse,
    mae,
    rouge_l,
    bert_score,
    ndcg_at_k,
    hit_rate_at_k,
)

OUTPUT_PATH = Path("data/ablation_results.json")

# ---------------------------------------------------------
# HACKATHON SAFE MODE (VERY IMPORTANT)
# ---------------------------------------------------------

SAMPLE_SIZE = 3  # 🔥 ultra-safe for TPD limit
REQUEST_DELAY_SECONDS = 4  # slower = safer
FAILURE_COOLDOWN_SECONDS = 20
MAX_RETRIES = 2

MAX_HISTORY = 2  # 🔥 reduces token usage massively
MAX_RECS = 5


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _build_persona(train_df: pd.DataFrame, user_id: str) -> dict:

    history = train_df[train_df["user_id"] == user_id]

    review_history = [
        {
            "item_id": str(row["item_id"]),
            "category": str(row.get("source", "unknown")),
            "rating": float(row["rating"]),
            "text": str(row["review_text"])[:120],  # 🔥 truncate text
            "timestamp": (
                str(row["timestamp"])
                if pd.notna(row["timestamp"])
                else None
            ),
        }
        for _, row in history.iterrows()
    ][:MAX_HISTORY]  # 🔥 limit history

    return {
        "user_id": user_id,
        "review_history": review_history,
        "avg_rating": (
            float(history["rating"].mean())
            if len(history) > 0
            else 3.0
        ),
        "preferred_categories": [],
        "tone_profile": "mixed",
        "conversation_history": [],
    }


# ---------------------------------------------------------
# SAFE WRAPPERS (IMPORTANT)
# ---------------------------------------------------------

async def _safe_call(fn, *args):

    for attempt in range(MAX_RETRIES):

        try:
            return await fn(*args)

        except Exception as e:

            msg = str(e)

            if "429" in msg or "rate_limit" in msg.lower():

                wait = FAILURE_COOLDOWN_SECONDS * (attempt + 1)

                print(f"    🔁 Rate limit → sleeping {wait}s")

                await asyncio.sleep(wait)
                continue

            print(f"    ❌ Failed: {e}")
            return None

    return None


# ---------------------------------------------------------
# TASK A
# ---------------------------------------------------------

async def _run_task_a_sample(test_df, train_df):

    sample = test_df.sample(
        n=min(SAMPLE_SIZE, len(test_df)),
        random_state=42,
    )

    y_true, y_pred = [], []
    generated, references = [], []

    for idx, (_, row) in enumerate(sample.iterrows(), 1):

        print(f"    Task A {idx}/{len(sample)}")

        persona = _build_persona(train_df, str(row["user_id"]))

        item = {
            "item_id": str(row["item_id"]),
            "name": str(row["item_id"]),
            "category": str(row.get("source", "unknown")),
            "attributes": {},
        }

        result = await _safe_call(run_task_a, persona, item)

        if result and result.get("simulated_rating") is not None:

            y_true.append(float(row["rating"]))
            y_pred.append(float(result["simulated_rating"]))

            if result.get("simulated_review"):

                generated.append(result["simulated_review"][:120])
                references.append(str(row["review_text"])[:120])

        await asyncio.sleep(REQUEST_DELAY_SECONDS)

    return np.array(y_true), np.array(y_pred), generated, references


# ---------------------------------------------------------
# TASK B
# ---------------------------------------------------------

async def _run_task_b_sample(test_df, train_df):

    user_ids = test_df["user_id"].unique()[:SAMPLE_SIZE]

    rows = []
    y_true_list, y_pred_list = [], []

    for idx, user_id in enumerate(user_ids, 1):

        print(f"    Task B {idx}/{len(user_ids)}")

        persona = _build_persona(train_df, str(user_id))

        result = await _safe_call(run_task_b, persona)

        if not result:
            continue

        recs = result.get("ranked_recommendations", [])[:MAX_RECS]

        user_test = test_df[test_df["user_id"] == user_id]

        for rec in recs:

            item_id = rec.get("item_id") if isinstance(rec, dict) else rec.item_id
            pred = rec.get("predicted_rating", 3.0) if isinstance(rec, dict) else rec.predicted_rating

            true_row = user_test[user_test["item_id"] == item_id]

            true = float(true_row["rating"].values[0]) if len(true_row) > 0 else 3.0

            y_true_list.append(true)
            y_pred_list.append(pred)

            rows.append({"user_id": user_id, "item_id": item_id})

        await asyncio.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(rows), np.array(y_true_list), np.array(y_pred_list)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def _task_a_metrics(y_true, y_pred, gen, ref):

    if len(y_true) == 0:
        return {"RMSE": None, "MAE": None}

    res = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }

    if gen:
        res["ROUGE-L"] = rouge_l(gen, ref)
        res["BERTScore-F1"] = bert_score(gen, ref).get("bertscore_f1")

    return res


def _task_b_metrics(df, yt, yp):

    if len(yt) == 0:
        return {"NDCG@10": None, "HitRate@10": None}

    return {
        "NDCG@10": ndcg_at_k(df, yt, yp, k=10),
        "HitRate@10": hit_rate_at_k(df, yt, yp, k=10),
    }


# ---------------------------------------------------------
# ABLATION RUNNER
# ---------------------------------------------------------

async def run_full(test_df, train_df):

    print("  Running FULL system...")

    a = await _run_task_a_sample(test_df, train_df)
    b = await _run_task_b_sample(test_df, train_df)

    return {
        "task_a": _task_a_metrics(*a),
        "task_b": _task_b_metrics(*b),
    }


ABLATION_CONFIGS = [
    ("Full system (baseline)", run_full),
]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

async def main():

    print("Loading data...")

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    start = time.time()

    results = {}

    for name, fn in ABLATION_CONFIGS:

        print(f"\n[{name}]")

        try:
            results[name] = await fn(test_df, train_df)

        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"FAILED: {e}")

    runtime = round(time.time() - start, 2)

    print("\n===== SUMMARY =====")
    print(json.dumps(results, indent=2))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "runtime_seconds": runtime,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"Runtime → {runtime}s")


if __name__ == "__main__":
    asyncio.run(main())