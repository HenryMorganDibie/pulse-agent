"""
PulseAgent — Evaluation Runner
Team HOKM · DSN × BCT Hackathon 3.0
"""

from __future__ import annotations

import asyncio
import json
import time
import numpy as np
import pandas as pd

from pathlib import Path

from src.evaluation.metrics import (
    rmse,
    mae,
    user_bias_alignment,
    rouge_l,
    bert_score,
    ndcg_at_k,
    hit_rate_at_k,
    cold_start_mae,
)

from src.agents.graph import run_task_a, run_task_b


OUTPUT_PATH = Path("data/eval_results.json")

# Groq free tier safety buffer
REQUEST_DELAY_SECONDS = 2.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_persona_from_row(
    train_df: pd.DataFrame,
    user_id: str,
) -> dict:
    """Build a UserPersona dict from a user's training history."""

    history = train_df[train_df["user_id"] == user_id]

    review_history = [
        {
            "item_id": str(row["item_id"]),
            "category": str(row.get("source", "unknown")),
            "rating": float(row["rating"]),
            "text": str(row["review_text"]),
            "timestamp": (
                str(row["timestamp"])
                if pd.notna(row["timestamp"])
                else None
            ),
        }
        for _, row in history.iterrows()
    ]

    avg_rating = (
        float(history["rating"].mean())
        if len(history) > 0
        else 3.0
    )

    return {
        "user_id": user_id,
        "review_history": review_history,
        "avg_rating": avg_rating,
        "preferred_categories": [],
        "tone_profile": "mixed",
        "conversation_history": [],
    }


# ---------------------------------------------------------------------------
# Task A evaluation
# ---------------------------------------------------------------------------

async def evaluate_task_a(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    sample_size: int = 10,
) -> dict:

    print(f"\n[Task A] Evaluating on {sample_size} samples...")

    sample = test_df.sample(
        n=min(sample_size, len(test_df)),
        random_state=42,
    )

    y_true = []
    y_pred = []

    generated_reviews = []
    reference_reviews = []
    user_avgs = []

    for idx, row in enumerate(sample.iterrows(), start=1):

        _, row = row

        try:

            print(
                f"[Task A] Processing "
                f"{idx}/{len(sample)}"
            )

            persona = _build_persona_from_row(
                train_df,
                str(row["user_id"]),
            )

            item = {
                "item_id": str(row["item_id"]),
                "name": str(row["item_id"]),
                "category": str(
                    row.get("source", "unknown")
                ),
                "attributes": {},
            }

            # ---------------------------------------------------
            # Retry-safe Task A execution
            # ---------------------------------------------------

            result = None

            for retry in range(3):

                try:

                    result = await run_task_a(
                        persona,
                        item,
                    )

                    break

                except Exception as e:

                    if "429" in str(e):

                        wait_time = (retry + 1) * 3

                        print(
                            f"[RATE LIMIT] "
                            f"Retrying in "
                            f"{wait_time}s..."
                        )

                        await asyncio.sleep(wait_time)

                        continue

                    raise e

            if result is None:
                continue

            # ---------------------------------------------------

            if result.get("simulated_rating") is not None:

                y_true.append(float(row["rating"]))

                y_pred.append(
                    float(result["simulated_rating"])
                )

                user_avgs.append(
                    float(persona["avg_rating"])
                )

                if result.get("simulated_review"):

                    generated_reviews.append(
                        result["simulated_review"]
                    )

                    reference_reviews.append(
                        str(row["review_text"])
                    )

            # Groq RPM protection
            await asyncio.sleep(
                REQUEST_DELAY_SECONDS
            )

        except Exception as e:

            print(
                f"[WARNING] Task A failed "
                f"for user {row['user_id']}: {e}"
            )

            await asyncio.sleep(5)

            continue

    if not y_true:
        return {
            "error":
            "No Task A predictions generated"
        }

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    user_avg_arr = np.array(user_avgs)

    results = {
        "task": "A",
        "n": len(y_true),

        "RMSE": rmse(
            y_true_arr,
            y_pred_arr,
        ),

        "MAE": mae(
            y_true_arr,
            y_pred_arr,
        ),

        "user_bias_alignment":
        user_bias_alignment(
            y_true_arr,
            y_pred_arr,
            user_avg_arr,
        ),
    }

    if generated_reviews:

        results["ROUGE-L"] = rouge_l(
            generated_reviews,
            reference_reviews,
        )

        results.update(
            bert_score(
                generated_reviews,
                reference_reviews,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Task B evaluation
# ---------------------------------------------------------------------------

async def evaluate_task_b(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    sample_size: int = 10,
    k: int = 10,
) -> dict:

    print(f"\n[Task B] Evaluating on {sample_size} users...")

    user_ids = test_df["user_id"].unique()[:sample_size]

    all_true = []
    all_pred = []
    result_df_rows = []

    for idx, user_id in enumerate(user_ids, start=1):

        try:

            print(
                f"[Task B] Processing "
                f"{idx}/{len(user_ids)}"
            )

            persona = _build_persona_from_row(
                train_df,
                str(user_id),
            )

            # ---------------------------------------------------
            # Retry-safe Task B execution
            # ---------------------------------------------------

            result = None

            for retry in range(3):

                try:

                    result = await run_task_b(
                        persona
                    )

                    break

                except Exception as e:

                    if "429" in str(e):

                        wait_time = (retry + 1) * 3

                        print(
                            f"[RATE LIMIT] "
                            f"Retrying in "
                            f"{wait_time}s..."
                        )

                        await asyncio.sleep(wait_time)

                        continue

                    raise e

            if result is None:
                continue

            # ---------------------------------------------------

            recs = result.get(
                "ranked_recommendations",
                [],
            )

            if not recs:
                continue

            user_test = test_df[
                test_df["user_id"] == user_id
            ]

            # Build category-level ground truth from the user's test items.
            # Catalog item_ids (gr_0, i_001, …) differ from dataset item_ids,
            # so we match on category instead: the user's avg rating in a
            # category is the relevance signal for recommended items in that
            # category.
            source_col = (
                "source" if "source" in user_test.columns
                else "category"
            )
            cat_truth = (
                user_test
                .groupby(source_col)["rating"]
                .mean()
                .to_dict()
            )

            for rec in recs:

                item_id = rec.get("item_id")
                rec_category = rec.get("category", "")

                pred_rating = rec.get(
                    "predicted_rating",
                    3.0,
                )

                # Use the user's average rating in this category as
                # ground truth.  Falls back to overall test-set avg.
                true_rating = cat_truth.get(
                    rec_category,
                    float(user_test["rating"].mean())
                    if len(user_test) > 0
                    else 3.0,
                )

                all_true.append(float(true_rating))
                all_pred.append(pred_rating)

                result_df_rows.append({
                    "user_id": user_id,
                    "item_id": item_id,
                })

            # Groq RPM protection
            await asyncio.sleep(
                REQUEST_DELAY_SECONDS
            )

        except Exception as e:

            print(
                f"[WARNING] Task B failed "
                f"for user {user_id}: {e}"
            )

            await asyncio.sleep(5)

            continue

    if not all_true:
        return {
            "error":
            "No Task B predictions generated"
        }

    y_true_arr = np.array(all_true)
    y_pred_arr = np.array(all_pred)

    result_df = pd.DataFrame(result_df_rows)

    results = {
        "task": "B",

        "n_users": len(user_ids),

        "NDCG@10": ndcg_at_k(
            result_df,
            y_true_arr,
            y_pred_arr,
            k=k,
        ),

        "HitRate@10": hit_rate_at_k(
            result_df,
            y_true_arr,
            y_pred_arr,
            k=k,
        ),

        "RMSE": rmse(
            y_true_arr,
            y_pred_arr,
        ),
    }

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_evaluation(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    sample_size: int = 10,
) -> None:

    print("Loading processed data...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(
        f"Train: {len(train_df):,} rows | "
        f"Test: {len(test_df):,} rows"
    )

    start_time = time.time()

    task_a_results = await evaluate_task_a(
        test_df,
        train_df,
        sample_size,
    )

    task_b_results = await evaluate_task_b(
        test_df,
        train_df,
        sample_size,
    )

    total_time = round(
        time.time() - start_time,
        2,
    )

    all_results = {
        "task_a": task_a_results,
        "task_b": task_b_results,
        "runtime_seconds": total_time,
    }

    print("\n--- TASK A RESULTS ---")

    for k, v in task_a_results.items():
        print(f"  {k}: {v}")

    print("\n--- TASK B RESULTS ---")

    for k, v in task_b_results.items():
        print(f"  {k}: {v}")

    print(f"\nTotal runtime: {total_time}s")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(run_evaluation())