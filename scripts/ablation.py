"""
PulseAgent — Ablation Study
Team HOKM · DSN × BCT Hackathon 3.0

Runs the full evaluation suite with one component disabled at a time.
Results are printed to terminal and saved to data/ablation_results.json.

Usage:
    python scripts/ablation.py

Requires:
    - data/processed/train.csv
    - data/processed/test.csv
    - GROQ_API_KEY set in environment
"""

from __future__ import annotations

import asyncio
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, AsyncMock

from src.agents.graph import run_task_a, run_task_b
from src.evaluation.metrics import (
    rmse, mae, rouge_l, bert_score,
    ndcg_at_k, hit_rate_at_k
)

OUTPUT_PATH = Path("data/ablation_results.json")
SAMPLE_SIZE = 50  # reduce if running slowly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_persona(train_df: pd.DataFrame, user_id: str) -> dict:
    history = train_df[train_df["user_id"] == user_id]
    return {
        "user_id": user_id,
        "review_history": [
            {
                "item_id":   str(row["item_id"]),
                "category":  str(row.get("source", "unknown")),
                "rating":    float(row["rating"]),
                "text":      str(row["review_text"]),
                "timestamp": str(row["timestamp"]) if pd.notna(row["timestamp"]) else None,
            }
            for _, row in history.iterrows()
        ],
        "avg_rating":           float(history["rating"].mean()) if len(history) > 0 else 3.0,
        "preferred_categories": [],
        "tone_profile":         "mixed",
        "conversation_history": [],
    }


async def _run_task_a_sample(test_df, train_df, sample_size=SAMPLE_SIZE):
    """Run Task A on a sample and return (y_true, y_pred, generated, references)."""
    sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    y_true, y_pred, generated, references = [], [], [], []

    for _, row in sample.iterrows():
        try:
            persona = _build_persona(train_df, str(row["user_id"]))
            item = {
                "item_id":    str(row["item_id"]),
                "name":       str(row["item_id"]),
                "category":   str(row.get("source", "unknown")),
                "attributes": {},
            }
            result = await run_task_a(persona, item)
            if result.get("simulated_rating") is not None:
                y_true.append(float(row["rating"]))
                y_pred.append(float(result["simulated_rating"]))
                if result.get("simulated_review"):
                    generated.append(result["simulated_review"])
                    references.append(str(row["review_text"]))
        except Exception as e:
            print(f"  [skip] {e}")

    return np.array(y_true), np.array(y_pred), generated, references


async def _run_task_b_sample(test_df, train_df, sample_size=SAMPLE_SIZE):
    """Run Task B on a sample and return (result_df, y_true, y_pred)."""
    user_ids = test_df["user_id"].unique()[:sample_size]
    rows, y_true_list, y_pred_list = [], [], []

    for user_id in user_ids:
        try:
            persona = _build_persona(train_df, str(user_id))
            result  = await run_task_b(persona)
            recs    = result.get("ranked_recommendations", [])
            user_test = test_df[test_df["user_id"] == user_id]

            for rec in recs:
                item_id = rec.get("item_id") if isinstance(rec, dict) else rec.item_id
                pred    = rec.get("predicted_rating", 3.0) if isinstance(rec, dict) else rec.predicted_rating
                true_row = user_test[user_test["item_id"] == item_id]
                true    = float(true_row["rating"].values[0]) if len(true_row) > 0 else 3.0

                y_true_list.append(true)
                y_pred_list.append(pred)
                rows.append({"user_id": user_id, "item_id": item_id})
        except Exception as e:
            print(f"  [skip] {e}")

    result_df = pd.DataFrame(rows)
    return result_df, np.array(y_true_list), np.array(y_pred_list)


def _task_a_metrics(y_true, y_pred, generated, references):
    if len(y_true) == 0:
        return {"RMSE": None, "MAE": None, "ROUGE-L": None, "BERTScore-F1": None}
    result = {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  mae(y_true, y_pred),
    }
    if generated:
        result["ROUGE-L"] = rouge_l(generated, references)
        bs = bert_score(generated, references)
        result["BERTScore-F1"] = bs.get("bertscore_f1")
    return result


def _task_b_metrics(result_df, y_true, y_pred):
    if len(y_true) == 0:
        return {"NDCG@10": None, "HitRate@10": None}
    return {
        "NDCG@10":    ndcg_at_k(result_df, y_true, y_pred, k=10),
        "HitRate@10": hit_rate_at_k(result_df, y_true, y_pred, k=10),
    }


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

async def run_full(test_df, train_df):
    """Baseline — full system, no components disabled."""
    print("  Running full system...")
    y_true, y_pred, gen, ref = await _run_task_a_sample(test_df, train_df)
    rdf, bt, bp = await _run_task_b_sample(test_df, train_df)
    return {
        "task_a": _task_a_metrics(y_true, y_pred, gen, ref),
        "task_b": _task_b_metrics(rdf, bt, bp),
    }


async def run_no_behavioural_pipeline(test_df, train_df):
    """Disable behavioural pipeline — persona uses default BehaviouralProfile."""
    print("  Disabling behavioural pipeline...")

    from src.schemas.models import BehaviouralProfile

    default_beh = BehaviouralProfile(
        avg_rating=3.0, rating_std=0.0, category_affinities={},
        recency_weighted_avg=3.0, rating_bias=0.0,
        is_harsh_rater=False, is_generous_rater=False,
    )

    async def _disabled(*args, **kwargs):
        return default_beh

    with patch("src.agents.persona_agent._behavioural_pipeline", new=_disabled):
        y_true, y_pred, gen, ref = await _run_task_a_sample(test_df, train_df)
        rdf, bt, bp = await _run_task_b_sample(test_df, train_df)

    return {
        "task_a": _task_a_metrics(y_true, y_pred, gen, ref),
        "task_b": _task_b_metrics(rdf, bt, bp),
    }


async def run_no_textual_pipeline(test_df, train_df):
    """Disable textual pipeline — persona uses default TextualProfile."""
    print("  Disabling textual pipeline...")

    from src.schemas.models import TextualProfile, ToneProfile

    default_txt = TextualProfile(
        dominant_tone=ToneProfile.MIXED, avg_review_length=0,
        sentiment_polarity=0.0, vocabulary_richness=0.0,
        uses_first_person=False, common_phrases=[],
    )

    async def _disabled(*args, **kwargs):
        return default_txt

    with patch("src.agents.persona_agent._textual_pipeline", new=_disabled):
        y_true, y_pred, gen, ref = await _run_task_a_sample(test_df, train_df)
        rdf, bt, bp = await _run_task_b_sample(test_df, train_df)

    return {
        "task_a": _task_a_metrics(y_true, y_pred, gen, ref),
        "task_b": _task_b_metrics(rdf, bt, bp),
    }


async def run_no_contextual_pipeline(test_df, train_df):
    """Disable contextual pipeline — persona uses default ContextualProfile."""
    print("  Disabling contextual pipeline...")

    from src.schemas.models import ContextualProfile

    default_ctx = ContextualProfile(
        is_cold_start=False, sparse_history=False,
        active_categories=[], cross_domain_signals=[], recency_days=None,
    )

    async def _disabled(*args, **kwargs):
        return default_ctx

    with patch("src.agents.persona_agent._contextual_pipeline", new=_disabled):
        y_true, y_pred, gen, ref = await _run_task_a_sample(test_df, train_df)
        rdf, bt, bp = await _run_task_b_sample(test_df, train_df)

    return {
        "task_a": _task_a_metrics(y_true, y_pred, gen, ref),
        "task_b": _task_b_metrics(rdf, bt, bp),
    }


async def run_no_chain_of_thought(test_df, train_df):
    """
    Disable chain-of-thought — skip rating inference step,
    use behavioural anchor directly as predicted rating.
    """
    print("  Disabling chain-of-thought (rating inference skipped)...")

    async def _no_cot(user_state, item):
        # Return behavioural anchor directly, no LLM call
        anchor = user_state.behavioural.recency_weighted_avg
        return round(min(5.0, max(1.0, anchor)), 1), ["No CoT — anchor used directly"]

    with patch("src.agents.review_agent._infer_rating", new=_no_cot):
        y_true, y_pred, gen, ref = await _run_task_a_sample(test_df, train_df)
        rdf, bt, bp = await _run_task_b_sample(test_df, train_df)

    return {
        "task_a": _task_a_metrics(y_true, y_pred, gen, ref),
        "task_b": _task_b_metrics(rdf, bt, bp),
    }


async def run_no_diversity_injection(test_df, train_df):
    """Disable diversity injection in RankingAgent."""
    print("  Disabling diversity injection...")

    def _no_diversity(scored_items, candidates_by_id, top_k=10):
        # Return top-k directly, no diversity reranking
        seen = set()
        deduped = []
        for item in scored_items:
            if item["item_id"] not in seen:
                seen.add(item["item_id"])
                deduped.append(item)
        return deduped[:top_k]

    with patch("src.agents.ranking_agent._inject_diversity", new=_no_diversity):
        y_true, y_pred, gen, ref = await _run_task_a_sample(test_df, train_df)
        rdf, bt, bp = await _run_task_b_sample(test_df, train_df)

    return {
        "task_a": _task_a_metrics(y_true, y_pred, gen, ref),
        "task_b": _task_b_metrics(rdf, bt, bp),
    }


async def run_no_intent_extraction(test_df, train_df):
    """Disable intent extraction — use empty intent string."""
    print("  Disabling intent extraction...")

    async def _no_intent(user_state, conversation_history):
        return "general recommendation", ["Intent extraction disabled"]

    with patch("src.agents.reasoning_agent._extract_intent", new=_no_intent):
        y_true, y_pred, gen, ref = await _run_task_a_sample(test_df, train_df)
        rdf, bt, bp = await _run_task_b_sample(test_df, train_df)

    return {
        "task_a": _task_a_metrics(y_true, y_pred, gen, ref),
        "task_b": _task_b_metrics(rdf, bt, bp),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    ("Full system (baseline)",          run_full),
    ("No behavioural pipeline",         run_no_behavioural_pipeline),
    ("No textual pipeline",             run_no_textual_pipeline),
    ("No contextual pipeline",          run_no_contextual_pipeline),
    ("No chain-of-thought prompting",   run_no_chain_of_thought),
    ("No diversity injection",          run_no_diversity_injection),
    ("No intent extraction",            run_no_intent_extraction),
]


async def main():
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df  = pd.read_csv("data/processed/test.csv")
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}\n")

    all_results = {}

    for name, fn in ABLATION_CONFIGS:
        print(f"[{name}]")
        try:
            result = await fn(test_df, train_df)
            all_results[name] = result
            print(f"  Task A — RMSE: {result['task_a'].get('RMSE')}  ROUGE-L: {result['task_a'].get('ROUGE-L')}  BERTScore: {result['task_a'].get('BERTScore-F1')}")
            print(f"  Task B — NDCG@10: {result['task_b'].get('NDCG@10')}  HitRate@10: {result['task_b'].get('HitRate@10')}")
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[name] = {"error": str(e)}
        print()

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<40} {'RMSE':>8} {'ROUGE-L':>10} {'NDCG@10':>10} {'HR@10':>8}")
    print("-" * 80)
    for name, result in all_results.items():
        ta = result.get("task_a", {})
        tb = result.get("task_b", {})
        print(
            f"{name:<40} "
            f"{str(ta.get('RMSE', '—')):>8} "
            f"{str(ta.get('ROUGE-L', '—')):>10} "
            f"{str(tb.get('NDCG@10', '—')):>10} "
            f"{str(tb.get('HitRate@10', '—')):>8}"
        )

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())