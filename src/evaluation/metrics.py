"""
PulseAgent — Evaluation Metrics
Team HOKM · DSN × BCT Hackathon 3.0

All metric functions used to evaluate Task A (user modeling) and Task B (recommendation).
Imported by runner.py and used in the solution paper ablation studies.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Rating accuracy — Task A
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(mean_absolute_error(y_true, y_pred)), 4)


def user_bias_alignment(y_true: np.ndarray, y_pred: np.ndarray, user_avg: np.ndarray) -> float:
    """How well predictions capture each user's rating bias vs the platform average."""
    bias_true = y_true - user_avg
    bias_pred = y_pred - user_avg
    return round(float(np.mean(np.abs(bias_true - bias_pred))), 4)


# ---------------------------------------------------------------------------
# Text quality — Task A
# ---------------------------------------------------------------------------

def rouge_l(generated: list[str], references: list[str]) -> float:
    """Mean ROUGE-L F1 between generated and reference reviews."""
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, gen)["rougeL"].fmeasure
            for ref, gen in zip(references, generated)
            if ref and gen
        ]
        return round(float(np.mean(scores)), 4) if scores else 0.0
    except ImportError:
        print("[WARNING] rouge-score not installed. Run: pip install rouge-score")
        return 0.0


def bert_score(generated: list[str], references: list[str]) -> dict:
    """Mean BERTScore P/R/F1 between generated and reference reviews."""
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(generated, references, lang="en", verbose=False)
        return {
            "bertscore_precision": round(float(P.mean()), 4),
            "bertscore_recall":    round(float(R.mean()), 4),
            "bertscore_f1":        round(float(F1.mean()), 4),
        }
    except ImportError:
        print("[WARNING] bert-score not installed. Run: pip install bert-score")
        return {}


# ---------------------------------------------------------------------------
# Ranking quality — Task B
# ---------------------------------------------------------------------------

def _dcg(relevances: np.ndarray) -> float:
    gains = relevances / np.log2(np.arange(2, len(relevances) + 2))
    return float(gains.sum())


def ndcg_at_k(df, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Mean NDCG@k across users.
    Treats true ratings as relevance scores.
    """
    import pandas as pd

    if "user_id" not in df.columns:
        return 0.0

    df = df.copy().reset_index(drop=True)
    scores = []

    for _, group in df.groupby("user_id"):
        idx      = group.index
        true_rel = y_true[idx]
        pred_rel = y_pred[idx]

        pred_order = np.argsort(pred_rel)[::-1][:k]
        true_order = np.argsort(true_rel)[::-1][:k]

        dcg_val  = _dcg(true_rel[pred_order])
        idcg_val = _dcg(true_rel[true_order])

        scores.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)

    return round(float(np.mean(scores)), 4) if scores else 0.0


def hit_rate_at_k(df, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, threshold: float = 4.0) -> float:
    """
    Fraction of users with at least one truly relevant item (rating >= threshold)
    in the top-k predicted items.
    """
    if "user_id" not in df.columns:
        return 0.0

    df = df.copy().reset_index(drop=True)
    hits, total = 0, 0

    for _, group in df.groupby("user_id"):
        idx           = group.index
        true_rel      = y_true[idx]
        pred_rel      = y_pred[idx]
        top_k_idx     = np.argsort(pred_rel)[::-1][:k]
        relevant_mask = true_rel >= threshold

        if relevant_mask.sum() == 0:
            continue

        hits  += int(relevant_mask[top_k_idx].any())
        total += 1

    return round(hits / total, 4) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Cold-start metrics
# ---------------------------------------------------------------------------

def cold_start_mae(df, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    results = {}

    df = df.copy().reset_index(drop=True)

    if "is_cold_user" in df.columns:
        mask = df["is_cold_user"].values.astype(bool)
        if mask.sum() > 0:
            results["cold_user_mae"] = round(
                float(mean_absolute_error(y_true[mask], y_pred[mask])), 4
            )

    if "is_cold_item" in df.columns:
        mask = df["is_cold_item"].values.astype(bool)
        if mask.sum() > 0:
            results["cold_item_mae"] = round(
                float(mean_absolute_error(y_true[mask], y_pred[mask])), 4
            )

    return results