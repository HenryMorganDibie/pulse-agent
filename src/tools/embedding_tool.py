"""
PulseAgent — EmbeddingTool
Team HOKM · DSN × BCT Hackathon 3.0

Sentence embedding utility used by the ReasoningAgent for semantic similarity.

Tries to use sentence-transformers if available; falls back to a lightweight
TF-IDF-style bag-of-words approach so the system runs without GPU dependencies.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend selection — sentence-transformers is optional
# ---------------------------------------------------------------------------

_model = None
_backend: str = "bow"  # "sbert" | "bow"


def _load_sbert() -> bool:
    """Attempt to load a lightweight sentence-transformers model."""
    global _model, _backend
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _backend = "sbert"
        logger.info("EmbeddingTool: using sentence-transformers backend (all-MiniLM-L6-v2)")
        return True
    except Exception as exc:
        logger.info("EmbeddingTool: sentence-transformers unavailable (%s) — using BoW fallback", exc)
        _backend = "bow"
        return False


# Attempt to load at import time; non-fatal if it fails
_load_sbert()


# ---------------------------------------------------------------------------
# BoW fallback
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    import re
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def _bow_embed(text: str, vocab: Optional[List[str]] = None) -> List[float]:
    """
    Bag-of-words TF vector. If vocab is None, uses the text's own tokens.
    Returns a unit-normalized float list.
    """
    tokens = _tokenize(text)
    counts = Counter(tokens)
    keys = vocab if vocab else list(counts.keys())

    vec = [float(counts.get(k, 0)) for k in keys]
    return _normalize(vec)


def _normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(text: str) -> List[float]:
    """
    Embed a single text string into a float vector.

    Uses sentence-transformers if available, otherwise BoW fallback.
    The vector is always L2-normalized.

    Args:
        text: Any string (query, item description, user intent, etc.)

    Returns:
        List[float] — the embedding vector.
    """
    if _backend == "sbert" and _model is not None:
        try:
            vec = _model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        except Exception as exc:
            logger.warning("EmbeddingTool: sbert encode failed (%s), falling back to BoW", exc)

    return _bow_embed(text)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two vectors.
    Both must be the same length; returns 0.0 if either is zero.

    When using the BoW backend on texts of different vocabularies, embed both
    texts with embed_pair() instead to ensure a shared vocabulary.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Vectors from embed() are already L2-normalized, so dot == cosine sim
    return max(-1.0, min(1.0, dot))


def embed_pair(text_a: str, text_b: str) -> tuple[List[float], List[float]]:
    """
    Embed two texts against a shared vocabulary (important for BoW backend).
    For sbert backend this is equivalent to calling embed() twice.

    Returns:
        Tuple of (vec_a, vec_b) suitable for cosine_similarity().
    """
    if _backend == "sbert" and _model is not None:
        try:
            vecs = _model.encode([text_a, text_b], normalize_embeddings=True)
            return vecs[0].tolist(), vecs[1].tolist()
        except Exception:
            pass

    # BoW — build shared vocab
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    vocab = list(set(tokens_a + tokens_b))

    return _bow_embed(text_a, vocab), _bow_embed(text_b, vocab)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Convenience function — embed two texts and return their cosine similarity.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        float in [-1.0, 1.0]; higher means more similar.
    """
    vec_a, vec_b = embed_pair(text_a, text_b)
    return cosine_similarity(vec_a, vec_b)


def rank_by_similarity(query: str, texts: List[str]) -> List[tuple[int, float]]:
    """
    Rank a list of texts by semantic similarity to a query.

    Args:
        query: The reference text.
        texts: Candidate texts to rank.

    Returns:
        List of (original_index, similarity_score) sorted descending by score.
    """
    if not texts:
        return []

    if _backend == "sbert" and _model is not None:
        try:
            all_texts = [query] + texts
            vecs = _model.encode(all_texts, normalize_embeddings=True)
            q_vec = vecs[0]
            scores = [(i, float(sum(q_vec * vecs[i + 1]))) for i in range(len(texts))]
            return sorted(scores, key=lambda x: x[1], reverse=True)
        except Exception:
            pass

    # BoW fallback — shared vocab across all texts
    all_text = " ".join([query] + texts)
    tokens_all = _tokenize(all_text)
    vocab = list(set(tokens_all))

    q_vec = _bow_embed(query, vocab)
    scores = []
    for i, text in enumerate(texts):
        t_vec = _bow_embed(text, vocab)
        scores.append((i, cosine_similarity(q_vec, t_vec)))

    return sorted(scores, key=lambda x: x[1], reverse=True)


def get_backend() -> str:
    """Return which backend is active: 'sbert' or 'bow'."""
    return _backend
