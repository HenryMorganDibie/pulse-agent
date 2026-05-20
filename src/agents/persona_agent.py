"""
PulseAgent — PersonaConstructionAgent (LangGraph Node 1)
Team HOKM · DSN × BCT Hackathon 3.0

Three concurrent async pipelines — behavioural, textual, contextual — each
fails independently. Results are merged into a typed UserState that both
Task A and Task B agents consume downstream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import httpx
from src.schemas.models import (
    AgentState,
    BehaviouralProfile,
    ContextualProfile,
    TextualProfile,
    ToneProfile,
    UserPersona,
    UserState,
)

logger = logging.getLogger(__name__)

# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# MODEL = "claude-sonnet-4-20250514"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _call_groq(system: str, user: str, max_tokens: int = 512) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
        )

        # IMPORTANT: print real error if it fails
        if response.status_code != 200:
            raise Exception(f"{response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON — never raises on bad input."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Pipeline 1 — Behavioural signals
# ---------------------------------------------------------------------------

async def _behavioural_pipeline(persona: UserPersona) -> Optional[BehaviouralProfile]:
    """
    Derives star-rating patterns, category affinities, and rating bias
    from the user's review history. Pure computation — no LLM call needed.
    Falls back to neutral defaults for cold-start users.
    """
    try:
        history = persona.review_history
        if not history:
            return BehaviouralProfile(
                avg_rating=3.0,
                rating_std=0.0,
                category_affinities={},
                recency_weighted_avg=3.0,
                rating_bias=0.0,
                is_harsh_rater=False,
                is_generous_rater=False,
            )

        ratings = [r.rating for r in history]
        avg = mean(ratings)
        std = stdev(ratings) if len(ratings) > 1 else 0.0

        # Category affinities — normalised review counts
        cat_counts: Dict[str, int] = {}
        for r in history:
            cat_counts[r.category] = cat_counts.get(r.category, 0) + 1
        total = sum(cat_counts.values()) or 1
        affinities = {cat: count / total for cat, count in cat_counts.items()}

        # Recency weighting — more recent reviews carry more weight
        sorted_history = sorted(
            history,
            key=lambda r: r.timestamp or "0000",
            reverse=True,
        )
        weights = [1 / (i + 1) for i in range(len(sorted_history))]
        recency_avg = sum(
            r.rating * w for r, w in zip(sorted_history, weights)
        ) / sum(weights)

        # Rating bias vs platform average (default platform avg = 3.7)
        platform_avg = 3.7
        bias = avg - platform_avg

        return BehaviouralProfile(
            avg_rating=round(avg, 2),
            rating_std=round(std, 2),
            category_affinities=affinities,
            recency_weighted_avg=round(recency_avg, 2),
            rating_bias=round(bias, 2),
            is_harsh_rater=avg < 3.0,
            is_generous_rater=avg >= 4.3,
        )

    except Exception as exc:
        logger.warning("Behavioural pipeline failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Pipeline 2 — Textual signals
# ---------------------------------------------------------------------------

async def _textual_pipeline(persona: UserPersona) -> Optional[TextualProfile]:
    """
    Calls Claude to analyse the linguistic patterns in the user's review
    history — tone, sentiment, vocabulary, writing style.
    """
    try:
        history = persona.review_history
        if not history:
            return TextualProfile(
                dominant_tone=ToneProfile.MIXED,
                avg_review_length=0,
                sentiment_polarity=0.0,
                vocabulary_richness=0.0,
                uses_first_person=False,
                common_phrases=[],
            )

        # Build a compact review corpus (cap at 10 most recent reviews)
        corpus = "\n".join(
            f"[{r.category} · {r.rating}★] {r.text}"
            for r in history[-10:]
        )

        system = (
            "You are a linguistic analyst specialising in user review behaviour. "
            "Analyse the provided review corpus and return ONLY valid JSON — "
            "no preamble, no markdown fences."
        )

        user = f"""Analyse these reviews and return a JSON object with exactly these keys:
- dominant_tone: one of "expressive", "analytical", "terse", "narrative", "mixed"
- avg_review_length: integer (average word count per review)
- sentiment_polarity: float between -1.0 (very negative) and 1.0 (very positive)
- vocabulary_richness: float between 0.0 and 1.0 (type-token ratio estimate)
- uses_first_person: boolean
- common_phrases: list of up to 5 short phrases that recur or characterise this writer

Reviews:
{corpus}"""

        raw = await _call_groq(system, user, max_tokens=400)
        data = _parse_json(raw)

        return TextualProfile(
            dominant_tone=ToneProfile(data.get("dominant_tone", "mixed")),
            avg_review_length=int(data.get("avg_review_length", 0)),
            sentiment_polarity=float(data.get("sentiment_polarity", 0.0)),
            vocabulary_richness=float(data.get("vocabulary_richness", 0.0)),
            uses_first_person=bool(data.get("uses_first_person", False)),
            common_phrases=data.get("common_phrases", []),
        )

    except Exception as exc:
        logger.warning("Textual pipeline failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Pipeline 3 — Contextual signals
# ---------------------------------------------------------------------------

async def _contextual_pipeline(persona: UserPersona) -> Optional[ContextualProfile]:
    """
    Derives cold-start flags, cross-domain signals, and temporal context
    from the user's review history. Pure computation — no LLM call needed.
    """
    try:
        history = persona.review_history
        is_cold_start = len(history) == 0
        sparse = len(history) < 5

        # Active categories
        active_cats = list({r.category for r in history})

        # Cross-domain signals — categories outside the user's top 2
        cat_counts: Dict[str, int] = {}
        for r in history:
            cat_counts[r.category] = cat_counts.get(r.category, 0) + 1
        top_cats = sorted(cat_counts, key=lambda c: cat_counts[c], reverse=True)[:2]
        cross_domain = [c for c in active_cats if c not in top_cats]

        # Days since last review
        recency_days: Optional[int] = None
        timestamped = [r for r in history if r.timestamp]
        if timestamped:
            latest = max(timestamped, key=lambda r: r.timestamp)  # type: ignore[arg-type]
            try:
                last_dt = datetime.fromisoformat(latest.timestamp)
                now = datetime.now(timezone.utc)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                recency_days = (now - last_dt).days
            except ValueError:
                pass

        return ContextualProfile(
            is_cold_start=is_cold_start,
            sparse_history=sparse,
            active_categories=active_cats,
            cross_domain_signals=cross_domain,
            recency_days=recency_days,
        )

    except Exception as exc:
        logger.warning("Contextual pipeline failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Node 1 — PersonaConstructionAgent
# ---------------------------------------------------------------------------

async def persona_construction_agent(state: AgentState) -> AgentState:
    """
    LangGraph Node 1 — runs three concurrent pipelines and merges results
    into a typed UserState. Each pipeline is fault-isolated: if one fails,
    the others still complete and we log the error rather than crashing.
    """
    persona: UserPersona = state["user_persona"]
    errors: List[str] = list(state.get("errors", []))

    logger.info("PersonaConstructionAgent: building state for user %s", persona.user_id)

    # Run all three pipelines concurrently
    behavioural, textual, contextual = await asyncio.gather(
        _behavioural_pipeline(persona),
        _textual_pipeline(persona),
        _contextual_pipeline(persona),
        return_exceptions=False,
    )

    pipeline_errors: List[str] = []

    # Fallback defaults if a pipeline returned None
    if behavioural is None:
        pipeline_errors.append("behavioural_pipeline: returned None, using defaults")
        behavioural = BehaviouralProfile(
            avg_rating=3.0,
            rating_std=0.0,
            category_affinities={},
            recency_weighted_avg=3.0,
            rating_bias=0.0,
            is_harsh_rater=False,
            is_generous_rater=False,
        )

    if textual is None:
        pipeline_errors.append("textual_pipeline: returned None, using defaults")
        textual = TextualProfile(
            dominant_tone=ToneProfile.MIXED,
            avg_review_length=0,
            sentiment_polarity=0.0,
            vocabulary_richness=0.0,
            uses_first_person=False,
            common_phrases=[],
        )

    if contextual is None:
        pipeline_errors.append("contextual_pipeline: returned None, using defaults")
        contextual = ContextualProfile(
            is_cold_start=True,
            sparse_history=True,
            active_categories=[],
            cross_domain_signals=[],
            recency_days=None,
        )

    user_state = UserState(
        user_id=persona.user_id,
        behavioural=behavioural,
        textual=textual,
        contextual=contextual,
        pipeline_errors=pipeline_errors,
    )

    logger.info(
        "PersonaConstructionAgent: state built for %s (errors=%d)",
        persona.user_id,
        len(pipeline_errors),
    )

    return {
        **state,
        "user_state": user_state,
        "errors": errors + pipeline_errors,
        "reasoning_trace": list(state.get("reasoning_trace", [])) + [
            f"PersonaConstructionAgent: built UserState for {persona.user_id}",
            f"  behavioural.avg_rating={behavioural.avg_rating}",
            f"  textual.dominant_tone={textual.dominant_tone}",
            f"  contextual.is_cold_start={contextual.is_cold_start}",
            *[f"  pipeline_error: {e}" for e in pipeline_errors],
        ],
    }