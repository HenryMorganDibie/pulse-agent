"""
PulseAgent — PersonaConstructionAgent (LangGraph Node 1)
Team HOKM · DSN × BCT Hackathon 3.0

HARDENED VERSION:
- Prevents ToneProfile enum crashes
- Sanitizes ALL LLM outputs
- Safe fallback at every layer
- Stable for ablation + production
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# SAFE TONE SPACE (CRITICAL)
# ---------------------------------------------------------------------------

ALLOWED_TONES = {
    "expressive",
    "analytical",
    "terse",
    "narrative",
    "mixed",
}

# ---------------------------------------------------------------------------
# SAFE HELPERS
# ---------------------------------------------------------------------------

def _safe_tone(value: Any) -> str:
    if not isinstance(value, str):
        return "mixed"
    return value if value in ALLOWED_TONES else "mixed"


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

        if response.status_code != 200:
            raise Exception(f"{response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> Dict[str, Any]:
    cleaned = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# PIPELINE 1 — BEHAVIOURAL (SAFE)
# ---------------------------------------------------------------------------

async def _behavioural_pipeline(persona: UserPersona) -> Optional[BehaviouralProfile]:
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

        cat_counts: Dict[str, int] = {}
        for r in history:
            cat_counts[r.category] = cat_counts.get(r.category, 0) + 1

        total = sum(cat_counts.values()) or 1
        affinities = {k: v / total for k, v in cat_counts.items()}

        sorted_history = sorted(
            history,
            key=lambda r: r.timestamp or "0000",
            reverse=True,
        )

        weights = [1 / (i + 1) for i in range(len(sorted_history))]
        recency_avg = sum(
            r.rating * w for r, w in zip(sorted_history, weights)
        ) / sum(weights)

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
# PIPELINE 2 — TEXTUAL (FULLY SAFE ENUM FIX)
# ---------------------------------------------------------------------------

async def _textual_pipeline(persona: UserPersona) -> Optional[TextualProfile]:
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

        corpus = "\n".join(
            f"[{r.category} · {r.rating}★] {r.text}"
            for r in history[-10:]
        )

        system = "Return ONLY valid JSON."

        user = f"""
Analyze reviews.

Return:
- dominant_tone: expressive | analytical | terse | narrative | mixed
- avg_review_length: int
- sentiment_polarity: float
- vocabulary_richness: float
- uses_first_person: bool
- common_phrases: list

Reviews:
{corpus}
"""

        raw = await _call_groq(system, user, max_tokens=400)
        data = _parse_json(raw)

        # ---------------- CRITICAL FIX ----------------
        tone_raw = data.get("dominant_tone", "mixed")
        tone = _safe_tone(tone_raw)

        return TextualProfile(
            dominant_tone=ToneProfile(tone),
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
# PIPELINE 3 — CONTEXTUAL
# ---------------------------------------------------------------------------

async def _contextual_pipeline(persona: UserPersona) -> Optional[ContextualProfile]:
    try:
        history = persona.review_history

        is_cold_start = len(history) == 0
        sparse = len(history) < 5

        active_cats = list({r.category for r in history})

        cat_counts: Dict[str, int] = {}
        for r in history:
            cat_counts[r.category] = cat_counts.get(r.category, 0) + 1

        top_cats = sorted(cat_counts, key=cat_counts.get, reverse=True)[:2]
        cross_domain = [c for c in active_cats if c not in top_cats]

        recency_days: Optional[int] = None
        timestamped = [r for r in history if r.timestamp]

        if timestamped:
            latest = max(timestamped, key=lambda r: r.timestamp)

            try:
                last_dt = datetime.fromisoformat(latest.timestamp)
                now = datetime.now(timezone.utc)

                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)

                recency_days = (now - last_dt).days
            except Exception:
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
# NODE 1 — MAIN
# ---------------------------------------------------------------------------

async def persona_construction_agent(state: AgentState) -> AgentState:
    persona: UserPersona = state["user_persona"]
    errors: List[str] = list(state.get("errors", []))

    behavioural, textual, contextual = await asyncio.gather(
        _behavioural_pipeline(persona),
        _textual_pipeline(persona),
        _contextual_pipeline(persona),
    )

    pipeline_errors: List[str] = []

    if behavioural is None:
        pipeline_errors.append("behavioural fallback")
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
        pipeline_errors.append("textual fallback")
        textual = TextualProfile(
            dominant_tone=ToneProfile.MIXED,
            avg_review_length=0,
            sentiment_polarity=0.0,
            vocabulary_richness=0.0,
            uses_first_person=False,
            common_phrases=[],
        )

    if contextual is None:
        pipeline_errors.append("contextual fallback")
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

    return {
        **state,
        "user_state": user_state,
        "errors": errors + pipeline_errors,
        "reasoning_trace": list(state.get("reasoning_trace", [])) + [
            f"PersonaConstructionAgent built for {persona.user_id}",
            f"avg_rating={behavioural.avg_rating}",
            f"tone={textual.dominant_tone}",
            f"cold_start={contextual.is_cold_start}",
        ],
    }