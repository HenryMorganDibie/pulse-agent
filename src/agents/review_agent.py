"""
PulseAgent — ReviewSimAgent (LangGraph Node 2A)
Team HOKM · DSN × BCT Hackathon 3.0

HACKATHON-STABLE VERSION
- Lower token usage
- Smaller Groq model
- Robust network handling
- Strong JSON parsing
- Safe fallbacks
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

import httpx

from src.schemas.models import (
    AgentState,
    ItemDetails,
    ToneProfile,
    UserState,
)

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "",
).strip()

# Faster + cheaper + more stable
MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# NETWORK SAFE GROQ CALL
# ---------------------------------------------------------------------------

async def _call_groq(
    system: str,
    user: str,
    max_tokens: int = 300,
) -> str:

    if not GROQ_API_KEY:
        raise Exception("Missing GROQ_API_KEY")

    try:

        async with httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
            ),
        ) as client:

            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization":
                        f"Bearer {GROQ_API_KEY}",
                    "Content-Type":
                        "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
            )

            if response.status_code != 200:

                raise Exception(
                    f"{response.status_code}: "
                    f"{response.text}"
                )

            return response.json()[
                "choices"
            ][0]["message"]["content"]

    except httpx.ConnectError:

        raise Exception(
            "Network connection failed"
        )

    except httpx.ReadTimeout:

        raise Exception(
            "Groq request timed out"
        )


# ---------------------------------------------------------------------------
# ROBUST JSON PARSER
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> Dict[str, Any]:

    cleaned = (
        raw.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    # direct parse
    try:
        return json.loads(cleaned)

    except Exception:
        pass

    # extract first object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:

        candidate = cleaned[start:end + 1]

        try:
            return json.loads(candidate)

        except Exception:
            pass

    # regex fallback
    match = re.search(
        r"\{[\s\S]*\}",
        cleaned,
    )

    if match:

        try:
            return json.loads(match.group())

        except Exception:
            pass

    logger.warning(
        "ReviewSimAgent JSON parse failed:\n%s",
        raw[:300],
    )

    return {}


# ---------------------------------------------------------------------------
# TONE HELPERS
# ---------------------------------------------------------------------------

def _tone_instructions(
    tone: ToneProfile,
) -> str:

    instructions = {

        ToneProfile.EXPRESSIVE:
            "Emotional and opinionated.",

        ToneProfile.ANALYTICAL:
            "Factual and measured.",

        ToneProfile.TERSE:
            "Very short and concise.",

        ToneProfile.NARRATIVE:
            "Tell a brief story.",

        ToneProfile.MIXED:
            "Natural conversational tone.",
    }

    return instructions.get(
        tone,
        instructions[ToneProfile.MIXED],
    )


# ---------------------------------------------------------------------------
# STEP 1 — RATING
# ---------------------------------------------------------------------------

async def _infer_rating(
    user_state: UserState,
    item: ItemDetails,
) -> tuple[float, List[str]]:

    trace: List[str] = []

    beh = user_state.behavioural

    anchor = (
        beh.recency_weighted_avg
        or beh.avg_rating
    )

    adjusted_anchor = round(
        min(
            5.0,
            max(
                1.0,
                anchor + (
                    beh.rating_bias * 0.3
                ),
            ),
        ),
        1,
    )

    trace.append(
        f"Anchor rating={adjusted_anchor}"
    )

    system = """
You predict user ratings.

Return ONLY valid JSON.
"""

    user_prompt = f"""
User avg rating:
{beh.avg_rating}

Rating bias:
{beh.rating_bias:+.2f}

Item:
{item.name}
({item.category})

Return:

{{
  "predicted_rating": 4.2,
  "confidence": 0.8,
  "reasoning": "Short reason"
}}
"""

    try:

        raw = await _call_groq(
            system,
            user_prompt,
            max_tokens=120,
        )

        data = _parse_json(raw)

        predicted = round(
            min(
                5.0,
                max(
                    1.0,
                    float(
                        data.get(
                            "predicted_rating",
                            adjusted_anchor,
                        )
                    ),
                ),
            ),
            1,
        )

        confidence = float(
            data.get(
                "confidence",
                0.7,
            )
        )

        reasoning = str(
            data.get(
                "reasoning",
                "Based on user history.",
            )
        )

        trace.append(
            f"Predicted={predicted}"
        )

        trace.append(
            f"Confidence={confidence:.2f}"
        )

        trace.append(
            f"Reasoning={reasoning}"
        )

        return predicted, trace

    except Exception as exc:

        trace.append(
            f"Fallback rating used: {exc}"
        )

        return adjusted_anchor, trace


# ---------------------------------------------------------------------------
# STEP 2 — REVIEW GENERATION
# ---------------------------------------------------------------------------

async def _generate_review(
    user_state: UserState,
    item: ItemDetails,
    predicted_rating: float,
) -> tuple[str, List[str]]:

    trace: List[str] = []

    txt = user_state.textual

    tone_instruction = (
        _tone_instructions(
            txt.dominant_tone
        )
    )

    avg_len = min(
        txt.avg_review_length or 60,
        80,
    )

    system = """
You generate realistic reviews.

Return ONLY valid JSON.
"""

    user_prompt = f"""
Write a realistic review.

Tone:
{tone_instruction}

Rating:
{predicted_rating} stars

Item:
{item.name}
({item.category})

Length:
~{avg_len} words

Return:

{{
  "review_text": "review here",
  "word_count": 50
}}
"""

    try:

        raw = await _call_groq(
            system,
            user_prompt,
            max_tokens=220,
        )

        data = _parse_json(raw)

        review_text = str(
            data.get(
                "review_text",
                "",
            )
        ).strip()

        if not review_text:

            review_text = (
                f"I enjoyed {item.name}. "
                f"It matched my expectations overall."
            )

        word_count = int(
            data.get(
                "word_count",
                len(review_text.split()),
            )
        )

        trace.append(
            f"Generated review "
            f"({word_count} words)"
        )

        return review_text, trace

    except Exception as exc:

        fallback = (
            f"I enjoyed {item.name}. "
            f"The experience was decent overall."
        )

        trace.append(
            f"Fallback review used: {exc}"
        )

        return fallback, trace


# ---------------------------------------------------------------------------
# STEP 3 — QUALITY
# ---------------------------------------------------------------------------

async def _score_review_quality(
    review_text: str,
    user_state: UserState,
    item: ItemDetails,
) -> tuple[float, List[str]]:

    trace: List[str] = []

    system = """
Score behavioural fidelity.

Return ONLY JSON.
"""

    user_prompt = f"""
Review:
{review_text[:300]}

Return:

{{
  "quality_score": 0.82,
  "notes": "Short note"
}}
"""

    try:

        raw = await _call_groq(
            system,
            user_prompt,
            max_tokens=100,
        )

        data = _parse_json(raw)

        score = round(
            min(
                1.0,
                max(
                    0.0,
                    float(
                        data.get(
                            "quality_score",
                            0.75,
                        )
                    ),
                ),
            ),
            2,
        )

        notes = str(
            data.get(
                "notes",
                "",
            )
        )

        trace.append(
            f"Quality={score}"
        )

        if notes:
            trace.append(notes)

        return score, trace

    except Exception as exc:

        trace.append(
            f"Fallback quality used: {exc}"
        )

        return 0.75, trace


# ---------------------------------------------------------------------------
# MAIN NODE
# ---------------------------------------------------------------------------

async def review_sim_agent(
    state: AgentState,
) -> AgentState:

    user_state: UserState = state[
        "user_state"
    ]

    item: ItemDetails = state[
        "item_details"
    ]

    errors: List[str] = list(
        state.get(
            "errors",
            [],
        )
    )

    trace: List[str] = list(
        state.get(
            "reasoning_trace",
            [],
        )
    )

    logger.info(
        "ReviewSimAgent: "
        "item=%s user=%s",
        item.item_id,
        user_state.user_id,
    )

    try:

        predicted_rating, rating_trace = (
            await _infer_rating(
                user_state,
                item,
            )
        )

        trace.extend(rating_trace)

        review_text, gen_trace = (
            await _generate_review(
                user_state,
                item,
                predicted_rating,
            )
        )

        trace.extend(gen_trace)

        quality_score, quality_trace = (
            await _score_review_quality(
                review_text,
                user_state,
                item,
            )
        )

        trace.extend(quality_trace)

        trace.append(
            "ReviewSimAgent complete"
        )

        return {
            **state,

            "simulated_rating":
                predicted_rating,

            "simulated_review":
                review_text,

            "review_quality_score":
                quality_score,

            "reasoning_trace":
                trace,

            "errors":
                errors,
        }

    except Exception as exc:

        logger.error(
            "ReviewSimAgent failed: %s",
            exc,
        )

        errors.append(
            f"ReviewSimAgent: {exc}"
        )

        fallback_rating = round(
            user_state.behavioural.avg_rating,
            1,
        )

        fallback_review = (
            f"I tried {item.name}. "
            f"It was okay overall."
        )

        return {
            **state,

            "simulated_rating":
                fallback_rating,

            "simulated_review":
                fallback_review,

            "review_quality_score":
                0.7,

            "reasoning_trace":
                trace + [
                    f"Fallback used: {exc}"
                ],

            "errors":
                errors,
        }