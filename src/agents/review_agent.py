"""
PulseAgent — ReviewSimAgent (LangGraph Node 2A)
Team HOKM · DSN × BCT Hackathon 3.0

Given a fully constructed UserState and an unseen item, this agent:
  1. Infers the star rating the user would give
  2. Generates a review in the user's authentic tone and style
  3. Scores the review quality against the user's history
  4. Returns a SimulatedReview with a full reasoning trace
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import httpx
from src.schemas.models import (
    AgentState,
    ItemDetails,
    SimulatedReview,
    ToneProfile,
    UserState,
)

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _call_claude(system: str, user: str, max_tokens: int = 800) -> str:
    """Async LLM call via Groq (OpenAI-compatible). Returns raw text content."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _tone_instructions(tone: ToneProfile) -> str:
    instructions = {
        ToneProfile.EXPRESSIVE:  "Use strong opinions and emotional language. Short punchy sentences. Don't hold back.",
        ToneProfile.ANALYTICAL:  "Be measured and factual. Reference specific attributes. Avoid hyperbole.",
        ToneProfile.TERSE:       "Keep it very brief — 2 to 3 sentences maximum. No filler.",
        ToneProfile.NARRATIVE:   "Tell a story. Give context — who you were with, why you went, what happened.",
        ToneProfile.MIXED:       "Write naturally. No particular constraint on style.",
    }
    return instructions.get(tone, instructions[ToneProfile.MIXED])


# ---------------------------------------------------------------------------
# Step 1 — Infer rating
# ---------------------------------------------------------------------------

async def _infer_rating(user_state: UserState, item: ItemDetails) -> tuple[float, List[str]]:
    """
    Infers the star rating this user would give the item.
    Uses the user's category affinity, rating bias, and recency-weighted average
    as anchors, then asks Claude to adjust based on item attributes.
    """
    trace: List[str] = []

    beh = user_state.behavioural
    category = item.category

    # Anchor: category-specific average if available, else user global average
    category_affinity = beh.category_affinities.get(category)
    if category_affinity is not None:
        # Reconstruct approximate category avg from history
        anchor = beh.recency_weighted_avg
        trace.append(f"Rating anchor: user has reviewed {category} before (affinity={category_affinity:.2f})")
    else:
        anchor = beh.avg_rating
        trace.append(f"Rating anchor: new category for user, using global avg={anchor}")

    # Adjust for rating bias
    adjusted_anchor = min(5.0, max(1.0, anchor + beh.rating_bias * 0.3))
    trace.append(f"Bias adjustment: {beh.rating_bias:+.2f} → adjusted anchor={adjusted_anchor:.2f}")

    system = (
        "You are a rating inference engine. Given a user's rating profile and an item, "
        "predict the star rating (1.0–5.0, one decimal place) the user would give. "
        "Return ONLY valid JSON — no preamble, no markdown."
    )

    user_prompt = f"""User rating profile:
- Average rating: {beh.avg_rating}
- Rating std dev: {beh.rating_std}
- Is harsh rater: {beh.is_harsh_rater}
- Is generous rater: {beh.is_generous_rater}
- Rating bias vs platform avg: {beh.rating_bias:+.2f}
- Recency-weighted avg: {beh.recency_weighted_avg}
- Adjusted anchor for this prediction: {adjusted_anchor:.2f}

Item to rate:
- Name: {item.name}
- Category: {item.category}
- Attributes: {json.dumps(item.attributes)}
- Platform avg rating: {item.avg_rating or "unknown"}

Return JSON with exactly these keys:
- predicted_rating: float (1.0–5.0)
- confidence: float (0.0–1.0)
- reasoning: string (one sentence explaining the prediction)"""

    raw = await _call_claude(system, user_prompt, max_tokens=200)
    data = _parse_json(raw)

    predicted = float(data.get("predicted_rating", adjusted_anchor))
    predicted = round(min(5.0, max(1.0, predicted)), 1)
    confidence = float(data.get("confidence", 0.7))
    reasoning = data.get("reasoning", "Predicted from user's historical rating patterns.")

    trace.append(f"Claude rating inference: {predicted}★ (confidence={confidence:.2f})")
    trace.append(f"  reasoning: {reasoning}")

    return predicted, trace


# ---------------------------------------------------------------------------
# Step 2 — Generate review
# ---------------------------------------------------------------------------

async def _generate_review(
    user_state: UserState,
    item: ItemDetails,
    predicted_rating: float,
) -> tuple[str, List[str]]:
    """
    Generates a review text that authentically matches the user's tone,
    vocabulary, and writing style — conditioned on the predicted rating.
    """
    trace: List[str] = []

    txt = user_state.textual
    tone_instruction = _tone_instructions(txt.dominant_tone)

    # Sample up to 3 of the user's past reviews as style anchors
    sample_reviews = user_state.behavioural  # used for avg length reference
    avg_len = txt.avg_review_length or 60

    system = (
        "You are a review generation engine. Write a product/service review that "
        "authentically matches the user's established writing style. "
        "Return ONLY valid JSON — no preamble, no markdown."
    )

    user_prompt = f"""Write a review for the item below that this user would realistically write.

User writing profile:
- Dominant tone: {txt.dominant_tone.value}
- Tone instruction: {tone_instruction}
- Average review length: ~{avg_len} words
- Sentiment polarity: {txt.sentiment_polarity:.2f} (scale: -1 very negative, +1 very positive)
- Uses first person: {txt.uses_first_person}
- Common phrases they use: {txt.common_phrases}

Item being reviewed:
- Name: {item.name}
- Category: {item.category}
- Attributes: {json.dumps(item.attributes)}

Star rating the user would give: {predicted_rating}★

Return JSON with exactly these keys:
- review_text: string (the generated review, matching the user's style)
- word_count: integer"""

    raw = await _call_claude(system, user_prompt, max_tokens=600)
    data = _parse_json(raw)

    review_text = data.get("review_text", "")
    word_count = int(data.get("word_count", len(review_text.split())))

    trace.append(f"Review generated: {word_count} words, tone={txt.dominant_tone.value}")

    return review_text, trace


# ---------------------------------------------------------------------------
# Step 3 — Score review quality
# ---------------------------------------------------------------------------

async def _score_review_quality(
    review_text: str,
    user_state: UserState,
    item: ItemDetails,
) -> tuple[float, List[str]]:
    """
    Asks Claude to score how well the generated review matches the user's
    established behavioural patterns — a proxy for behavioural fidelity.
    """
    trace: List[str] = []

    system = (
        "You are a review quality evaluator assessing behavioural fidelity. "
        "Return ONLY valid JSON — no preamble, no markdown."
    )

    user_prompt = f"""Score the generated review for behavioural fidelity — how well it matches the user's profile.

User profile summary:
- Avg rating: {user_state.behavioural.avg_rating}
- Dominant tone: {user_state.textual.dominant_tone.value}
- Avg review length: {user_state.textual.avg_review_length} words
- Uses first person: {user_state.textual.uses_first_person}

Generated review:
\"{review_text}\"

Return JSON with exactly these keys:
- quality_score: float (0.0–1.0, where 1.0 = perfect behavioural match)
- notes: string (one sentence on what could be improved)"""

    raw = await _call_claude(system, user_prompt, max_tokens=200)
    data = _parse_json(raw)

    score = float(data.get("quality_score", 0.75))
    notes = data.get("notes", "")

    trace.append(f"Quality score: {score:.2f} — {notes}")

    return score, trace


# ---------------------------------------------------------------------------
# Node 2A — ReviewSimAgent
# ---------------------------------------------------------------------------

async def review_sim_agent(state: AgentState) -> AgentState:
    """
    LangGraph Node 2A — Task A path only.
    Runs rating inference → review generation → quality scoring sequentially.
    """
    user_state: UserState = state["user_state"]
    item: ItemDetails = state["item_details"]
    errors: List[str] = list(state.get("errors", []))
    trace: List[str] = list(state.get("reasoning_trace", []))

    logger.info("ReviewSimAgent: simulating review for item=%s user=%s", item.item_id, user_state.user_id)

    try:
        # Step 1 — infer rating
        predicted_rating, rating_trace = await _infer_rating(user_state, item)
        trace.extend(rating_trace)

        # Step 2 — generate review
        review_text, gen_trace = await _generate_review(user_state, item, predicted_rating)
        trace.extend(gen_trace)

        # Step 3 — score quality
        quality_score, quality_trace = await _score_review_quality(review_text, user_state, item)
        trace.extend(quality_trace)

        trace.append(f"ReviewSimAgent: complete — rating={predicted_rating}★, quality={quality_score:.2f}")

        return {
            **state,
            "simulated_rating": predicted_rating,
            "simulated_review": review_text,
            "review_quality_score": quality_score,
            "reasoning_trace": trace,
            "errors": errors,
        }

    except Exception as exc:
        logger.error("ReviewSimAgent failed: %s", exc)
        errors.append(f"ReviewSimAgent: {exc}")
        return {
            **state,
            "simulated_rating": user_state.behavioural.avg_rating,
            "simulated_review": "",
            "review_quality_score": 0.0,
            "reasoning_trace": trace + [f"ReviewSimAgent: failed — {exc}"],
            "errors": errors,
        }
