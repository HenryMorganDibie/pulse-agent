"""
PulseAgent — RankingAgent (LangGraph Node 3B)
Team HOKM · DSN × BCT Hackathon 3.0
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import httpx

from src.schemas.models import (
    AgentState,
    CandidateItem,
    RankedItem,
    UserState,
)

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"

_TOP_K = 10

# ---------------------------------------------------------------------------
# LLM CALL
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


# ---------------------------------------------------------------------------
# JSON PARSER
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> Any:
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
# HELPERS
# ---------------------------------------------------------------------------

def _candidate_summary(candidate: CandidateItem) -> str:
    attr_str = ", ".join(f"{k}={v}" for k, v in candidate.attributes.items())
    return f"[{candidate.item_id}] {candidate.name} ({candidate.category}) — {attr_str}"


# ---------------------------------------------------------------------------
# STEP 1 — SCORING
# ---------------------------------------------------------------------------

async def _score_and_rank(
    candidates: List[CandidateItem],
    user_state: UserState,
    intent: str,
) -> tuple[List[Dict[str, Any]], List[str]]:

    trace: List[str] = []

    if not candidates:
        return [], ["RankingAgent: no candidates"]

    beh = user_state.behavioural
    txt = user_state.textual

    candidate_list = "\n".join(
        f"{i+1}. {_candidate_summary(c)}"
        for i, c in enumerate(candidates[:15])
    )

    system = (
        "You are a personalised ranking engine. "
        "Return ONLY valid JSON array."
    )

    user_prompt = f"""
Rank these items:

User intent: {intent}

User profile:
- avg rating: {beh.avg_rating}
- tone: {txt.dominant_tone.value}
- sentiment: {txt.sentiment_polarity}

Candidates:
{candidate_list}

Return JSON list:
[
  {{
    "item_id": "...",
    "predicted_rating": 1-5,
    "relevance_score": 0-1,
    "explanation": "..."
  }}
]
"""

    raw = await _call_groq(system, user_prompt, 1500)
    data = _parse_json(raw)

    if not isinstance(data, list):
        data = []

    trace.append(f"RankingAgent: scored {len(data)} items")
    return data, trace


# ---------------------------------------------------------------------------
# STEP 2 — DIVERSITY
# ---------------------------------------------------------------------------

def _inject_diversity(
    scored_items: List[Dict[str, Any]],
    candidates_by_id: Dict[str, CandidateItem],
    top_k: int = _TOP_K,
) -> List[Dict[str, Any]]:

    if not scored_items:
        return []

    seen = set()
    deduped = []

    for item in scored_items:
        item_id = item["item_id"]
        if item_id not in seen:
            seen.add(item_id)
            deduped.append(item)

    selected = []
    remaining = deduped
    recent_categories = []

    while remaining and len(selected) < top_k:
        placed = False

        for i, item in enumerate(remaining):
            item_id = item["item_id"]
            cat = candidates_by_id.get(item_id).category if item_id in candidates_by_id else "unknown"

            if recent_categories[-2:].count(cat) < 2 or i == len(remaining) - 1:
                selected.append(item)
                recent_categories.append(cat)
                remaining.pop(i)
                placed = True
                break

        if not placed:
            selected.append(remaining.pop(0))

    return selected


# ---------------------------------------------------------------------------
# STEP 3 — NDCG
# ---------------------------------------------------------------------------

def _assign_ndcg_scores(ranked_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    if not ranked_items:
        return []

    ideal = sorted(
        [float(i.get("relevance_score", 0.5)) for i in ranked_items],
        reverse=True,
    )

    ideal_dcg = sum(rel / (i + 2) for i, rel in enumerate(ideal)) or 1.0

    result = []

    for i, item in enumerate(ranked_items):
        rel = float(item.get("relevance_score", 0.5))
        dcg = rel / (i + 2)
        ndcg = round(min(1.0, dcg / ideal_dcg), 4)

        result.append({**item, "ndcg_score": ndcg})

    return result


# ---------------------------------------------------------------------------
# STEP 4 — EXPLANATION
# ---------------------------------------------------------------------------

async def _generate_explanation(
    ranked_items: List[RankedItem],
    user_state: UserState,
    intent: str,
) -> str:

    if not ranked_items:
        return "No recommendations available."

    top3 = ", ".join(r.name for r in ranked_items[:3])

    system = "You are a recommendation explainer."

    user_prompt = f"""
Explain why these were recommended:

Intent: {intent}
Top items: {top3}
"""

    try:
        return (await _call_groq(system, user_prompt, 200)).strip()
    except Exception:
        return f"Recommendations based on your interest in {intent}."


# ---------------------------------------------------------------------------
# NODE 3B — MAIN AGENT
# ---------------------------------------------------------------------------

async def ranking_agent(state: AgentState) -> AgentState:

    candidates: List[CandidateItem] = state.get("candidate_items", [])
    user_state: UserState = state["user_state"]
    intent: str = state.get("inferred_intent", "")
    trace: List[str] = list(state.get("reasoning_trace", []))
    errors: List[str] = list(state.get("errors", []))

    try:
        candidates_by_id = {c.item_id: c for c in candidates}

        scored_items, scoring_trace = await _score_and_rank(
            candidates, user_state, intent
        )
        trace.extend(scoring_trace)

        if not scored_items:
            scored_items = [
                {
                    "item_id": c.item_id,
                    "predicted_rating": user_state.behavioural.avg_rating,
                    "relevance_score": c.retrieval_score,
                    "explanation": f"Matches {c.category}",
                }
                for c in sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)
            ]

        diverse = _inject_diversity(scored_items, candidates_by_id)
        scored_with_ndcg = _assign_ndcg_scores(diverse)

        ranked: List[RankedItem] = []

        for idx, item in enumerate(scored_with_ndcg, start=1):

            item_id = item["item_id"]
            candidate = candidates_by_id.get(item_id)

            if not candidate:
                continue

            predicted_rating = float(item.get("predicted_rating", user_state.behavioural.avg_rating))
            explanation = item.get("explanation", "")
            ndcg = item.get("ndcg_score")

            ranked.append(
                RankedItem(
                    rank=idx,
                    item_id=item_id,
                    name=candidate.name,
                    category=candidate.category,
                    predicted_rating=round(min(5.0, max(1.0, predicted_rating)), 1),
                    explanation=explanation,
                    ndcg_score=ndcg,
                )
            )

        explanation_text = await _generate_explanation(ranked, user_state, intent)

        return {
            **state,
            "ranked_recommendations": ranked,
            "explanation": explanation_text,
            "reasoning_trace": trace,
            "errors": errors,
        }

    except Exception as exc:
        logger.error("RankingAgent failed: %s", exc)
        errors.append(str(exc))

        fallback = [
            RankedItem(
                rank=i + 1,
                item_id=c.item_id,
                name=c.name,
                category=c.category,
                predicted_rating=round(user_state.behavioural.avg_rating, 1),
                explanation=f"Based on {c.category}",
                ndcg_score=None,
            )
            for i, c in enumerate(
                sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)[:_TOP_K]
            )
        ]

        return {
            **state,
            "ranked_recommendations": fallback,
            "explanation": "Fallback recommendations generated.",
            "reasoning_trace": trace + ["RankingAgent failed"],
            "errors": errors,
        }