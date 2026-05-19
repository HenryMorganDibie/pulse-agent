"""
PulseAgent — RankingAgent (LangGraph Node 3B)
Team HOKM · DSN × BCT Hackathon 3.0

Task B path only. Receives candidate items from ReasoningAgent (Node 2B) and:
  1. Scores each candidate's contextual relevance to the user + intent
  2. Predicts the rating the user would give each candidate
  3. Injects diversity across categories to avoid a monotonic ranked list
  4. Produces a final ranked list of RankedItem with per-item explanations
  5. Generates an overall explanation of the recommendation set
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

# Maximum recommendations to return
_TOP_K = 10


# ---------------------------------------------------------------------------
# Helpers — same pattern as Member 1's agents
# ---------------------------------------------------------------------------

async def _call_claude(system: str, user: str, max_tokens: int = 1200) -> str:
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


def _parse_json(raw: str) -> Any:
    """Strip markdown fences and parse JSON — never raises on bad input."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _candidate_summary(candidate: CandidateItem) -> str:
    """Compact string representation of a candidate for prompt building."""
    attr_str = ", ".join(f"{k}={v}" for k, v in candidate.attributes.items())
    return f"[{candidate.item_id}] {candidate.name} ({candidate.category}) — {attr_str}"


# ---------------------------------------------------------------------------
# Step 1 — Score and rank candidates via Claude
# ---------------------------------------------------------------------------

async def _score_and_rank(
    candidates: List[CandidateItem],
    user_state: UserState,
    intent: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Asks Claude to evaluate each candidate against the user profile and intent,
    returning predicted ratings and per-item explanations in ranked order.

    Returns (scored_items, trace_lines) where scored_items is a list of dicts
    with keys: item_id, predicted_rating, relevance_score, explanation.
    """
    trace: List[str] = []

    if not candidates:
        return [], ["RankingAgent: no candidates to score"]

    beh = user_state.behavioural
    txt = user_state.textual

    # Prepare compact candidate list for the prompt (cap at 15 to stay within tokens)
    candidate_list = "\n".join(
        f"{i+1}. {_candidate_summary(c)}"
        for i, c in enumerate(candidates[:15])
    )

    system = (
        "You are a personalised recommendation ranking engine. "
        "Given a user's profile and a list of candidate items, score and rank them. "
        "Return ONLY valid JSON — no preamble, no markdown fences."
    )

    user_prompt = f"""Rank these candidate items for this user based on their profile and intent.

User profile:
- Average rating: {beh.avg_rating} (std={beh.rating_std})
- Is generous rater: {beh.is_generous_rater}, is harsh rater: {beh.is_harsh_rater}
- Category affinities (higher = stronger preference): {dict(beh.category_affinities)}
- Dominant tone/personality: {txt.dominant_tone.value}
- Sentiment polarity: {txt.sentiment_polarity:.2f} (-1=negative, +1=positive)

User's current intent: "{intent}"

Candidate items:
{candidate_list}

For each candidate return a JSON object. Return a JSON array of objects with these keys:
- item_id: string (must match the [item_id] from the candidate list)
- predicted_rating: float (1.0–5.0, what this user would likely rate this item)
- relevance_score: float (0.0–1.0, how well it matches their intent right now)
- explanation: string (1–2 sentences personalised to this user's profile and intent)

Order the array from most to least relevant. Include all candidates."""

    raw = await _call_claude(system, user_prompt, max_tokens=1500)
    data = _parse_json(raw)

    if not isinstance(data, list):
        # Claude sometimes wraps in an object — try to unwrap
        if isinstance(data, dict):
            data = data.get("rankings", data.get("items", data.get("recommendations", [])))
        if not isinstance(data, list):
            data = []

    trace.append(f"RankingAgent: Claude scored {len(data)} candidates")
    return data, trace


# ---------------------------------------------------------------------------
# Step 2 — Diversity injection
# ---------------------------------------------------------------------------

def _inject_diversity(
    scored_items: List[Dict[str, Any]],
    candidates_by_id: Dict[str, CandidateItem],
    top_k: int = _TOP_K,
) -> List[Dict[str, Any]]:
    """
    Ensures the top-K list isn't dominated by a single category.

    Strategy: sliding window — if the last 2 items are the same category,
    skip ahead to find a different-category item before continuing.

    This improves the NDCG diversity component without sacrificing relevance
    at the top of the list (rank 1 is always the highest-scored item).
    """
    if not scored_items:
        return []

    # Deduplicate by item_id before diversity pass (Claude sometimes repeats items)
    seen_ids: set = set()
    deduped: List[Dict[str, Any]] = []
    for item in scored_items:
        iid = item.get("item_id")
        if iid not in seen_ids:
            seen_ids.add(iid)
            deduped.append(item)

    selected: List[Dict[str, Any]] = []
    remaining = deduped
    recent_categories: List[str] = []

    while remaining and len(selected) < top_k:
        placed = False
        for i, item in enumerate(remaining):
            cat = candidates_by_id.get(item.get("item_id", ""), CandidateItem(
                item_id="", name="", category="unknown", retrieval_score=0.0
            )).category

            # Allow item if last 2 picks weren't the same category, or if we're out of options
            if recent_categories[-2:].count(cat) < 2 or i == len(remaining) - 1:
                selected.append(item)
                recent_categories.append(cat)
                remaining.pop(i)
                placed = True
                break

        if not placed:
            # Safety valve — just take the next item
            selected.append(remaining.pop(0))

    return selected


# ---------------------------------------------------------------------------
# Step 3 — NDCG score assignment
# ---------------------------------------------------------------------------

def _assign_ndcg_scores(ranked_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assigns a per-position NDCG-style score to each item based on relevance_score
    and rank position. NDCG@K = sum of (rel_i / log2(i+2)) for i in 0..K-1,
    normalised by the ideal ordering.

    Each item gets its individual contribution score for interpretability.
    """
    if not ranked_items:
        return []

    # Ideal DCG (if items were sorted by descending relevance)
    ideal_rels = sorted(
        [float(item.get("relevance_score", 0.5)) for item in ranked_items],
        reverse=True,
    )
    ideal_dcg = sum(
        rel / (i + 2) * (1 / 0.6309)  # log2(2) normalisation factor
        for i, rel in enumerate(ideal_rels)
    ) or 1.0

    result = []
    for i, item in enumerate(ranked_items):
        rel = float(item.get("relevance_score", 0.5))
        dcg_contribution = rel / (i + 2) * (1 / 0.6309)
        ndcg = min(1.0, round(dcg_contribution / ideal_dcg, 4))
        result.append({**item, "ndcg_score": ndcg})

    return result


# ---------------------------------------------------------------------------
# Step 4 — Overall explanation
# ---------------------------------------------------------------------------

async def _generate_explanation(
    ranked_items: List[RankedItem],
    user_state: UserState,
    intent: str,
) -> tuple[str, List[str]]:
    """
    Generates a single conversational explanation summarising the recommendation
    set — why these items, why now, what they have in common.
    """
    trace: List[str] = []

    if not ranked_items:
        return "No recommendations could be generated.", []

    top3 = ", ".join(f"\"{r.name}\"" for r in ranked_items[:3])

    system = (
        "You are a recommendation assistant giving a friendly summary. "
        "Write 2–3 sentences explaining why these items were chosen for this user. "
        "Be specific to their preferences — do not use generic phrases."
    )

    user_prompt = f"""Summarise why these recommendations fit the user.

User intent: "{intent}"
User avg rating: {user_state.behavioural.avg_rating}, tone: {user_state.textual.dominant_tone.value}
Top recommendations: {top3}
Total recommendations: {len(ranked_items)}

Write a concise, personalised 2–3 sentence summary. No JSON — plain text only."""

    try:
        explanation = await _call_claude(system, user_prompt, max_tokens=200)
        explanation = explanation.strip()
        trace.append("RankingAgent: overall explanation generated")
    except Exception as exc:
        logger.warning("RankingAgent: explanation generation failed: %s", exc)
        explanation = f"Based on your preferences, we found {len(ranked_items)} items matching your intent: {intent}."

    return explanation, trace


# ---------------------------------------------------------------------------
# Node 3B — RankingAgent
# ---------------------------------------------------------------------------

async def ranking_agent(state: AgentState) -> AgentState:
    """
    LangGraph Node 3B — Task B path only.

    Reads:
        state["candidate_items"]   — shortlisted by ReasoningAgent (Node 2B)
        state["user_state"]        — built by Node 1
        state["inferred_intent"]   — set by Node 2B

    Writes:
        state["ranked_recommendations"]  — final List[RankedItem]
        state["explanation"]             — overall recommendation explanation
        state["reasoning_trace"]         — appended chain-of-thought
    """
    candidates: List[CandidateItem] = state.get("candidate_items", [])
    user_state: UserState = state["user_state"]
    intent: str = state.get("inferred_intent", "") or ""
    errors: List[str] = list(state.get("errors", []))
    trace: List[str] = list(state.get("reasoning_trace", []))

    logger.info(
        "RankingAgent: ranking %d candidates for user=%s",
        len(candidates), user_state.user_id,
    )

    try:
        # Build a lookup dict for diversity injection
        candidates_by_id: Dict[str, CandidateItem] = {c.item_id: c for c in candidates}

        # Step 1 — score and rank via Claude
        scored_items, scoring_trace = await _score_and_rank(candidates, user_state, intent)
        trace.extend(scoring_trace)

        if not scored_items:
            # Fallback: rank by retrieval_score if Claude scoring failed
            scored_items = [
                {
                    "item_id": c.item_id,
                    "predicted_rating": user_state.behavioural.avg_rating,
                    "relevance_score": c.retrieval_score,
                    "explanation": f"Matches your interest in {c.category}.",
                }
                for c in sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)
            ]
            trace.append("RankingAgent: Claude scoring failed, using retrieval_score fallback")

        # Step 2 — inject diversity
        diverse_items = _inject_diversity(scored_items, candidates_by_id)
        trace.append(f"RankingAgent: diversity injection → {len(diverse_items)} items")

        # Step 3 — assign NDCG scores
        scored_with_ndcg = _assign_ndcg_scores(diverse_items)

        # Step 4 — build RankedItem list
        ranked_recommendations: List[RankedItem] = []
        for rank, item_data in enumerate(scored_with_ndcg, start=1):
            item_id = item_data.get("item_id", "")
            candidate = candidates_by_id.get(item_id)
            if candidate is None:
                continue

            ranked_recommendations.append(RankedItem(
                rank=rank,
                item_id=item_id,
                name=candidate.name,
                category=candidate.category,
                predicted_rating=round(
                    min(5.0, max(1.0, float(item_data.get("predicted_rating", user_state.behavioural.avg_rating)))),
                    1,
                ),
                explanation=str(item_data.get("explanation", f"Recommended based on your {candidate.category} preferences.")),
                ndcg_score=item_data.get("ndcg_score"),
            ))

        # Step 5 — generate overall explanation
        explanation, exp_trace = await _generate_explanation(ranked_recommendations, user_state, intent)
        trace.extend(exp_trace)

        trace.append(
            f"RankingAgent: complete — {len(ranked_recommendations)} recommendations, "
            f"top={ranked_recommendations[0].name if ranked_recommendations else 'none'}"
        )

        return {
            **state,
            "ranked_recommendations": ranked_recommendations,
            "explanation": explanation,
            "reasoning_trace": trace,
            "errors": errors,
        }

    except Exception as exc:
        logger.error("RankingAgent failed: %s", exc)
        errors.append(f"RankingAgent: {exc}")

        # Graceful fallback — return candidates as-is with minimal ranking
        fallback: List[RankedItem] = [
            RankedItem(
                rank=i + 1,
                item_id=c.item_id,
                name=c.name,
                category=c.category,
                predicted_rating=round(user_state.behavioural.avg_rating, 1),
                explanation=f"Recommended based on your interest in {c.category}.",
                ndcg_score=None,
            )
            for i, c in enumerate(
                sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)[:_TOP_K]
            )
        ]

        return {
            **state,
            "ranked_recommendations": fallback,
            "explanation": f"Here are {len(fallback)} items based on your preferences.",
            "reasoning_trace": trace + [f"RankingAgent: failed — {exc}, using fallback ranking"],
            "errors": errors,
        }
