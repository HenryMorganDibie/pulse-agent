"""
PulseAgent — RankingAgent (LangGraph Node 3B)
Team HOKM · DSN × BCT Hackathon 3.0

HACKATHON SAFE VERSION
- Lower token usage
- Better JSON recovery
- Strong fallback handling
- More stable Groq calls
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
    CandidateItem,
    RankedItem,
    UserState,
)

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "",
).strip()

# FAST + CHEAP + STABLE
MODEL = "llama-3.1-8b-instant"

_TOP_K = 5


# ---------------------------------------------------------------------------
# LLM CALL
# ---------------------------------------------------------------------------

async def _call_groq(
    system: str,
    user: str,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> str:

    async with httpx.AsyncClient(
        timeout=45.0,
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
                "temperature": temperature,
            },
        )

        if response.status_code != 200:

            raise Exception(
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        return (
            data["choices"][0]
            ["message"]["content"]
        )


# ---------------------------------------------------------------------------
# SAFE JSON PARSER
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> Any:
    """
    Ultra-robust parser for malformed LLM JSON.
    """

    if not raw:
        return []

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

    # locate first JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start != -1 and end != -1:

        candidate = cleaned[
            start:end + 1
        ]

        try:
            return json.loads(candidate)

        except Exception:
            pass

    # regex extraction
    match = re.search(
        r"\[[\s\S]*\]",
        cleaned,
    )

    if match:

        try:
            return json.loads(
                match.group()
            )

        except Exception:
            pass

    # emergency recovery
    partial_items = re.findall(
        r'\{[\s\S]*?"item_id"[\s\S]*?\}',
        cleaned,
    )

    recovered = []

    for p in partial_items:

        try:
            recovered.append(
                json.loads(p)
            )

        except Exception:
            continue

    if recovered:

        logger.warning(
            "RankingAgent partially "
            "recovered malformed JSON"
        )

        return recovered

    logger.warning(
        "RankingAgent failed "
        "to parse JSON:\n%s",
        raw[:500],
    )

    return []


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _candidate_summary(
    candidate: CandidateItem,
) -> str:

    attrs = []

    for k, v in list(
        candidate.attributes.items()
    )[:3]:

        attrs.append(f"{k}={v}")

    attr_str = ", ".join(attrs)

    return (
        f"[{candidate.item_id}] "
        f"{candidate.name} "
        f"({candidate.category}) "
        f"{attr_str}"
    )


# ---------------------------------------------------------------------------
# STEP 1 — SCORING
# ---------------------------------------------------------------------------

async def _score_and_rank(
    candidates: List[CandidateItem],
    user_state: UserState,
    intent: str,
) -> tuple[
    List[Dict[str, Any]],
    List[str],
]:

    trace: List[str] = []

    if not candidates:

        return [], [
            "RankingAgent: "
            "no candidates"
        ]

    beh = user_state.behavioural

    candidate_list = "\n".join(
        f"{i+1}. {_candidate_summary(c)}"
        for i, c in enumerate(
            candidates[:5]
        )
    )

    system = """
You are a recommendation engine.

CRITICAL:
- Output ONLY JSON array
- No markdown
- No commentary
- Keep explanations SHORT
"""

    user_prompt = f"""
User avg rating:
{beh.avg_rating:.1f}

Intent:
{intent}

Items:
{candidate_list}

Return EXACT JSON:

[
  {{
    "item_id":"i_001",
    "predicted_rating":4.2,
    "relevance_score":0.84,
    "explanation":"Good category match."
  }}
]

Rules:
- Max 5 items
- predicted_rating between 1 and 5
- relevance_score between 0 and 1
- VERY SHORT explanations
"""

    try:

        raw = await _call_groq(
            system,
            user_prompt,
            max_tokens=350,
        )

    except Exception as e:

        logger.warning(
            "Groq call failed: %s",
            e,
        )

        return [], [
            f"Groq failed: {e}"
        ]

    logger.info(
        "RankingAgent raw:\n%s",
        raw[:300],
    )

    data = _parse_json(raw)

    if not isinstance(data, list):

        data = []

    cleaned_items = []

    for item in data:

        try:

            cleaned_items.append({

                "item_id":
                    str(
                        item.get(
                            "item_id",
                            "",
                        )
                    ),

                "predicted_rating":
                    round(
                        min(
                            5.0,
                            max(
                                1.0,
                                float(
                                    item.get(
                                        "predicted_rating",
                                        beh.avg_rating,
                                    )
                                ),
                            ),
                        ),
                        1,
                    ),

                "relevance_score":
                    round(
                        min(
                            1.0,
                            max(
                                0.0,
                                float(
                                    item.get(
                                        "relevance_score",
                                        0.5,
                                    )
                                ),
                            ),
                        ),
                        3,
                    ),

                "explanation":
                    str(
                        item.get(
                            "explanation",
                            "Good match.",
                        )
                    )[:100],
            })

        except Exception:
            continue

    trace.append(
        f"RankingAgent scored "
        f"{len(cleaned_items)} items"
    )

    return cleaned_items, trace


# ---------------------------------------------------------------------------
# STEP 2 — DIVERSITY
# ---------------------------------------------------------------------------

def _inject_diversity(
    scored_items: List[Dict[str, Any]],
    candidates_by_id: Dict[
        str,
        CandidateItem,
    ],
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

    deduped = sorted(
        deduped,
        key=lambda x:
            x.get(
                "relevance_score",
                0.0,
            ),
        reverse=True,
    )

    return deduped[:top_k]


# ---------------------------------------------------------------------------
# STEP 3 — NDCG
# ---------------------------------------------------------------------------

def _assign_ndcg_scores(
    ranked_items: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:

    if not ranked_items:

        return []

    ideal = sorted(
        [
            float(
                i.get(
                    "relevance_score",
                    0.5,
                )
            )
            for i in ranked_items
        ],
        reverse=True,
    )

    ideal_dcg = sum(
        rel / (i + 2)
        for i, rel in enumerate(
            ideal
        )
    )

    if ideal_dcg == 0:

        ideal_dcg = 1.0

    result = []

    for i, item in enumerate(
        ranked_items
    ):

        rel = float(
            item.get(
                "relevance_score",
                0.5,
            )
        )

        dcg = rel / (i + 2)

        ndcg = round(
            min(
                1.0,
                dcg / ideal_dcg,
            ),
            4,
        )

        result.append({
            **item,
            "ndcg_score": ndcg,
        })

    return result


# ---------------------------------------------------------------------------
# EXPLANATION
# ---------------------------------------------------------------------------

async def _generate_explanation(
    ranked_items: List[RankedItem],
    user_state: UserState,
    intent: str,
) -> str:

    if not ranked_items:

        return (
            "No recommendations "
            "available."
        )

    top_names = ", ".join(
        r.name
        for r in ranked_items[:3]
    )

    return (
        f"Recommendations generated "
        f"using user preference signals "
        f"and intent '{intent}'. "
        f"Top matches: {top_names}."
    )


# ---------------------------------------------------------------------------
# MAIN NODE
# ---------------------------------------------------------------------------

async def ranking_agent(
    state: AgentState,
) -> AgentState:

    candidates = state.get(
        "candidate_items",
        [],
    )

    user_state = state[
        "user_state"
    ]

    intent = state.get(
        "inferred_intent",
        "",
    )

    trace = list(
        state.get(
            "reasoning_trace",
            [],
        )
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    try:

        candidates_by_id = {
            c.item_id: c
            for c in candidates
        }

        scored_items, scoring_trace = (
            await _score_and_rank(
                candidates,
                user_state,
                intent,
            )
        )

        trace.extend(
            scoring_trace
        )

        # FALLBACK
        if not scored_items:

            logger.warning(
                "RankingAgent using "
                "fallback ranking"
            )

            scored_items = [
                {
                    "item_id":
                        c.item_id,

                    "predicted_rating":
                        round(
                            user_state
                            .behavioural
                            .avg_rating,
                            1,
                        ),

                    "relevance_score":
                        c.retrieval_score,

                    "explanation":
                        f"Matches "
                        f"{c.category}",
                }
                for c in sorted(
                    candidates,
                    key=lambda x:
                        x.retrieval_score,
                    reverse=True,
                )[:_TOP_K]
            ]

        diverse = _inject_diversity(
            scored_items,
            candidates_by_id,
        )

        scored_with_ndcg = (
            _assign_ndcg_scores(
                diverse
            )
        )

        ranked = []

        for idx, item in enumerate(
            scored_with_ndcg,
            start=1,
        ):

            item_id = item["item_id"]

            candidate = (
                candidates_by_id.get(
                    item_id
                )
            )

            if not candidate:

                continue

            ranked.append(
                RankedItem(
                    rank=idx,

                    item_id=item_id,

                    name=candidate.name,

                    category=
                        candidate.category,

                    predicted_rating=
                        item.get(
                            "predicted_rating",
                            3.0,
                        ),

                    explanation=
                        item.get(
                            "explanation",
                            "",
                        ),

                    ndcg_score=
                        item.get(
                            "ndcg_score"
                        ),
                )
            )

        explanation_text = (
            await _generate_explanation(
                ranked,
                user_state,
                intent,
            )
        )

        return {
            **state,

            "ranked_recommendations":
                ranked,

            "explanation":
                explanation_text,

            "reasoning_trace":
                trace,

            "errors":
                errors,
        }

    except Exception as exc:

        logger.error(
            "RankingAgent failed: %s",
            exc,
        )

        errors.append(str(exc))

        fallback = [
            RankedItem(
                rank=i + 1,

                item_id=c.item_id,

                name=c.name,

                category=c.category,

                predicted_rating=round(
                    user_state
                    .behavioural
                    .avg_rating,
                    1,
                ),

                explanation=
                    f"Based on "
                    f"{c.category}",

                ndcg_score=None,
            )
            for i, c in enumerate(
                sorted(
                    candidates,
                    key=lambda x:
                        x.retrieval_score,
                    reverse=True,
                )[:_TOP_K]
            )
        ]

        return {
            **state,

            "ranked_recommendations":
                fallback,

            "explanation":
                "Fallback recommendations "
                "generated.",

            "reasoning_trace":
                trace + [
                    "RankingAgent "
                    "fallback used"
                ],

            "errors":
                errors,
        }