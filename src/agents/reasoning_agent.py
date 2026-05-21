"""
PulseAgent — ReasoningAgent (LangGraph Node 2B)
Team HOKM · DSN × BCT Hackathon 3.0

Task B path only. Receives a fully constructed UserState and:
  1. Extracts the user's intent from their query / conversation history
  2. Detects cold-start conditions and selects the appropriate strategy
  3. Handles cross-domain bridging (e.g. food reviewer → book recommendation)
  4. Retrieves a shortlist of candidate items via RetrievalTool
  5. Passes candidates + intent downstream to RankingAgent (Node 3B)
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
    Message,
    UserState,
)
from src.tools.retrieval_tool import retrieve_candidates

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
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


def _format_conversation(history: List[Message]) -> str:
    """Format conversation history as a readable string for the prompt."""
    if not history:
        return "(no conversation history)"
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in history[-6:])


# ---------------------------------------------------------------------------
# Step 1 — Intent extraction
# ---------------------------------------------------------------------------

async def _extract_intent(
    user_state: UserState,
    conversation_history: List[Message],
) -> tuple[str, List[str]]:
    """
    Asks Claude to infer what the user actually wants right now — distilling
    the conversation history and user profile into a concise intent string.

    Returns (intent_string, trace_lines).
    """
    trace: List[str] = []

    beh = user_state.behavioural
    ctx = user_state.contextual

    system = (
        "You are an intent extraction engine for a recommendation system. "
        "Given a user's profile and conversation history, infer what they want. "
        "Return ONLY valid JSON — no preamble, no markdown fences."
    )

    user_prompt = f"""Extract the user's recommendation intent from the conversation below.
Be specific — capture the actual mood, occasion, constraints, and preferences expressed.
Do not produce vague intents like "general recommendation" or "something fun".

User profile:
- Known categories (most reviewed first): {list(beh.category_affinities.keys())}
- Is new user (cold-start): {ctx.is_cold_start}
- Avg rating: {beh.avg_rating} | generous_rater: {beh.is_generous_rater} | harsh_rater: {beh.is_harsh_rater}
- Cross-domain signals (categories explored outside their main ones): {ctx.cross_domain_signals}

Conversation history:
{_format_conversation(conversation_history)}

Return JSON with exactly these keys:
- intent: string — specific intent capturing occasion + mood + constraints (e.g. "relaxed Saturday evening out, budget under $$, prefers cozy low-noise venues" not just "casual outing")
- target_categories: list of 1–3 category strings most relevant to this request
- budget_sensitivity: one of "budget" | "moderate" | "splurge" | "unknown"
- mood_keywords: list of 3–5 specific vibe words pulled directly from the conversation (e.g. ["chill", "low-key", "not too loud"])
- occasion: string — what kind of outing/activity this is (e.g. "solo weekend wind-down", "date night", "group hangout", "quick lunch")
- is_cross_domain: boolean — true if the request targets a category the user has rarely or never reviewed"""

    raw = await _call_groq(system, user_prompt, max_tokens=300)
    data = _parse_json(raw)

    intent = data.get("intent", "general recommendation")
    occasion = data.get("occasion", "")
    target_cats = data.get("target_categories", list(beh.category_affinities.keys())[:2])
    budget = data.get("budget_sensitivity", "unknown")
    mood_kws = data.get("mood_keywords", [])
    is_cross = data.get("is_cross_domain", False)

    # Enrich intent string with occasion for the ranking agent
    if occasion and occasion.lower() not in intent.lower():
        intent = f"{intent} ({occasion})"

    trace.append(f"Intent extracted: \"{intent}\"")
    trace.append(f"  target_categories={target_cats}, budget={budget}, mood={mood_kws}")
    if is_cross:
        trace.append("  cross-domain request detected")

    return intent, trace


# ---------------------------------------------------------------------------
# Step 2 — Cold-start detection and strategy selection
# ---------------------------------------------------------------------------

def _cold_start_strategy(
    user_state: UserState,
    intent: str,
) -> tuple[str, List[str]]:
    """
    Decides how to handle the request based on how much we know about the user.

    Strategies:
      - "normal"       — enough history to use preference-based retrieval
      - "sparse"       — limited history; widen category net, rely more on intent
      - "cold_start"   — new user; fall back to popularity + intent matching
      - "cross_domain" — known user moving into a new category; bridge from known preferences

    Returns (strategy_name, trace_lines).
    """
    trace: List[str] = []
    ctx = user_state.contextual
    beh = user_state.behavioural

    if ctx.is_cold_start:
        strategy = "cold_start"
        trace.append("Cold-start strategy: new user — using popularity + intent matching")
    elif ctx.sparse_history:
        strategy = "sparse"
        trace.append("Sparse-history strategy: < 5 reviews — widening category net")
    elif intent and not any(
        cat.lower() in intent.lower() for cat in ctx.active_categories
    ):
        strategy = "cross_domain"
        trace.append(
            f"Cross-domain strategy: intent suggests new territory "
            f"(known={ctx.active_categories}, cross={ctx.cross_domain_signals})"
        )
    else:
        strategy = "normal"
        trace.append(f"Normal strategy: {len(beh.category_affinities)} known categories")

    return strategy, trace


# ---------------------------------------------------------------------------
# Step 3 — Candidate shortlisting
# ---------------------------------------------------------------------------

def _shortlist_candidates(
    user_state: UserState,
    intent: str,
    strategy: str,
    n: int = 15,
) -> tuple[List[CandidateItem], List[str]]:
    """
    Retrieves candidate items using the RetrievalTool, adjusting retrieval
    parameters based on the cold-start strategy.

    Returns (candidates, trace_lines).
    """
    trace: List[str] = []

    if strategy == "cold_start":
        # New user — retrieve broadly across all categories, sorted by keyword overlap
        candidates = retrieve_candidates(user_state, query=intent, n=n)
        trace.append(f"Cold-start retrieval: {len(candidates)} items across all categories")

    elif strategy == "sparse":
        # Limited history — use intent heavily, slightly widen top-N
        candidates = retrieve_candidates(user_state, query=intent, n=n + 5)
        candidates = candidates[:n]
        trace.append(f"Sparse retrieval: {len(candidates)} items, intent-weighted")

    elif strategy == "cross_domain":
        # Cross-domain — mix preferred categories with intent-based discovery
        preferred = retrieve_candidates(user_state, query=intent, n=n // 2)
        # Also pull from intent-matching regardless of past categories
        all_candidates = retrieve_candidates(user_state, query=intent, n=n)
        # Merge: preferred first, then fill with intent-matched not already included
        seen_ids = {c.item_id for c in preferred}
        extra = [c for c in all_candidates if c.item_id not in seen_ids]
        candidates = (preferred + extra)[:n]
        trace.append(
            f"Cross-domain retrieval: {len(preferred)} preferred + {len(extra)} intent-matched"
        )

    else:
        # Normal — standard retrieval
        candidates = retrieve_candidates(user_state, query=intent, n=n)
        trace.append(f"Normal retrieval: {len(candidates)} candidates")

    return candidates, trace


# ---------------------------------------------------------------------------
# Node 2B — ReasoningAgent
# ---------------------------------------------------------------------------

async def reasoning_agent(state: AgentState) -> AgentState:
    """
    LangGraph Node 2B — Task B path only.

    Reads:
        state["user_state"]                     — built by Node 1
        state["user_persona"].conversation_history — multi-turn context

    Writes:
        state["inferred_intent"]    — concise intent string
        state["candidate_items"]    — shortlisted CandidateItem list for Node 3B
        state["reasoning_trace"]    — appended chain-of-thought
    """
    user_state: UserState = state["user_state"]
    conversation_history = state["user_persona"].conversation_history
    errors: List[str] = list(state.get("errors", []))
    trace: List[str] = list(state.get("reasoning_trace", []))

    logger.info("ReasoningAgent: starting for user=%s", user_state.user_id)

    try:
        # Step 1 — extract intent from conversation + profile
        intent, intent_trace = await _extract_intent(user_state, conversation_history)
        trace.extend(intent_trace)

        # Step 2 — determine retrieval strategy based on user data richness
        strategy, strategy_trace = _cold_start_strategy(user_state, intent)
        trace.extend(strategy_trace)

        # Step 3 — retrieve candidate items
        candidates, retrieval_trace = _shortlist_candidates(user_state, intent, strategy)
        trace.extend(retrieval_trace)

        trace.append(
            f"ReasoningAgent: complete — intent=\"{intent}\", "
            f"strategy={strategy}, candidates={len(candidates)}"
        )

        return {
            **state,
            "inferred_intent": intent,
            "candidate_items": candidates,
            "reasoning_trace": trace,
            "errors": errors,
        }

    except Exception as exc:
        logger.error("ReasoningAgent failed: %s", exc)
        errors.append(f"ReasoningAgent: {exc}")

        # Graceful fallback — return whatever we retrieved before the failure
        fallback_candidates = retrieve_candidates(user_state, query="", n=10)
        return {
            **state,
            "inferred_intent": "general recommendation (fallback)",
            "candidate_items": fallback_candidates,
            "reasoning_trace": trace + [f"ReasoningAgent: failed — {exc}, using fallback retrieval"],
            "errors": errors,
        }