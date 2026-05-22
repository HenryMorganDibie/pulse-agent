"""
PulseAgent — FastAPI Backend
Team HOKM · DSN × BCT Hackathon 3.0

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then visit: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.graph import run_task_a, run_task_b

logger = logging.getLogger(__name__)

from src.api.schemas import (
    ReviewHistoryItem,
    UserPersonaRequest,
    UserPersonaWithHistory,
    ItemDetailsRequest,
    Message,
    SimulateReviewRequest,
    SimulateReviewResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendedItem,
)

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────

app = FastAPI(
    title="PulseAgent API",
    description="DSN × BCT Hackathon 3.0 — Team HOKM | User Modeling + Recommendation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────

# class ReviewHistoryItem(BaseModel):
#     item_id:   str
#     category:  str
#     rating:    float = Field(..., ge=1.0, le=5.0)
#     text:      str
#     timestamp: Optional[str] = None


# class UserPersonaRequest(BaseModel):
#     user_id:              str
#     review_history:       List[ReviewHistoryItem] = []
#     avg_rating:           Optional[float]         = None
#     preferred_categories: List[str]               = []
#     tone_profile:         str                     = "mixed"


# class ItemDetailsRequest(BaseModel):
#     item_id:    str
#     name:       str
#     category:   str
#     attributes: Dict[str, Any] = {}
#     avg_rating: Optional[float] = None


# class Message(BaseModel):
#     role:    str
#     content: str


# class UserPersonaWithHistory(UserPersonaRequest):
#     conversation_history: List[Message] = []


# # ── Task A ──────────────────────────────

# class SimulateReviewRequest(BaseModel):
#     user_persona: UserPersonaRequest
#     item_details: ItemDetailsRequest


# class SimulateReviewResponse(BaseModel):
#     simulated_rating: float
#     simulated_review: str
#     confidence:       float
#     reasoning_trace:  List[str]


# # ── Task B ──────────────────────────────

# class RecommendRequest(BaseModel):
#     user_persona: UserPersonaWithHistory


# class RecommendedItem(BaseModel):
#     rank:             int
#     item_id:          str
#     name:             str
#     category:         str
#     predicted_rating: float
#     explanation:      str
#     ndcg_score:       Optional[float] = None


class RecommendResponse(BaseModel):
    recommendations: List[RecommendedItem]
    inferred_intent: str
    cold_start:      bool
    reasoning_trace: List[str]


# ─────────────────────────────────────────
# Health check
# ─────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "PulseAgent is running", "team": "HOKM"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "timestamp": time.time()}


# ─────────────────────────────────────────
# Task A — POST /simulate-review
# ─────────────────────────────────────────

@app.post(
    "/simulate-review",
    response_model=SimulateReviewResponse,
    tags=["Task A — User Modeling"],
)
async def simulate_review(request: SimulateReviewRequest):
    """
    Given a user persona and an unseen item, simulate how that user
    would rate and review the item — capturing their tone, rating
    behaviour, and writing style.
    """
    try:
        persona_dict = {
            "user_id": request.user_persona.user_id,
            "review_history": [
                {
                    "item_id":   r.item_id,
                    "category":  r.category,
                    "rating":    r.rating,
                    "text":      r.text,
                    "timestamp": r.timestamp,
                }
                for r in request.user_persona.review_history
            ],
            "avg_rating":           request.user_persona.avg_rating,
            "preferred_categories": request.user_persona.preferred_categories,
            "tone_profile":         request.user_persona.tone_profile,
            "conversation_history": [],
        }

        item_dict = {
            "item_id":    request.item_details.item_id,
            "name":       request.item_details.name,
            "category":   request.item_details.category,
            "attributes": request.item_details.attributes,
            "avg_rating": request.item_details.avg_rating,
        }

        result = await run_task_a(persona_dict, item_dict)

        return SimulateReviewResponse(
            simulated_rating=result.get("simulated_rating", 3.0),
            simulated_review=result.get("simulated_review", ""),
            confidence=result.get("review_quality_score", 0.0),
            reasoning_trace=result.get("reasoning_trace", []),
        )

    except Exception as exc:
        logger.error("simulate_review failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────
# Task B — POST /recommend
# ─────────────────────────────────────────

@app.post(
    "/recommend",
    response_model=RecommendResponse,
    tags=["Task B — Recommendation"],
)
async def recommend(request: RecommendRequest):
    """
    Given a user persona and optional conversation context, return a
    ranked list of personalised recommendations. Handles cold-start,
    cross-domain, and multi-turn scenarios.
    """
    try:
        persona_dict = {
            "user_id": request.user_persona.user_id,
            "review_history": [
                {
                    "item_id":   r.item_id,
                    "category":  r.category,
                    "rating":    r.rating,
                    "text":      r.text,
                    "timestamp": r.timestamp,
                }
                for r in request.user_persona.review_history
            ],
            "avg_rating":           request.user_persona.avg_rating,
            "preferred_categories": request.user_persona.preferred_categories,
            "tone_profile":         request.user_persona.tone_profile,
            "conversation_history": [
                {"role": m.role, "content": m.content}
                for m in request.user_persona.conversation_history
            ],
        }

        result = await run_task_b(persona_dict)

        recs = result.get("ranked_recommendations", [])
        recommendations = [
            RecommendedItem(
                rank=             r.rank             if hasattr(r, "rank")             else r.get("rank", i + 1),
                item_id=          r.item_id          if hasattr(r, "item_id")          else r.get("item_id", ""),
                name=             r.name             if hasattr(r, "name")             else r.get("name", ""),
                category=         r.category         if hasattr(r, "category")         else r.get("category", ""),
                predicted_rating= r.predicted_rating if hasattr(r, "predicted_rating") else r.get("predicted_rating", 3.0),
                explanation=      r.explanation      if hasattr(r, "explanation")      else r.get("explanation", ""),
                ndcg_score=       r.ndcg_score       if hasattr(r, "ndcg_score")       else r.get("ndcg_score"),
            )
            for i, r in enumerate(recs)
        ]

        return RecommendResponse(
            recommendations=recommendations,
            inferred_intent=result.get("inferred_intent", ""),
            cold_start=result.get("user_state").contextual.is_cold_start
                if result.get("user_state") else False,
            reasoning_trace=result.get("reasoning_trace", []),
        )

    except Exception as exc:
        logger.error("recommend failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))