"""
PulseAgent — API Request/Response Schemas
Team HOKM · DSN × BCT Hackathon 3.0

Public-facing Pydantic models for the FastAPI layer.
These are separate from the internal schemas in src/schemas/models.py —
they define what callers send and receive, not what flows inside the pipeline.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────
# Shared
# ─────────────────────────────────────────

class ReviewHistoryItem(BaseModel):
    item_id:   str
    category:  str
    rating:    float = Field(..., ge=1.0, le=5.0)
    text:      str
    timestamp: Optional[str] = None


class Message(BaseModel):
    role:    str  # "user" | "assistant"
    content: str


class UserPersonaRequest(BaseModel):
    user_id:              str
    review_history:       List[ReviewHistoryItem] = Field(default_factory=list)
    avg_rating:           Optional[float]         = None
    preferred_categories: List[str]               = Field(default_factory=list)
    tone_profile:         str                     = "mixed"


class UserPersonaWithHistory(UserPersonaRequest):
    """UserPersona extended with conversation history for Task B multi-turn."""
    conversation_history: List[Message] = Field(default_factory=list)


class ItemDetailsRequest(BaseModel):
    item_id:    str
    name:       str
    category:   str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    avg_rating: Optional[float] = None


# ─────────────────────────────────────────
# Task A — /simulate-review
# ─────────────────────────────────────────

class SimulateReviewRequest(BaseModel):
    user_persona: UserPersonaRequest
    item_details: ItemDetailsRequest


class SimulateReviewResponse(BaseModel):
    simulated_rating: float = Field(..., ge=1.0, le=5.0)
    simulated_review: str
    confidence:       float = Field(..., ge=0.0, le=1.0)
    reasoning_trace:  List[str]


# ─────────────────────────────────────────
# Task B — /recommend
# ─────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_persona: UserPersonaWithHistory


class RecommendedItem(BaseModel):
    rank:             int
    item_id:          str
    name:             str
    category:         str
    predicted_rating: float = Field(..., ge=1.0, le=5.0)
    explanation:      str
    ndcg_score:       Optional[float] = None


class RecommendResponse(BaseModel):
    recommendations: List[RecommendedItem]
    inferred_intent: str
    cold_start:      bool
    reasoning_trace: List[str]


# ─────────────────────────────────────────
# Health
# ─────────────────────────────────────────

class HealthResponse(BaseModel):
    status:    str
    timestamp: float