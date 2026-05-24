"""
PulseAgent — Shared Pydantic Types
Team HOKM · DSN × BCT Hackathon 3.0

All data structures flowing through the LangGraph pipeline are defined here.
This is the single source of truth for the entire team — every agent, tool,
API handler, and test imports from this file.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ToneProfile(str, Enum):
    EXPRESSIVE  = "expressive"   # strong opinions, emotional language
    ANALYTICAL  = "analytical"   # fact-forward, measured
    TERSE       = "terse"        # short, clipped sentences
    NARRATIVE   = "narrative"    # storytelling, context-heavy
    NIGERIAN    = "nigerian"     # Nigerian English — pidgin-influenced, warm, direct
    MIXED       = "mixed"        # no dominant pattern detected


class SignalType(str, Enum):
    BEHAVIOURAL = "behavioural"
    TEXTUAL     = "textual"
    CONTEXTUAL  = "contextual"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class ReviewRecord(BaseModel):
    """A single historical review written by the user."""
    item_id:  str
    category: str
    rating:   float = Field(..., ge=1.0, le=5.0)
    text:     str
    timestamp: Optional[str] = None          # ISO-8601 when available


class ItemDetails(BaseModel):
    """The unseen item to be reviewed (Task A only)."""
    item_id:    str
    name:       str
    category:   str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    avg_rating: Optional[float] = None       # platform aggregate if available


class Message(BaseModel):
    """A single turn in a multi-turn conversation (Task B)."""
    role:    str   # "user" | "assistant"
    content: str


class UserPersona(BaseModel):
    """
    Raw user input — review history, stated preferences, and
    any conversation context already collected.
    """
    user_id:               str
    review_history:        List[ReviewRecord]   = Field(default_factory=list)
    avg_rating:            Optional[float]      = None
    preferred_categories:  List[str]            = Field(default_factory=list)
    tone_profile:          ToneProfile          = ToneProfile.MIXED
    conversation_history:  List[Message]        = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Constructed user state (output of PersonaConstructionAgent)
# ---------------------------------------------------------------------------

class BehaviouralProfile(BaseModel):
    """Derived from star-rating patterns and category behaviour."""
    avg_rating:            float
    rating_std:            float
    category_affinities:   Dict[str, float]    # category → normalised weight
    recency_weighted_avg:  float
    rating_bias:           float               # user's offset from item avg
    is_harsh_rater:        bool
    is_generous_rater:     bool


class TextualProfile(BaseModel):
    """Derived from linguistic analysis of past reviews."""
    dominant_tone:         ToneProfile
    avg_review_length:     int                 # words
    sentiment_polarity:    float               # -1.0 to 1.0
    vocabulary_richness:   float               # type-token ratio
    uses_first_person:     bool
    common_phrases:        List[str]           = Field(default_factory=list)


class ContextualProfile(BaseModel):
    """Derived from temporal and cross-category patterns."""
    is_cold_start:         bool
    sparse_history:        bool                # < 5 reviews
    active_categories:     List[str]
    cross_domain_signals:  List[str]           # categories explored outside main cluster
    recency_days:          Optional[int]       # days since last review


class UserState(BaseModel):
    """
    Fully constructed user representation — output of Node 1.
    Shared by both Task A and Task B downstream agents.
    """
    user_id:        str
    behavioural:    BehaviouralProfile
    textual:        TextualProfile
    contextual:     ContextualProfile
    pipeline_errors: List[str] = Field(default_factory=list)  # non-fatal pipeline failures


# ---------------------------------------------------------------------------
# Task A output models
# ---------------------------------------------------------------------------

class SimulatedReview(BaseModel):
    """Output of ReviewSimAgent (Node 2A)."""
    simulated_rating:      float = Field(..., ge=1.0, le=5.0)
    simulated_review:      str
    confidence:            float = Field(..., ge=0.0, le=1.0)
    reasoning_trace:       List[str]


# ---------------------------------------------------------------------------
# Task B output models
# ---------------------------------------------------------------------------

class CandidateItem(BaseModel):
    """An item shortlisted by ReasoningAgent before final ranking."""
    item_id:          str
    name:             str
    category:         str
    attributes:       Dict[str, Any] = Field(default_factory=dict)
    retrieval_score:  float          # raw similarity / BM25 score
    predicted_rating: Optional[float] = None


class RankedItem(BaseModel):
    """A single ranked recommendation — output of RankingAgent (Node 3B)."""
    rank:             int
    item_id:          str
    name:             str
    category:         str
    predicted_rating: float
    explanation:      str
    ndcg_score:       Optional[float] = None


class RecommendationResult(BaseModel):
    """Full Task B output."""
    recommendations:  List[RankedItem]
    inferred_intent:  str
    cold_start:       bool
    reasoning_trace:  List[str]


# ---------------------------------------------------------------------------
# LangGraph AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The single typed state object that flows through every LangGraph node.
    Agents read from and write to this dict — nothing is passed as arguments.
    """

    # ── Inputs ──────────────────────────────────────────────────────────────
    user_persona:         UserPersona
    item_details:         Optional[ItemDetails]      # Task A only
    task:                 str                        # "A" | "B"

    # ── Node 1 output ────────────────────────────────────────────────────────
    user_state:           Optional[UserState]

    # ── Task A outputs (Node 2A) ─────────────────────────────────────────────
    simulated_rating:     Optional[float]
    simulated_review:     Optional[str]
    review_quality_score: Optional[float]
    reasoning_trace:      List[str]

    # ── Task B outputs (Nodes 2B + 3B) ───────────────────────────────────────
    inferred_intent:      Optional[str]
    candidate_items:      List[CandidateItem]
    ranked_recommendations: List[RankedItem]
    explanation:          Optional[str]

    # ── Meta ─────────────────────────────────────────────────────────────────
    errors:               List[str]
