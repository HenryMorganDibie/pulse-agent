"""
PulseAgent — RetrievalTool
Team HOKM · DSN × BCT Hackathon 3.0

Item candidate retrieval for Task B (RecommendationEngine).

Provides a mock catalog that works immediately without any dataset dependency.
Member 3 can swap in real data by calling load_catalog(items) at startup.

Retrieval strategy: category affinity scoring + keyword overlap + price affinity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.schemas.models import CandidateItem, UserState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock item catalog — covers multiple categories for cross-domain demos
# Member 3: call load_catalog(your_items) at startup to replace this.
# ---------------------------------------------------------------------------

_MOCK_CATALOG: List[Dict[str, Any]] = [
    # Food
    {"item_id": "i_001", "name": "The Grill House",          "category": "Food",      "attributes": {"cuisine": "American",  "price_range": "$$",  "vibe": "casual"}},
    {"item_id": "i_002", "name": "Sakura Ramen",             "category": "Food",      "attributes": {"cuisine": "Japanese",  "price_range": "$",   "vibe": "cozy"}},
    {"item_id": "i_003", "name": "La Bella Italia",          "category": "Food",      "attributes": {"cuisine": "Italian",   "price_range": "$$$", "vibe": "romantic"}},
    {"item_id": "i_004", "name": "Taco Libre",               "category": "Food",      "attributes": {"cuisine": "Mexican",   "price_range": "$",   "vibe": "lively"}},
    {"item_id": "i_005", "name": "The Vegan Corner",         "category": "Food",      "attributes": {"cuisine": "Vegan",     "price_range": "$$",  "vibe": "chill"}},
    {"item_id": "i_006", "name": "Burger Republic",          "category": "Food",      "attributes": {"cuisine": "American",  "price_range": "$",   "vibe": "casual"}},
    {"item_id": "i_007", "name": "Spice Garden",             "category": "Food",      "attributes": {"cuisine": "Indian",    "price_range": "$$",  "vibe": "warm"}},
    # Nightlife
    {"item_id": "i_101", "name": "The Rooftop Lounge",       "category": "Nightlife", "attributes": {"type": "bar",       "price_range": "$$",  "vibe": "chill"}},
    {"item_id": "i_102", "name": "Eclipse Nightclub",        "category": "Nightlife", "attributes": {"type": "club",      "price_range": "$$$", "vibe": "energetic"}},
    {"item_id": "i_103", "name": "The Jazz Cellar",          "category": "Nightlife", "attributes": {"type": "jazz bar",  "price_range": "$$",  "vibe": "sophisticated"}},
    {"item_id": "i_104", "name": "Brew & Board",             "category": "Nightlife", "attributes": {"type": "pub",       "price_range": "$",   "vibe": "casual"}},
    {"item_id": "i_105", "name": "Skyline Sky Bar",          "category": "Nightlife", "attributes": {"type": "bar",       "price_range": "$$$", "vibe": "upscale"}},
    # Books
    {"item_id": "i_201", "name": "Atomic Habits",            "category": "Books",     "attributes": {"genre": "Self-help", "author": "James Clear",   "pages": 320}},
    {"item_id": "i_202", "name": "The Midnight Library",     "category": "Books",     "attributes": {"genre": "Fiction",   "author": "Matt Haig",     "pages": 288}},
    {"item_id": "i_203", "name": "Dune",                     "category": "Books",     "attributes": {"genre": "Sci-fi",    "author": "Frank Herbert",  "pages": 688}},
    {"item_id": "i_204", "name": "Educated",                 "category": "Books",     "attributes": {"genre": "Memoir",    "author": "Tara Westover",  "pages": 352}},
    {"item_id": "i_205", "name": "Project Hail Mary",        "category": "Books",     "attributes": {"genre": "Sci-fi",    "author": "Andy Weir",      "pages": 480}},
    {"item_id": "i_206", "name": "The Alchemist",            "category": "Books",     "attributes": {"genre": "Fiction",   "author": "Paulo Coelho",   "pages": 208}},
    # Movies
    {"item_id": "i_301", "name": "Interstellar",             "category": "Movies",    "attributes": {"genre": "Sci-fi",    "director": "Nolan",    "runtime_min": 169}},
    {"item_id": "i_302", "name": "The Grand Budapest Hotel", "category": "Movies",    "attributes": {"genre": "Comedy",    "director": "Anderson", "runtime_min": 99}},
    {"item_id": "i_303", "name": "Parasite",                 "category": "Movies",    "attributes": {"genre": "Thriller",  "director": "Bong",     "runtime_min": 132}},
    {"item_id": "i_304", "name": "Everything Everywhere",    "category": "Movies",    "attributes": {"genre": "Comedy",    "director": "Daniels",  "runtime_min": 139}},
    # Shopping
    {"item_id": "i_401", "name": "Central Market",           "category": "Shopping",  "attributes": {"type": "market",      "price_range": "$",   "vibe": "busy"}},
    {"item_id": "i_402", "name": "The Vintage Store",        "category": "Shopping",  "attributes": {"type": "boutique",    "price_range": "$$",  "vibe": "quirky"}},
    {"item_id": "i_403", "name": "Tech Haven",               "category": "Shopping",  "attributes": {"type": "electronics", "price_range": "$$$", "vibe": "modern"}},
    # Outdoors
    {"item_id": "i_501", "name": "Riverside Park Trail",     "category": "Outdoors",  "attributes": {"type": "hiking",  "difficulty": "easy",   "duration_hr": 2}},
    {"item_id": "i_502", "name": "Summit Ridge Hike",        "category": "Outdoors",  "attributes": {"type": "hiking",  "difficulty": "hard",   "duration_hr": 6}},
    {"item_id": "i_503", "name": "City Cycling Route",       "category": "Outdoors",  "attributes": {"type": "cycling", "difficulty": "medium", "duration_hr": 3}},
    # Arts & Entertainment
    {"item_id": "i_601", "name": "City Art Museum",          "category": "Arts",      "attributes": {"type": "museum",  "price_range": "$$",  "vibe": "cultural"}},
    {"item_id": "i_602", "name": "Improv Comedy Night",      "category": "Arts",      "attributes": {"type": "comedy",  "price_range": "$",   "vibe": "fun"}},
    {"item_id": "i_603", "name": "Symphony at the Hall",     "category": "Arts",      "attributes": {"type": "music",   "price_range": "$$$", "vibe": "formal"}},
]

# Live catalog — replaced by Member 3's data at runtime via load_catalog()
_catalog: List[Dict[str, Any]] = list(_MOCK_CATALOG)


def load_catalog(items: List[Dict[str, Any]]) -> None:
    """
    Member 3 calls this at startup to replace the mock catalog with real data.

    Each item dict must have: item_id (str), name (str), category (str),
    attributes (dict). Optional: avg_rating (float).

    Example:
        from src.tools.retrieval_tool import load_catalog
        load_catalog(preprocessed_yelp_items)
    """
    global _catalog
    _catalog = items
    logger.info("RetrievalTool: catalog loaded with %d items", len(_catalog))


def get_catalog() -> List[Dict[str, Any]]:
    """Return the current catalog (mock or real)."""
    return _catalog


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _category_score(item: Dict[str, Any], user_state: UserState) -> float:
    """Score by how strongly the user prefers this item's category (0.0–1.0)."""
    category = item.get("category", "")
    affinities = user_state.behavioural.category_affinities

    if category in affinities:
        return affinities[category]

    # Cross-domain: category the user has explored but doesn't dominate
    if category in user_state.contextual.cross_domain_signals:
        return 0.3

    # Completely new category — small baseline so it remains surfaceable
    return 0.1


def _keyword_score(item: Dict[str, Any], query: str) -> float:
    """Lightweight keyword overlap between query and item fields (0.0–1.0)."""
    if not query:
        return 0.0

    query_tokens = set(query.lower().split())
    item_text = " ".join([
        item.get("name", ""),
        item.get("category", ""),
        *[str(v) for v in item.get("attributes", {}).values()],
    ]).lower()
    item_tokens = set(item_text.split())

    overlap = query_tokens & item_tokens
    return len(overlap) / len(query_tokens) if query_tokens else 0.0


def _price_affinity_score(item: Dict[str, Any], user_state: UserState) -> float:
    """Slight boost when price range matches the user's spending behaviour."""
    price_range = item.get("attributes", {}).get("price_range", "")
    is_generous = user_state.behavioural.is_generous_rater

    if price_range == "$$$" and is_generous:
        return 0.15
    if price_range == "$" and not is_generous:
        return 0.1
    return 0.0


# ---------------------------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------------------------

def retrieve_candidates(
    user_state: UserState,
    query: str = "",
    n: int = 20,
    category_filter: Optional[str] = None,
) -> List[CandidateItem]:
    """
    Retrieve the top-N candidate items for a user given an optional query.

    Scoring = weighted sum:
      - category affinity  × 0.50
      - keyword overlap    × 0.35
      - price affinity     × 0.15

    Args:
        user_state:       Constructed user profile from Node 1.
        query:            Free-text query or inferred intent string.
        n:                Maximum candidates to return.
        category_filter:  Restrict results to a single category (optional).

    Returns:
        List[CandidateItem] sorted by retrieval_score descending.
    """
    scored: List[tuple[float, Dict[str, Any]]] = []

    for item in _catalog:
        if category_filter and item.get("category") != category_filter:
            continue

        cat_s   = _category_score(item, user_state)
        kw_s    = _keyword_score(item, query)
        price_s = _price_affinity_score(item, user_state)

        total = (cat_s * 0.50) + (kw_s * 0.35) + (price_s * 0.15)
        scored.append((total, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    candidates: List[CandidateItem] = []
    for score, item in scored[:n]:
        candidates.append(CandidateItem(
            item_id=item["item_id"],
            name=item["name"],
            category=item["category"],
            attributes=item.get("attributes", {}),
            retrieval_score=round(score, 4),
            predicted_rating=None,  # RankingAgent fills this in
        ))

    logger.info(
        "RetrievalTool: %d candidates for user=%s query=%r",
        len(candidates), user_state.user_id, query[:60] if query else "",
    )
    return candidates
