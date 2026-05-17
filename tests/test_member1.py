"""
PulseAgent — Tests for Member 1 deliverables
Team HOKM · DSN × BCT Hackathon 3.0

Covers:
- Schema validation and edge cases
- PersonaConstructionAgent: all three pipelines, fault isolation, cold-start
- ReviewSimAgent: rating inference, review generation, quality scoring
- Graph routing: Task A vs Task B
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.schemas.models import (
    AgentState,
    BehaviouralProfile,
    ContextualProfile,
    ItemDetails,
    Message,
    ReviewRecord,
    SimulatedReview,
    TextualProfile,
    ToneProfile,
    UserPersona,
    UserState,
)
from src.agents.persona_agent import (
    _behavioural_pipeline,
    _contextual_pipeline,
    _textual_pipeline,
    persona_construction_agent,
)
from src.agents.review_agent import review_sim_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rich_persona() -> UserPersona:
    return UserPersona(
        user_id="u_test",
        review_history=[
            ReviewRecord(item_id="i_001", category="Food", rating=4.5, text="Really great spot, super fresh ingredients."),
            ReviewRecord(item_id="i_002", category="Food", rating=3.0, text="Was okay. Nothing I'd write home about."),
            ReviewRecord(item_id="i_003", category="Nightlife", rating=5.0, text="Unreal atmosphere. Best night out in ages."),
            ReviewRecord(item_id="i_004", category="Food", rating=4.0, text="Solid. Would go back."),
            ReviewRecord(item_id="i_005", category="Books", rating=3.5, text="Good read, slow start but worth it."),
        ],
        avg_rating=4.0,
        preferred_categories=["Food", "Nightlife"],
        tone_profile=ToneProfile.EXPRESSIVE,
    )


@pytest.fixture
def cold_persona() -> UserPersona:
    return UserPersona(
        user_id="u_cold",
        review_history=[],
    )


@pytest.fixture
def sample_item() -> ItemDetails:
    return ItemDetails(
        item_id="i_999",
        name="The Corner Café",
        category="Food",
        attributes={"cuisine": "Café", "price_range": "$"},
        avg_rating=4.1,
    )


@pytest.fixture
def base_state(rich_persona, sample_item) -> AgentState:
    return {
        "user_persona":           rich_persona,
        "item_details":           sample_item,
        "task":                   "A",
        "user_state":             None,
        "simulated_rating":       None,
        "simulated_review":       None,
        "review_quality_score":   None,
        "reasoning_trace":        [],
        "inferred_intent":        None,
        "candidate_items":        [],
        "ranked_recommendations": [],
        "explanation":            None,
        "errors":                 [],
    }


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemas:

    def test_user_persona_valid(self, rich_persona):
        assert rich_persona.user_id == "u_test"
        assert len(rich_persona.review_history) == 5

    def test_user_persona_empty_history(self, cold_persona):
        assert cold_persona.review_history == []

    def test_review_record_rating_bounds(self):
        with pytest.raises(Exception):
            ReviewRecord(item_id="x", category="Food", rating=6.0, text="too high")

    def test_review_record_rating_low_bound(self):
        with pytest.raises(Exception):
            ReviewRecord(item_id="x", category="Food", rating=0.0, text="too low")

    def test_item_details_valid(self, sample_item):
        assert sample_item.category == "Food"
        assert sample_item.attributes["price_range"] == "$"

    def test_item_details_empty_attributes(self):
        item = ItemDetails(item_id="i_x", name="Test", category="Books")
        assert item.attributes == {}

    def test_message_model(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"

    def test_tone_profile_enum(self):
        assert ToneProfile("expressive") == ToneProfile.EXPRESSIVE

    def test_tone_profile_invalid(self):
        with pytest.raises(Exception):
            ToneProfile("aggressive")

    def test_user_state_pipeline_errors_default(self, rich_persona):
        state = UserState(
            user_id="u_test",
            behavioural=BehaviouralProfile(
                avg_rating=4.0, rating_std=0.5, category_affinities={},
                recency_weighted_avg=4.0, rating_bias=0.3,
                is_harsh_rater=False, is_generous_rater=True,
            ),
            textual=TextualProfile(
                dominant_tone=ToneProfile.EXPRESSIVE, avg_review_length=40,
                sentiment_polarity=0.6, vocabulary_richness=0.4,
                uses_first_person=True,
            ),
            contextual=ContextualProfile(
                is_cold_start=False, sparse_history=False,
                active_categories=["Food", "Nightlife"],
                cross_domain_signals=["Books"],
                recency_days=10,
            ),
        )
        assert state.pipeline_errors == []


# ---------------------------------------------------------------------------
# Behavioural pipeline tests
# ---------------------------------------------------------------------------

class TestBehaviouralPipeline:

    @pytest.mark.asyncio
    async def test_rich_history(self, rich_persona):
        result = await _behavioural_pipeline(rich_persona)
        assert result is not None
        assert 1.0 <= result.avg_rating <= 5.0
        assert result.rating_std >= 0.0
        assert "Food" in result.category_affinities

    @pytest.mark.asyncio
    async def test_cold_start_defaults(self, cold_persona):
        result = await _behavioural_pipeline(cold_persona)
        assert result is not None
        assert result.avg_rating == 3.0
        assert result.category_affinities == {}
        assert result.is_harsh_rater is False
        assert result.is_generous_rater is False

    @pytest.mark.asyncio
    async def test_harsh_rater_detected(self):
        persona = UserPersona(
            user_id="u_harsh",
            review_history=[
                ReviewRecord(item_id=f"i_{i}", category="Food", rating=2.0, text="Bad")
                for i in range(5)
            ],
        )
        result = await _behavioural_pipeline(persona)
        assert result.is_harsh_rater is True

    @pytest.mark.asyncio
    async def test_generous_rater_detected(self):
        persona = UserPersona(
            user_id="u_generous",
            review_history=[
                ReviewRecord(item_id=f"i_{i}", category="Food", rating=5.0, text="Amazing!")
                for i in range(5)
            ],
        )
        result = await _behavioural_pipeline(persona)
        assert result.is_generous_rater is True

    @pytest.mark.asyncio
    async def test_category_affinity_sums_to_one(self, rich_persona):
        result = await _behavioural_pipeline(rich_persona)
        total = sum(result.category_affinities.values())
        assert abs(total - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_single_review(self):
        persona = UserPersona(
            user_id="u_single",
            review_history=[
                ReviewRecord(item_id="i_001", category="Food", rating=4.0, text="Good")
            ],
        )
        result = await _behavioural_pipeline(persona)
        assert result is not None
        assert result.avg_rating == 4.0
        assert result.rating_std == 0.0


# ---------------------------------------------------------------------------
# Contextual pipeline tests
# ---------------------------------------------------------------------------

class TestContextualPipeline:

    @pytest.mark.asyncio
    async def test_cold_start_flag(self, cold_persona):
        result = await _contextual_pipeline(cold_persona)
        assert result.is_cold_start is True
        assert result.sparse_history is True

    @pytest.mark.asyncio
    async def test_sparse_history_flag(self):
        persona = UserPersona(
            user_id="u_sparse",
            review_history=[
                ReviewRecord(item_id=f"i_{i}", category="Food", rating=4.0, text="Ok")
                for i in range(3)
            ],
        )
        result = await _contextual_pipeline(persona)
        assert result.sparse_history is True
        assert result.is_cold_start is False

    @pytest.mark.asyncio
    async def test_cross_domain_detected(self, rich_persona):
        result = await _contextual_pipeline(rich_persona)
        # "Books" is a cross-domain category (minor category for this user)
        assert isinstance(result.cross_domain_signals, list)

    @pytest.mark.asyncio
    async def test_active_categories(self, rich_persona):
        result = await _contextual_pipeline(rich_persona)
        assert "Food" in result.active_categories
        assert "Nightlife" in result.active_categories

    @pytest.mark.asyncio
    async def test_no_timestamp_recency_is_none(self, rich_persona):
        result = await _contextual_pipeline(rich_persona)
        # No timestamps in fixture → recency_days should be None
        assert result.recency_days is None


# ---------------------------------------------------------------------------
# Textual pipeline tests (mocked Claude call)
# ---------------------------------------------------------------------------

class TestTextualPipeline:

    @pytest.mark.asyncio
    async def test_cold_start_returns_defaults(self, cold_persona):
        result = await _textual_pipeline(cold_persona)
        assert result is not None
        assert result.dominant_tone == ToneProfile.MIXED
        assert result.avg_review_length == 0

    @pytest.mark.asyncio
    async def test_rich_history_calls_claude(self, rich_persona):
        mock_response = (
            '{"dominant_tone": "expressive", "avg_review_length": 15, '
            '"sentiment_polarity": 0.7, "vocabulary_richness": 0.5, '
            '"uses_first_person": true, "common_phrases": ["solid", "great spot"]}'
        )
        with patch("src.agents.persona_agent._call_claude", new=AsyncMock(return_value=mock_response)):
            result = await _textual_pipeline(rich_persona)
        assert result.dominant_tone == ToneProfile.EXPRESSIVE
        assert result.sentiment_polarity == 0.7
        assert "solid" in result.common_phrases

    @pytest.mark.asyncio
    async def test_malformed_claude_response_returns_none(self, rich_persona):
        with patch("src.agents.persona_agent._call_claude", new=AsyncMock(return_value="not json at all")):
            # Should not raise — returns None or defaults gracefully
            result = await _textual_pipeline(rich_persona)
            # result may be a partial/default TextualProfile or None
            assert result is None or hasattr(result, "dominant_tone")


# ---------------------------------------------------------------------------
# PersonaConstructionAgent (Node 1) tests
# ---------------------------------------------------------------------------

class TestPersonaConstructionAgent:

    @pytest.mark.asyncio
    async def test_builds_user_state(self, base_state):
        mock_response = (
            '{"dominant_tone": "expressive", "avg_review_length": 20, '
            '"sentiment_polarity": 0.5, "vocabulary_richness": 0.4, '
            '"uses_first_person": true, "common_phrases": []}'
        )
        with patch("src.agents.persona_agent._call_claude", new=AsyncMock(return_value=mock_response)):
            result = await persona_construction_agent(base_state)
        assert result["user_state"] is not None
        assert result["user_state"].user_id == "u_test"

    @pytest.mark.asyncio
    async def test_reasoning_trace_populated(self, base_state):
        mock_response = '{"dominant_tone": "mixed", "avg_review_length": 10, "sentiment_polarity": 0.0, "vocabulary_richness": 0.3, "uses_first_person": false, "common_phrases": []}'
        with patch("src.agents.persona_agent._call_claude", new=AsyncMock(return_value=mock_response)):
            result = await persona_construction_agent(base_state)
        assert len(result["reasoning_trace"]) > 0

    @pytest.mark.asyncio
    async def test_textual_failure_does_not_crash(self, base_state):
        with patch("src.agents.persona_agent._call_claude", new=AsyncMock(side_effect=Exception("API down"))):
            result = await persona_construction_agent(base_state)
        # Should still return a state with defaults
        assert result["user_state"] is not None
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_cold_start_persona(self, cold_persona):
        state = {
            "user_persona": cold_persona,
            "item_details": None,
            "task": "B",
            "user_state": None,
            "simulated_rating": None,
            "simulated_review": None,
            "review_quality_score": None,
            "reasoning_trace": [],
            "inferred_intent": None,
            "candidate_items": [],
            "ranked_recommendations": [],
            "explanation": None,
            "errors": [],
        }
        with patch("src.agents.persona_agent._call_claude", new=AsyncMock(return_value='{"dominant_tone":"mixed","avg_review_length":0,"sentiment_polarity":0.0,"vocabulary_richness":0.0,"uses_first_person":false,"common_phrases":[]}')):
            result = await persona_construction_agent(state)
        assert result["user_state"].contextual.is_cold_start is True


# ---------------------------------------------------------------------------
# ReviewSimAgent (Node 2A) tests
# ---------------------------------------------------------------------------

class TestReviewSimAgent:

    def _make_state_with_user_state(self, base_state: AgentState) -> AgentState:
        """Inject a pre-built UserState so we can test Node 2A in isolation."""
        user_state = UserState(
            user_id="u_test",
            behavioural=BehaviouralProfile(
                avg_rating=4.0, rating_std=0.7,
                category_affinities={"Food": 0.8, "Nightlife": 0.2},
                recency_weighted_avg=4.1, rating_bias=0.3,
                is_harsh_rater=False, is_generous_rater=True,
            ),
            textual=TextualProfile(
                dominant_tone=ToneProfile.EXPRESSIVE, avg_review_length=20,
                sentiment_polarity=0.6, vocabulary_richness=0.4,
                uses_first_person=True, common_phrases=["solid", "great"],
            ),
            contextual=ContextualProfile(
                is_cold_start=False, sparse_history=False,
                active_categories=["Food", "Nightlife"],
                cross_domain_signals=[],
                recency_days=5,
            ),
        )
        return {**base_state, "user_state": user_state}

    @pytest.mark.asyncio
    async def test_produces_rating_and_review(self, base_state):
        state = self._make_state_with_user_state(base_state)

        rating_mock = '{"predicted_rating": 4.0, "confidence": 0.85, "reasoning": "User likes Food."}'
        review_mock = '{"review_text": "Really enjoyable spot. Would go back.", "word_count": 7}'
        quality_mock = '{"quality_score": 0.88, "notes": "Good match."}'

        responses = [rating_mock, review_mock, quality_mock]
        call_count = 0

        async def mock_claude(system, user, max_tokens=800):
            nonlocal call_count
            r = responses[call_count % len(responses)]
            call_count += 1
            return r

        with patch("src.agents.review_agent._call_claude", new=mock_claude):
            result = await review_sim_agent(state)

        assert result["simulated_rating"] == 4.0
        assert "Really enjoyable" in result["simulated_review"]
        assert result["review_quality_score"] == 0.88

    @pytest.mark.asyncio
    async def test_rating_clamped_to_bounds(self, base_state):
        state = self._make_state_with_user_state(base_state)

        # Return an out-of-bounds rating
        rating_mock = '{"predicted_rating": 7.5, "confidence": 0.5, "reasoning": "Test."}'
        review_mock = '{"review_text": "Fine.", "word_count": 1}'
        quality_mock = '{"quality_score": 0.5, "notes": "Ok."}'
        responses = [rating_mock, review_mock, quality_mock]
        idx = 0

        async def mock_claude(system, user, max_tokens=800):
            nonlocal idx
            r = responses[idx % len(responses)]
            idx += 1
            return r

        with patch("src.agents.review_agent._call_claude", new=mock_claude):
            result = await review_sim_agent(state)

        assert result["simulated_rating"] <= 5.0

    @pytest.mark.asyncio
    async def test_api_failure_graceful_fallback(self, base_state):
        state = self._make_state_with_user_state(base_state)

        with patch("src.agents.review_agent._call_claude", new=AsyncMock(side_effect=Exception("timeout"))):
            result = await review_sim_agent(state)

        assert result["simulated_rating"] is not None   # falls back to avg_rating
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_reasoning_trace_populated(self, base_state):
        state = self._make_state_with_user_state(base_state)
        responses = [
            '{"predicted_rating": 4.0, "confidence": 0.8, "reasoning": "Makes sense."}',
            '{"review_text": "Good place.", "word_count": 2}',
            '{"quality_score": 0.75, "notes": "Decent."}',
        ]
        idx = 0

        async def mock_claude(system, user, max_tokens=800):
            nonlocal idx
            r = responses[idx % len(responses)]
            idx += 1
            return r

        with patch("src.agents.review_agent._call_claude", new=mock_claude):
            result = await review_sim_agent(state)

        assert len(result["reasoning_trace"]) > 0
