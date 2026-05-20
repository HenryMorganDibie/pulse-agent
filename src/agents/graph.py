"""
PulseAgent — LangGraph Graph Construction
Team HOKM · DSN × BCT Hackathon 3.0

Wires the full pipeline for both Task A and Task B.
Node 1 (PersonaConstructionAgent) is shared.
Task routing happens after Node 1 based on state["task"].
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.persona_agent import persona_construction_agent
from src.agents.review_agent import review_sim_agent
from src.schemas.models import AgentState
from src.agents.reasoning_agent import reasoning_agent
from src.agents.ranking_agent import ranking_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_persona(state: AgentState) -> Literal["review_sim_agent", "reasoning_agent"]:
    """
    After Node 1 completes, route to Task A or Task B based on state["task"].
    """
    task = state.get("task", "A")
    if task == "A":
        logger.info("Graph: routing to Task A (ReviewSimAgent)")
        return "review_sim_agent"
    logger.info("Graph: routing to Task B (ReasoningAgent)")
    return "reasoning_agent"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Constructs and compiles the full PulseAgent LangGraph pipeline.

    Graph topology:
        START
          └─► persona_construction_agent (Node 1, shared)
                ├─► review_sim_agent (Node 2A — Task A)
                │       └─► END
                └─► reasoning_agent (Node 2B — Task B)
                        └─► ranking_agent (Node 3B — Task B)
                                └─► END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("persona_construction_agent", persona_construction_agent)
    graph.add_node("review_sim_agent", review_sim_agent)
    graph.add_node("reasoning_agent", reasoning_agent)
    graph.add_node("ranking_agent", ranking_agent)

    # Edges
    graph.add_edge(START, "persona_construction_agent")

    graph.add_conditional_edges(
        "persona_construction_agent",
        _route_after_persona,
        {
            "review_sim_agent": "review_sim_agent",
            "reasoning_agent":  "reasoning_agent",
        },
    )

    graph.add_edge("review_sim_agent", END)
    graph.add_edge("reasoning_agent",  "ranking_agent")
    graph.add_edge("ranking_agent",    END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public run functions
# ---------------------------------------------------------------------------

async def run_task_a(
    user_persona: dict,
    item_details: dict,
) -> dict:
    """
    Run the full Task A pipeline (user modeling + review simulation).

    Args:
        user_persona: dict matching UserPersona schema
        item_details: dict matching ItemDetails schema

    Returns:
        Final AgentState as a dict
    """
    from src.schemas.models import ItemDetails, UserPersona

    graph = build_graph()

    initial_state: AgentState = {
        "user_persona":           UserPersona(**user_persona),
        "item_details":           ItemDetails(**item_details),
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

    result = await graph.ainvoke(initial_state)
    return result


async def run_task_b(user_persona: dict) -> dict:
    """
    Run the full Task B pipeline (recommendation).

    Args:
        user_persona: dict matching UserPersona schema (may include conversation_history)

    Returns:
        Final AgentState as a dict
    """
    from src.schemas.models import UserPersona

    graph = build_graph()

    initial_state: AgentState = {
        "user_persona":           UserPersona(**user_persona),
        "item_details":           None,
        "task":                   "B",
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

    result = await graph.ainvoke(initial_state)
    return result