"""
PulseAgent — Streamlit Frontend
Team HOKM · DSN × BCT Hackathon 3.0

Run locally:
    streamlit run frontend/app.py

Requires the FastAPI backend running at http://localhost:8000
"""

from __future__ import annotations

import json
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PulseAgent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E4057;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2E4057;
    }
    .review-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 1.2rem;
        font-style: italic;
        font-size: 1.05rem;
        margin: 1rem 0;
    }
    .trace-item {
        font-size: 0.85rem;
        color: #555;
        padding: 0.2rem 0;
    }
    .tag {
        display: inline-block;
        background: #2E4057;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">🧠 PulseAgent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">DSN × BCT Hackathon 3.0 — Team HOKM | User Modeling + Recommendation</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — User Persona Builder
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("👤 User Persona")

    user_id = st.text_input("User ID", value="u_001")
    tone_profile = st.selectbox(
        "Tone Profile",
        ["expressive", "analytical", "terse", "narrative", "mixed"],
        index=4,
    )
    preferred_categories = st.multiselect(
        "Preferred Categories",
        ["Food", "Nightlife", "Books", "Movies", "Shopping", "Outdoors", "Arts"],
        default=["Food", "Nightlife"],
    )

    st.subheader("Review History")
    st.caption("Add past reviews to build the user's persona.")

    if "review_history" not in st.session_state:
        st.session_state.review_history = [
            {"item_id": "i_001", "category": "Food",     "rating": 4.0, "text": "Really solid spot. Food was fresh and service was quick."},
            {"item_id": "i_002", "category": "Food",     "rating": 3.5, "text": "Decent enough but nothing special. Wouldn't rush back."},
            {"item_id": "i_003", "category": "Nightlife","rating": 4.5, "text": "Great atmosphere. Exactly what I needed."},
        ]

    for i, review in enumerate(st.session_state.review_history):
        with st.expander(f"Review {i+1} — {review['category']} ({review['rating']}★)"):
            st.text(review["text"][:120] + ("..." if len(review["text"]) > 120 else ""))

    with st.expander("➕ Add a review"):
        new_category = st.selectbox("Category", ["Food", "Nightlife", "Books", "Movies", "Shopping"], key="new_cat")
        new_rating   = st.slider("Rating", 1.0, 5.0, 4.0, 0.5, key="new_rating")
        new_text     = st.text_area("Review text", key="new_text")
        if st.button("Add review"):
            st.session_state.review_history.append({
                "item_id":  f"i_{len(st.session_state.review_history)+1:03}",
                "category": new_category,
                "rating":   new_rating,
                "text":     new_text,
            })
            st.success("Review added.")
            st.rerun()

    if st.button("🗑 Reset history"):
        st.session_state.review_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Build persona dict (shared by both tasks)
# ---------------------------------------------------------------------------

def build_persona(with_conversation=None):
    avg = (
        sum(r["rating"] for r in st.session_state.review_history) /
        len(st.session_state.review_history)
    ) if st.session_state.review_history else 3.5

    return {
        "user_id":               user_id,
        "review_history":        st.session_state.review_history,
        "avg_rating":            round(avg, 2),
        "preferred_categories":  preferred_categories,
        "tone_profile":          tone_profile,
        "conversation_history":  with_conversation or [],
    }

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_a, tab_b = st.tabs(["📝 Task A — Review Simulation", "🎯 Task B — Recommendation"])

# ============================================================
# TASK A
# ============================================================

with tab_a:
    st.subheader("Simulate a Review")
    st.caption("Given the user's history and an unseen item, predict how they would rate and review it.")

    col1, col2 = st.columns(2)

    with col1:
        item_name     = st.text_input("Item name",     value="The Grill House")
        item_category = st.selectbox("Item category",  ["Food", "Nightlife", "Books", "Movies", "Shopping", "Outdoors", "Arts"])
        item_id       = st.text_input("Item ID",       value="i_999")

    with col2:
        cuisine    = st.text_input("Cuisine / Type",  value="American")
        price      = st.selectbox("Price range",      ["$", "$$", "$$$"])
        item_avg   = st.slider("Item platform avg ★", 1.0, 5.0, 4.0, 0.1)

    run_a = st.button("▶ Simulate Review", type="primary", use_container_width=True)

    if run_a:
        payload = {
            "user_persona": build_persona(),
            "item_details": {
                "item_id":    item_id,
                "name":       item_name,
                "category":   item_category,
                "attributes": {"cuisine": cuisine, "price_range": price},
                "avg_rating": item_avg,
            },
        }

        with st.spinner("Running PersonaConstructionAgent + ReviewSimAgent..."):
            try:
                resp = requests.post(f"{API_BASE}/simulate-review", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()

                # Results
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Simulated Rating", f"★ {data['simulated_rating']}")
                col_r2.metric("Confidence", f"{data['confidence']:.0%}")
                col_r3.metric("Reviews in history", len(st.session_state.review_history))

                st.markdown("**Generated Review**")
                st.markdown(f'<div class="review-box">{data["simulated_review"]}</div>', unsafe_allow_html=True)

                with st.expander("🔍 Reasoning Trace"):
                    for step in data.get("reasoning_trace", []):
                        st.markdown(f'<div class="trace-item">• {step}</div>', unsafe_allow_html=True)

                with st.expander("📋 Raw JSON"):
                    st.json(data)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running at http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# TASK B
# ============================================================

with tab_b:
    st.subheader("Get Personalised Recommendations")
    st.caption("Tell the agent what you want. It will reason about your intent and return a ranked list.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Conversation display
    if st.session_state.conversation:
        st.markdown("**Conversation**")
        for msg in st.session_state.conversation:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

    # Input
    user_query = st.chat_input("What are you looking for? (e.g. 'something chill for the weekend')")

    if user_query:
        st.session_state.conversation.append({"role": "user", "content": user_query})

        persona = build_persona(with_conversation=st.session_state.conversation)
        payload = {"user_persona": persona}

        with st.spinner("Running ReasoningAgent + RankingAgent..."):
            try:
                resp = requests.post(f"{API_BASE}/recommend", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()

                # Intent + cold start
                col_i1, col_i2 = st.columns([3, 1])
                col_i1.info(f"**Intent:** {data.get('inferred_intent', '—')}")
                col_i2.warning("Cold-start" if data.get("cold_start") else "✅ Warm user")

                # Recommendations table
                recs = data.get("recommendations", [])
                if recs:
                    st.markdown("**Top Recommendations**")
                    for rec in recs:
                        with st.container():
                            c1, c2, c3 = st.columns([3, 1, 1])
                            c1.markdown(f"**{rec['rank']}. {rec['name']}**  \n{rec['explanation']}")
                            c2.markdown(f"★ {rec['predicted_rating']}")
                            c3.markdown(f"`{rec['category']}`")
                            st.divider()

                # Add assistant reply to conversation
                top3 = ", ".join(r["name"] for r in recs[:3]) if recs else "nothing found"
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Here are my top picks: {top3}. Intent detected: {data.get('inferred_intent', '—')}",
                })

                with st.expander("🔍 Reasoning Trace"):
                    for step in data.get("reasoning_trace", []):
                        st.markdown(f'<div class="trace-item">• {step}</div>', unsafe_allow_html=True)

                with st.expander("📋 Raw JSON"):
                    st.json(data)

                st.rerun()

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running at http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.conversation:
        if st.button("🗑 Clear conversation"):
            st.session_state.conversation = []
            st.rerun()