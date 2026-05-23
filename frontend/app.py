"""
PulseAgent — Streamlit Frontend
Team HOKM · DSN × BCT Hackathon 3.0

Run locally:
streamlit run frontend/app.py

Requires the FastAPI backend running at http://localhost:8000
"""

from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ——————————————————————————————————————————
# Page config
# ——————————————————————————————————————————

st.set_page_config(
    page_title="PulseAgent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ——————————————————————————————————————————
# Styling
# ——————————————————————————————————————————

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
    .review-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 1.2rem;
        font-style: italic;
        font-size: 1.05rem;
        margin: 1rem 0;
        border-left: 4px solid #2E4057;
    }
    .trace-item {
        font-size: 0.85rem;
        color: #555;
        padding: 0.2rem 0;
    }
    .empty-state {
        text-align: center;
        color: #999;
        padding: 2rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ——————————————————————————————————————————
# Header
# ——————————————————————————————————————————

st.markdown('<div class="main-header">🧠 PulseAgent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">DSN × BCT Hackathon 3.0 — Team HOKM | Review Simulation</div>', unsafe_allow_html=True)

# ——————————————————————————————————————————
# Sidebar — User Persona Builder
# ——————————————————————————————————————————

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

    st.divider()
    st.subheader("📋 Review History")
    st.caption("These past reviews define the user's taste and writing style.")

    if "review_history" not in st.session_state:
        st.session_state.review_history = [
            {"item_id": "i_001", "category": "Food",      "rating": 4.0, "text": "Really solid spot. Food was fresh and service was quick."},
            {"item_id": "i_002", "category": "Food",      "rating": 3.5, "text": "Decent enough but nothing special. Wouldn't rush back."},
            {"item_id": "i_003", "category": "Nightlife", "rating": 4.5, "text": "Great atmosphere. Exactly what I needed."},
        ]

    if st.session_state.review_history:
        for i, review in enumerate(st.session_state.review_history):
            with st.expander(f"Review {i+1} — {review['category']} ({review['rating']}★)"):
                st.write(review["text"])
                if st.button("🗑 Remove", key=f"remove_{i}"):
                    st.session_state.review_history.pop(i)
                    st.rerun()
    else:
        st.markdown('<div class="empty-state">No reviews yet. Add one below.</div>', unsafe_allow_html=True)

    st.divider()
    with st.expander("➕ Add a Review"):
        new_category = st.selectbox("Category", ["Food", "Nightlife", "Books", "Movies", "Shopping", "Outdoors", "Arts"], key="new_cat")
        new_rating   = st.slider("Rating", 1.0, 5.0, 4.0, 0.5, key="new_rating")
        new_text     = st.text_area("Review text", placeholder="Write the review here...", key="new_text")
        if st.button("Add Review", type="primary"):
            if new_text.strip():
                st.session_state.review_history.append({
                    "item_id":  f"i_{len(st.session_state.review_history)+1:03d}",
                    "category": new_category,
                    "rating":   new_rating,
                    "text":     new_text.strip(),
                })
                st.success("Review added!")
                st.rerun()
            else:
                st.warning("Please write some review text first.")

    if st.button("🗑 Reset All Reviews"):
        st.session_state.review_history = []
        st.rerun()

# ——————————————————————————————————————————
# Build persona dict
# ——————————————————————————————————————————

def build_persona():
    avg = (
        sum(r["rating"] for r in st.session_state.review_history) /
        len(st.session_state.review_history)
    ) if st.session_state.review_history else 3.5

    return {
        "user_id":              user_id,
        "review_history":       st.session_state.review_history,
        "avg_rating":           round(avg, 2),
        "preferred_categories": preferred_categories,
        "tone_profile":         tone_profile,
        "conversation_history": [],
    }

# ——————————————————————————————————————————
# Main — Simulate Review
# ——————————————————————————————————————————

st.subheader("📝 Simulate a Review")
st.caption("Fill in the item details below. The agent will predict how this user would rate and review it based on their persona.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Item Details**")
    item_name     = st.text_input("Item Name",     value="The Grill House")
    item_id       = st.text_input("Item ID",       value="i_999")
    item_category = st.selectbox("Item Category",  ["Food", "Nightlife", "Books", "Movies", "Shopping", "Outdoors", "Arts"])

with col2:
    st.markdown("**Item Attributes**")
    cuisine  = st.text_input("Cuisine / Type",     value="American")
    price    = st.selectbox("Price Range",         ["$", "$$", "$$$"])
    item_avg = st.slider("Platform Avg Rating ★", 1.0, 5.0, 4.0, 0.1)

st.divider()

# Persona summary before submitting
with st.expander("👁 Preview Persona Being Sent"):
    st.json(build_persona())

run_a = st.button("▶ Simulate Review", type="primary", use_container_width=True)

if run_a:
    if not st.session_state.review_history:
        st.warning("Add at least one review to the persona before simulating.")
    else:
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

                st.divider()
                st.subheader("✅ Simulation Result")

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Simulated Rating",   f"★ {data['simulated_rating']}")
                col_r2.metric("Confidence",         f"{data['confidence']:.0%}")
                col_r3.metric("Reviews in Persona", len(st.session_state.review_history))

                st.markdown("**Generated Review**")
                review_text = data.get("simulated_review") or "_No review text returned._"
                st.markdown(f'<div class="review-box">{review_text}</div>', unsafe_allow_html=True)

                with st.expander("🔍 Reasoning Trace"):
                    trace = data.get("reasoning_trace", [])
                    if trace:
                        for step in trace:
                            st.markdown(f'<div class="trace-item">• {step}</div>', unsafe_allow_html=True)
                    else:
                        st.write("No trace available.")

                with st.expander("📋 Raw JSON"):
                    st.json(data)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure the FastAPI server is running at http://localhost:8000")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")