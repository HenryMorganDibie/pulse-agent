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

# ——————————————————————————————————————————
# Header
# ——————————————————————————————————————————

st.markdown('<div class="main-header">🧠 PulseAgent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">DSN × BCT Hackathon 3.0 — Team HOKM | User Modeling + Recommendation</div>', unsafe_allow_html=True)

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

    st.subheader("Review History")
    st.caption("Add past reviews to build the user's persona.")

    if "review_history" not in st.session_state:
        st.session_state.review_history = [
            {"item_id": "i_001", "category": "Food",      "rating": 4.0, "text": "Really solid spot. Food was fresh and service was quick."},
            {"item_id": "i_002", "category": "Food",      "rating": 3.5, "text": "Decent enough but nothing special. Wouldn't rush back."},
            {"item_id": "i_003", "category": "Nightlife", "rating": 4.5, "text": "Great atmosphere. Exactly what I needed."},
        ]

    for i, review in enumerate(st.session_state.review_history):
        with st.expander(f"Review {i+1} — {review['category']} ({review['rating']}★)"):
            st.text(review["text"][:120] + ("..." if len(review["text"]) > 120 else ""))

    with st.expander("➕ Add a review"):
        new_category = st.selectbox("Category", ["Food", "Nightlife", "Books", "Movies", "Shopping"], key="new_cat")
        new_rating   = st.slider("Rating", 1.0, 5.0, 4.0, 0.5, key="new_rating")
        new_text     = st.text_area("Review text", key="new_text")
        if st.button("Add Review"):
            if new_text.strip():
                st.session_state.review_history.append({
                    "item_id": f"i_{len(st.session_state.review_history)+1:03d}",
                    "category": new_category,
                    "rating": new_rating,
                    "text": new_text.strip(),
                })
                st.success("Review added to persona!")
            else:
                st.error("Review text cannot be empty.")