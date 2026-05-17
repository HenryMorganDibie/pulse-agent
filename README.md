# PulseAgent

An autonomous multi-agent system that models user behaviour from review history to simulate
how users would rate and review unseen items — and deliver personalised recommendations
that reason about intent, context, and preference before ranking.

Built for the DSN × BCT Hackathon 3.0 LLM Agent Challenge. Powered by Claude Sonnet and LangGraph.

---

## Architecture

```
UserPersona + ItemDetails / UserPersona
          │
          ▼
┌──────────────────────────────────────────────────┐
│            PersonaConstructionAgent              │  ← Three concurrent pipelines
│            (LangGraph Node 1)                    │
│                                                  │
│  ┌────────────────────┐  ┌─────────────────────┐ │
│  │  Behavioural       │  │  Textual             │ │
│  │  Signal Pipeline   │  │  Signal Pipeline     │ │
│  │                    │  │                      │ │
│  │  • Star patterns   │  │  • Review history    │ │
│  │  • Category drift  │  │  • Sentiment profile │ │
│  │  • Recency weights │  │  • Writing style     │ │
│  └────────────────────┘  └─────────────────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────────┐│
│  │          Contextual Signal Pipeline          ││
│  │   cold-start flags · category history ·      ││
│  │   cross-domain transitions · recency         ││
│  └──────────────────────────────────────────────┘│
│                                                  │
│  → Merge + build typed UserState                 │
└──────────────────────┬───────────────────────────┘
                       │ UserState
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐        ┌────────────────────┐
  │  TASK A PATH  │        │   TASK B PATH      │
  └───────┬───────┘        └─────────┬──────────┘
          │                          │
          ▼                          ▼
┌──────────────────┐      ┌──────────────────────┐
│  ReviewSimAgent  │      │  ReasoningAgent       │  ← LangGraph Node 2B
│  (Node 2A)       │      │                       │
│                  │      │  • Intent extraction  │
│  • Infer rating  │      │  • Cold-start probe   │
│  • Generate tone │      │  • Cross-domain bridge│
│  • Score quality │      │  • Multi-turn memory  │
└────────┬─────────┘      └──────────┬────────────┘
         │                           │
         │                           ▼
         │              ┌──────────────────────────┐
         │              │  RankingAgent             │  ← LangGraph Node 3B
         │              │                           │
         │              │  • Score by relevance     │
         │              │  • NDCG@10 optimised      │
         │              │  • Explanation generation │
         │              └──────────┬────────────────┘
         │                         │
         ▼                         ▼
  ┌──────────────┐       ┌─────────────────────┐
  │ SimulatedRev │       │  RankedRecommendation│
  │  (JSON/API)  │       │  List (JSON/API)     │
  └──────────────┘       └─────────────────────┘
```

---

## Tasks

### Task A — User Modeling
Given a user's review history and an unseen item, simulate the review that user would write —
capturing their star rating, tone, vocabulary, and reasoning.

### Task B — Recommendation
Given a user persona and optional conversation context, reason through what the user actually
wants and return a ranked list of personalised recommendations — handling cold-start and
cross-domain scenarios.

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/HenryMorganDibie/pulse-agent
cd pulse-agent
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Set API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run

**Task A — Simulate a review**

```bash
python main.py --task A \
    --user-id u_001 \
    --item-name "The Grill House" \
    --item-category "Food" \
    --output terminal
```

**Task B — Get recommendations**

```bash
python main.py --task B \
    --user-id u_042 \
    --query "something chill for the weekend" \
    --output terminal
```

**Export to JSON**

```bash
python main.py --task A \
    --user-id u_001 \
    --item-name "The Grill House" \
    --item-category "Food" \
    --output json \
    --out-file brief.json
```

---

## Output

### Terminal (Rich)

Colour-coded output with simulated rating, generated review text, quality score, and a
full reasoning trace showing every decision the agent made.

### JSON

```json
{
  "simulated_rating": 4.5,
  "simulated_review": "Honestly one of the better spots I've been to in a while. Food came out hot, service was solid. Worth the price.",
  "confidence": 0.87,
  "reasoning_trace": [
    "PersonaConstructionAgent: built UserState for u_001",
    "  behavioural.avg_rating=4.0",
    "  textual.dominant_tone=expressive",
    "  contextual.is_cold_start=False",
    "Rating anchor: user has reviewed Food before (affinity=0.80)",
    "Bias adjustment: +0.30 → adjusted anchor=4.21",
    "Claude rating inference: 4.5★ (confidence=0.87)",
    "Review generated: 18 words, tone=expressive",
    "Quality score: 0.87 — Strong match with user's established tone and length."
  ]
}
```

---

## Docker

```bash
# Build and run both API + Streamlit UI
docker-compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

---

## API

### Task A — `POST /simulate-review`

**Request**
```json
{
  "user_persona": {
    "user_id": "u_001",
    "review_history": [
      { "item_id": "i_042", "category": "Food", "rating": 4.0, "text": "Great spot, very consistent." }
    ],
    "avg_rating": 3.8,
    "preferred_categories": ["Food", "Nightlife"],
    "tone_profile": "expressive"
  },
  "item_details": {
    "item_id": "i_199",
    "name": "The Grill House",
    "category": "Food",
    "attributes": { "cuisine": "American", "price_range": "$$" }
  }
}
```

**Response**
```json
{
  "simulated_rating": 4.5,
  "simulated_review": "Honestly one of the better spots I've been to in a while.",
  "confidence": 0.87,
  "reasoning_trace": ["..."]
}
```

### Task B — `POST /recommend`

**Request**
```json
{
  "user_persona": {
    "user_id": "u_042",
    "review_history": [],
    "conversation_history": [
      { "role": "user", "content": "Something chill for the weekend, not too expensive" }
    ]
  }
}
```

**Response**
```json
{
  "recommendations": [
    {
      "rank": 1,
      "item_id": "i_305",
      "name": "The Rooftop Lounge",
      "category": "Nightlife",
      "predicted_rating": 4.3,
      "explanation": "Matches your preference for casual, mid-range spots and weekend social outings.",
      "ndcg_score": 0.91
    }
  ],
  "inferred_intent": "casual weekend outing, budget-conscious",
  "cold_start": false,
  "reasoning_trace": ["..."]
}
```

---

## Datasets

| Dataset | Source |
|---|---|
| Yelp | [yelp.com/dataset](https://yelp.com/dataset) |
| Amazon Reviews | [HuggingFace — McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) |
| Goodreads | [HuggingFace — mengzaiqiao/goodreads](https://huggingface.co/datasets/mengzaiqiao/goodreads) |

---

## Project Structure

```
pulse-agent/
│
├── main.py                          # CLI entrypoint
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── README.md
│
├── src/
│   ├── schemas/
│   │   └── models.py                # All Pydantic types (shared ground truth)
│   ├── agents/
│   │   ├── persona_agent.py         # Node 1: 3-pipeline persona construction
│   │   ├── review_agent.py          # Node 2A: rating + review simulation
│   │   ├── reasoning_agent.py       # Node 2B: intent + cold-start + multi-turn
│   │   ├── ranking_agent.py         # Node 3B: candidate scoring + ranking
│   │   └── graph.py                 # LangGraph graph construction
│   ├── tools/
│   │   ├── retrieval_tool.py        # Item candidate retrieval
│   │   └── embedding_tool.py        # Sentence embedding utility
│   ├── data/
│   │   ├── yelp_loader.py
│   │   ├── amazon_loader.py
│   │   ├── goodreads_loader.py
│   │   └── preprocessor.py
│   ├── evaluation/
│   │   ├── metrics.py               # ROUGE-L, BERTScore, RMSE, NDCG@10
│   │   └── runner.py
│   ├── api/
│   │   ├── main.py                  # FastAPI app
│   │   └── schemas.py
│   └── utils/
│       └── formatter.py
│
├── frontend/
│   └── app.py                       # Streamlit demo UI
│
└── tests/
    ├── test_schemas.py
    ├── test_persona_agent.py
    ├── test_review_agent.py
    ├── test_reasoning_agent.py
    ├── test_ranking_agent.py
    ├── test_data_loaders.py
    ├── test_metrics.py
    ├── test_api.py
    └── test_integration.py
```

---

## Key Design Decisions

**Three concurrent persona pipelines.** Behavioural signals, textual signals, and contextual
signals run as concurrent asyncio tasks. Each fails independently — if the textual pipeline
errors, the behavioural and contextual pipelines still complete and signal detection runs on
whatever was collected.

**Single shared UserState.** Both tasks derive from the same typed persona representation
built in Node 1. Task A and Task B are always working from the same understanding of the
user — no drift between the reviewer and the recommender.

**Cold-start as a first-class scenario.** The ReasoningAgent explicitly detects cold-start
conditions and switches strategy — falling back to demographic priors, cross-domain transfer,
or explicit clarifying questions in multi-turn mode.

**Structured outputs throughout.** Every object flowing through the LangGraph graph is typed
with Pydantic. The LLM returns JSON; the parser handles malformed responses gracefully
without crashing the pipeline.

**Auditable reasoning traces.** Every simulated review and every recommendation carries a
`reasoning_trace` — the model's chain-of-thought explaining why this rating, why this tone,
why this item. Judges can reconstruct every decision.

---

## Testing

```bash
pytest tests/ -v
```

---

## Team

| Member | Role | Ownership |
|---|---|---|
| Henry | Team Lead · Task A · Architecture | LangGraph orchestration, PersonaConstructionAgent, ReviewSimAgent, Docker, shared schemas |
| Kenneth | Task B · Data Engineering | Unified ingestion pipeline, cross-domain preprocessing, train/test datasets |
| Michael | Recommendation Systems | ReasoningAgent, RankingAgent, retrieval, cold-start handling |
| Kindness | API · Frontend · Solution Paper | FastAPI, Streamlit UI, testing, paper writing |