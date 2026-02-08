# Predixion Debt Audit System

The development workflow for this assessment followed a rigorous "Learn-First, Build-Fast" methodology. The initial Minimum Viable Product (MVP) was constructed using a personal library of modular functions to rapidly establish the core architecture. To elevate the system to production standards, Large Language Models (LLMs) were utilized as a technical audit tool—specifically for identifying edge cases, enforcing strict error handling, and scanning for security vulnerabilities. Additionally, the concurrency model was refined through AI-assisted review to optimize asynchronous throughput, incorporating patterns inspired by Rust-based safety principles. Finally, the documentation was structured to ensure that all deployment and usage instructions are clear, reproducible, and verifiable.
An automated evaluation system for Indian debt collection voice agent conversations. It analyzes Hinglish (Hindi-English code-mixed) call transcripts and produces structured quality scores across five dimensions, aligned with RBI (Reserve Bank of India) Fair Practices Code.

## Architecture Overview

The system uses a **Multi-Pass Reasoning (MPR)** pipeline with three sequential stages:

```
Transcript (Hinglish JSON)
        |
        v
+-------------------+
| Pass 1: State     |  LLM extracts structured "State Map" from raw Hinglish
| Extraction        |  (intents, critical events, sentiment, compliance flags)
+-------------------+
        |
        v
+-------------------+
| Pass 2: Precedent |  State Map is vectorized (FastEmbed) and searched
| Retrieval         |  against Qdrant DB of pre-graded ground truth calls
+-------------------+
        |
        v
+-------------------+
| Pass 3: Delta     |  LLM compares current call vs. closest precedent
| Audit             |  to produce calibrated scores + explanations
+-------------------+
        |
        v
  AuditResult JSON (scores, flags, suggestions)
```

### Why Three Passes?

- **Pass 1** normalizes the noisy Hinglish transcript into a clean structured representation, reducing the complexity for downstream reasoning.
- **Pass 2** finds a calibration anchor — a similar call that was already human-graded — so the system doesn't score in a vacuum.
- **Pass 3** uses the precedent as a reference point, producing consistent scores even when the LLM's own baseline might vary.

## Evaluation Dimensions (1-5 scale)

| Dimension | What It Measures |
|-----------|-----------------|
| **Compliance & Ethics** | RBI guideline adherence, no threats/harassment, proper identity disclosure |
| **Goal Achievement** | Payment commitment secured, clear dates/amounts, callbacks scheduled |
| **Communication Quality** | Natural Hinglish, clear explanations of dues/charges, active listening |
| **Empathy & Tone** | Respect for hardship, professional yet caring, tone adaptation |
| **Objection Handling** | Handling excuses/disputes, offering solutions (EMI plans, settlements) |

## Tech Stack

- **Python 3.10+** — core runtime
- **Redis** — async job queue (conversations in, results out)
- **Qdrant** — vector database storing embedded ground truth precedents
- **OpenAI SDK** —  Due to gpu constraints - I had to use groq api, If cost and gpu werent an issue i would have prefered to use mistral-nemo or gpt 4o
- **FastEmbed** — local embedding model (`BAAI/bge-small-en-v1.5`) for vectorization
- **Pydantic** — all data contracts are validated models

## Project Structure

```
predixion/
├── docker-compose.yml          # Redis + Qdrant infrastructure
├── requirements.txt            # Pinned Python dependencies
├── .env                        # Environment configuration
├── main.py                     # CLI entry point (enqueue/worker/batch/results)
├── sample_conversations.json   # 10 sample Hinglish collection call transcripts
├── sample_outputs/             # Generated audit results
├── scripts/
│   └── seed_qdrant.py          # Loads ground truth into Qdrant vector DB
└── src/
    ├── core/
    │   ├── config.py           # pydantic-settings configuration
    │   └── schema.py           # All Pydantic data models
    ├── infra/
    │   └── clients.py          # Async singleton clients (Redis, Qdrant, OpenAI)
    ├── engine/
    │   ├── state_extractor.py  # Pass 1: Hinglish transcript -> StateMap
    │   ├── precedent_retriever.py  # Pass 2: StateMap -> nearest Ground Truth
    │   └── auditor.py          # Pass 3: Delta comparison -> AuditResult
    └── workers/
        └── orchestrator.py     # Async worker loop tying all passes together
```

## Setup & Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- A local LLM server (Ollama with `mistral-nemo`, or any OpenAI-compatible endpoint)

### Step 1: Start Infrastructure

```bash
cd predixion
docker-compose up -d
```

Wait for Redis and Qdrant to become healthy:
```bash
docker-compose ps
```

### Step 2: Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Start the Local LLM

If using Ollama:
```bash
ollama pull mistral-nemo
ollama serve   # Runs on http://localhost:11434
```

### Step 4: Configure Environment

Edit `.env` if your setup differs from defaults:
```
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL_NAME=mistral-nemo
```

### Step 5: Seed Ground Truth into Qdrant

```bash
python -m scripts.seed_qdrant sample_conversations.json
```

This embeds all 10 sample conversations' ground truth labels into the vector DB.

### Step 6: Run the System

**Option A — Batch mode (recommended for testing):**
```bash
python main.py --batch sample_conversations.json
```
Processes all 10 conversations sequentially, prints results, saves to `sample_outputs/batch_results.json`.

**Option B — Queue mode (production-like):**
```bash
# Terminal 1: Load conversations into Redis queue
python main.py --enqueue sample_conversations.json

# Terminal 2: Start the worker
python main.py --worker

# Terminal 3 (after processing): View results
python main.py --results
```

## Sample Output

Each conversation produces an output like:

```json
{
  "conversation_id": "coll_002",
  "overall_score": 1,
  "dimensions": {
    "compliance_ethics": {
      "score": 1,
      "explanation": "Severe violations: agent made threats ('ghar pe log aayenge'), dismissed genuine hardship as 'bahana', showed no empathy for job loss situation."
    },
    "goal_achievement": {
      "score": 1,
      "explanation": "No payment commitment secured. Customer hung up due to agent's hostile behavior."
    },
    "communication_quality": {
      "score": 2,
      "explanation": "Agent communicated the basic debt information but failed to explain options or listen to customer concerns."
    },
    "empathy_tone": {
      "score": 1,
      "explanation": "Completely dismissed customer's hardship. Hostile and threatening tone throughout."
    },
    "objection_handling": {
      "score": 1,
      "explanation": "Did not offer any restructuring, EMI plans, or alternatives when customer expressed inability to pay."
    }
  },
  "outcome": "refused",
  "summary": "Severe compliance violation call. Agent used threatening language and dismissed genuine hardship.",
  "compliance_flags": [
    "Threatening language - 'ghar pe log aayenge'",
    "Dismissed hardship as excuses - 'yeh sab bahana hai'",
    "No options offered for restructuring"
  ],
  "improvement_suggestions": [
    "Never threaten physical visits or legal action as a coercion tactic",
    "When customer expresses hardship, acknowledge it and offer restructuring options",
    "Follow RBI guidelines on fair treatment of borrowers in financial distress"
  ],
  "sentiment_shifts": [
    "Customer started cooperative but became hostile after agent's threats",
    "Customer threatened to file a complaint before hanging up"
  ]
}
```

## Extensibility

**Adding new evaluation dimensions:**
1. Add the dimension to `AuditResult.dimensions` in `src/core/schema.py`
2. Add scoring criteria to the `AUDIT_SYSTEM_PROMPT` in `src/engine/auditor.py`
3. The system will automatically include it in outputs

**Swapping the LLM:**
Change `LLM_BASE_URL` and `LLM_MODEL_NAME` in `.env`. Any OpenAI-compatible API works (GPT-4, Claude via proxy, local vLLM, etc.).

**Adding new ground truth:**
Add conversations to `sample_conversations.json` and re-run `python -m scripts.seed_qdrant`.

## Design Decisions & Trade-offs

See `DESIGN_WRITEUP.md` for a detailed 1-page explanation of architectural choices and trade-offs.

## Evaluation Against Ground Truth

The system's scores can be compared against the `ground_truth` labels in `sample_conversations.json`. The 10 sample conversations cover a range of scenarios:
- Excellent calls (score 5): proper empathy, compliance, negotiation
- Poor calls (score 1-2): threats, harassment, minimal communication
- Medium calls (score 3-4): adequate but with specific gaps

The precedent retrieval mechanism helps maintain consistency by anchoring new evaluations to these human-graded baselines.
