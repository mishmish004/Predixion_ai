# Design Write-up: Predixion Debt Audit System

## Architecture Choice: Multi-Pass Reasoning (MPR)

Instead of sending the entire transcript to an LLM in a single prompt and asking for scores, I split the evaluation into three passes. This was driven by a practical observation: LLMs produce more reliable structured outputs when each prompt has a focused, narrow task.

**Pass 1 (State Extraction)** converts the raw Hinglish transcript into a structured State Map. This is important because Hinglish is code-mixed — a single sentence might switch between Hindi and English mid-phrase. By having the LLM first "translate" the conversation into structured intents and events, we reduce ambiguity for the scoring step. The State Map captures intent flow (what each speaker was trying to do), critical events (threats, empathy moments, compliance violations), and sentiment trajectory.

**Pass 2 (Precedent Retrieval)** uses the State Map to find the most similar previously-graded call in a Qdrant vector database. The idea is calibration: if a new call looks similar to a past call that was scored 1/5 for compliance violations, the system has a reference point. This addresses a well-known LLM problem — inconsistent scoring across runs. The embedding is done locally via FastEmbed (BAAI/bge-small-en-v1.5), so no external API calls are needed for vectorization.

**Pass 3 (Delta Audit)** gives the LLM the transcript, the State Map, and the precedent, asking it to produce final scores. Having the precedent as an anchor helps the LLM calibrate: "this call is similar to one scored 5/5 but lacks empathy, so empathy should be lower."

## Key Trade-offs

**LLM dependency for Hinglish understanding.** I rely on the LLM (mistral-nemo via Ollama) to understand Hinglish rather than building custom NLP pipelines. This is practical for a 3-day assessment — building a Hinglish intent classifier from scratch would take weeks. The trade-off is that quality depends heavily on the LLM's Hinglish capability. The prompts include explicit Hinglish glossary hints (e.g., "dhamki" = threat) to help.

**Structured JSON output from LLMs.** LLMs sometimes produce malformed JSON. I handle this with cleanup (stripping markdown fences) and Pydantic validation. A more robust approach would use function calling or constrained generation, but those aren't universally available across all LLM providers.

**Redis as a queue.** For a production system, a proper message broker (RabbitMQ, Kafka) would be better. Redis LPUSH/LPOP is simple and sufficient for this scale, and it doubles as the results store.

**Embedding model choice.** bge-small-en-v1.5 is English-focused, which is suboptimal for Hinglish. A multilingual model (e.g., multilingual-e5) would be better for production. I chose bge-small because it's lightweight, fast to download, and "good enough" when the State Map text is already in English (Pass 1 extracts English summaries from Hinglish).

**Self-evaluation concern.** When processing sample conversations, the system may retrieve the same conversation as its own precedent from Qdrant. In production, you'd filter out the current conversation ID from search results. I've kept it simple here since the ground truth seeding and evaluation are separate concerns in the assessment.

## What I Would Improve With More Time

1. **Confidence scores** — have the LLM output confidence per dimension, flag low-confidence scores for human review.
2. **Few-shot examples in prompts** — include 2-3 graded examples directly in the audit prompt for better calibration.
3. **Multilingual embeddings** — switch to a model that handles Hindi/Hinglish natively.
4. **Unit tests** — mock the LLM responses and test the parsing/validation pipeline.
5. **Streaming results** — use Redis pub/sub to stream results as they complete.
6. **Dashboard** — a simple web UI to visualize scores across calls and flag outliers.

## AI Tools Used

I used Claude (Anthropic) as a coding assistant to help structure the project, draft prompts, and review code. All architectural decisions, prompt engineering, and system design are my own work.
