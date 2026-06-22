# Compliance Q&A Bot — Setup & Run Guide

## Prerequisites

| Dependency | Min Version | Check |
|---|---|---|
| Python | 3.12.x (not 3.14) | `python3.12 --version` |
| Docker | 20+ | `docker --version` |
| Ollama | 0.3+ | `ollama --version` |
| llama.cpp / `llama-server` | recent | `llama-server --version` |
| uv (recommended) | any | `uv --version` |

> **Why Python 3.12?** LlamaIndex/Pydantic break on Python 3.14. Use `python3.12` explicitly or `uv python install 3.12`.

### macOS install

```bash
brew install ollama docker uv llama.cpp
uv python install 3.12
ollama serve   # background
```

---

## Step 1 — Configure

```bash
cd compliance-bot
cp .env.example .env
```

`.env` is the single source of truth. `config.py` only has fallback defaults.

### Key toggles

| Variable | Purpose |
|---|---|
| `LLM_BACKEND` | `ollama` (default) or `openai-compatible` (llama-server / vLLM via `/v1/chat/completions`) |
| `USE_REMOTE_OLLAMA` | `true` → use `OLLAMA_REMOTE_URL` (Spark) |
| `USE_REMOTE_QDRANT` | `true` → use `QDRANT_REMOTE_URL` (Spark) |
| `EMBEDDING_SOURCE` | `huggingface` or `ollama` |
| `EMBEDDING_MODEL` | model name; dim must match `QDRANT_VECTOR_DIM` |
| `RERANKER_ENABLED` | `true` to enable `/v1/rerank` reranking |
| `RERANKER_BACKEND` | `llama-server` (local) or `vllm` (remote) |

---

## Step 2 — Install Dependencies

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# OR plain Python
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:
```bash
python -c "from config import settings; print(settings.llm_model, settings.embedding_model)"
```

---

## Step 3 — Start Local Infrastructure

```bash
docker compose up -d   # Qdrant (6333) + Phoenix (6006)
```

Verify:
```bash
curl http://localhost:6333/healthz   # Qdrant
open http://localhost:6006           # Phoenix UI
```

---

## Step 4 — Pull Ollama Models

```bash
# LLM
ollama pull qwen2.5:32b-instruct-q8_0   # ~33 GB, 50-90s on M4 Pro
# OR smaller for dev
ollama pull qwen2.5:14b                 # ~9 GB

# Embedding (only if EMBEDDING_SOURCE=ollama)
ollama pull embeddinggemma              # 768 dim
ollama pull qwen3-embedding             # 4096 dim
```

`EMBEDDING_SOURCE=huggingface` (default) downloads automatically on first use to `~/.cache/huggingface/`. We use `nvidia/llama-nemotron-embed-1b-v2` (2048 dim) by default.

---

## Step 5 — Start Reranker (llama-server)

The reranker runs as a separate llama-server process. Default config in `.env`:

```
RERANKER_ENABLED=true
RERANKER_BACKEND=llama-server
RERANKER_URL=http://localhost:8081
RERANKER_MODEL=qwen3-reranker-4b-q8
```

Start the reranker:
```bash
llama-server -hf Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp:Q8_0 \
  --reranking --pooling rank --embedding --port 8081
```

> **Important**: only the `Voodisss/...-llama_cpp` GGUF includes the classifier head needed for proper rerank scores. Other community GGUFs produce garbage scores.

Verify:
```bash
curl -s http://localhost:8081/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"<Instruct>: x\n<Query>: software install policy",
       "documents":["Team Members are forbidden to install software.","Annual leave..."],
       "top_n":2}' | python3 -m json.tool
```
Expected: first doc ~0.97, second ~0.0002.

To disable reranker entirely: `RERANKER_ENABLED=false`.

---

## Step 6 — Add Policy Documents

```bash
mkdir -p policies
cp /path/to/your/policies/*.docx policies/
```

`.docx` only. Use Word's Heading 1/2/3 styles for hierarchy. Auto-numbered clauses are detected via `NumberingResolver` (handles cross-`numId` continuation that Word renders as a single sequence).

---

## Step 7 — Ingest

```bash
PYTHONPATH=. python scripts/ingest_all.py --folder ./policies
```

Expected: `Done. Ingested 52 documents, ~1602 total chunks.`

Verify:
```bash
curl -s http://localhost:6333/collections/compliance_policies | python3 -m json.tool | grep points_count
```

Re-ingestion is safe — old chunks for each doc are deleted before insert. After changing the **embedding model or dimension**, delete the collection first:

```bash
PYTHONPATH=. python -c "from qdrant_client import QdrantClient; QdrantClient('http://localhost:6333').delete_collection('compliance_policies')"
PYTHONPATH=. python scripts/ingest_all.py --folder ./policies
```

---

## Step 8 — Test the Pipeline

### Test a query

```bash
PYTHONPATH=. python scripts/test_query.py -q "What is the policy on software installation?"
```

Returns `ComplianceAnswer` JSON with `answer`, `citations[]` (with `source_number`, `doc_title`, `section`, `clause`, `clause_number`, `quote`), and `escalation`.

---

## Step 9 — View Traces

```
http://localhost:6006
```

- **Agentic**: full ReAct trace — every Thought/Action/Observation, tool calls, LLM prompts, latency

---

## Step 10 — Run Teams Bot

The bot polls Microsoft Graph and runs the RAG pipeline directly via Python imports — no HTTP layer between bot and pipeline.

### Required `.env` vars

```
TEAMS_TENANT_ID=...
TEAMS_CLIENT_ID=...
TEAMS_CLIENT_SECRET=...
TEAMS_REFRESH_TOKEN=...
```

The refresh token is obtained one-time via the device-code flow (use `scripts/get_refresh_token.py` from the original `legal-compliance-qa-agent` repo). The bot rotates and persists the token to `channels/teams/data/refresh_token.json`.

### Start

```bash
PYTHONPATH=. python scripts/start_teams_bot.py
```

Send a message in Teams to the bot user. The bot replies with the answer + a rating prompt (`-1`, `0`, `1`, `2`).

### Rating values

- **-1** — bot answered, but should have been escalated
- **0** — wrong
- **1** — partially correct
- **2** — correct

Detection rule: `message.strip()` must equal exactly `"-1"`, `"0"`, `"1"`, or `"2"`. Anything else (sentences, typos, Cyrillic) is treated as a new question.

### Feedback storage

Each rating saves to BOTH:
- `channels/teams/data/feedback.jsonl` (append-only, easy to load with pandas)
- `channels/teams/data/feedback.db` (SQLite with indexes on `rating`, `timestamp`)

Both gitignored.

### Inspecting feedback

#### Local (running bot via `python scripts/start_teams_bot.py`)

Files are directly in `channels/teams/data/`:

```bash
# Tail the JSONL
tail -f channels/teams/data/feedback.jsonl

# Pretty-print
cat channels/teams/data/feedback.jsonl | jq .

# Query SQLite — count by rating
sqlite3 channels/teams/data/feedback.db "SELECT rating, COUNT(*) FROM feedback GROUP BY rating;"

# Recent entries
sqlite3 -header -column channels/teams/data/feedback.db \
  "SELECT id, rating, user, substr(question, 1, 50) AS q, timestamp FROM feedback ORDER BY timestamp DESC LIMIT 10;"

# Bad ratings only (for review)
sqlite3 channels/teams/data/feedback.db \
  "SELECT timestamp, rating, question, answer FROM feedback WHERE rating <= 0;"
```

Or via Python:

```bash
python -c "
from channels.teams.feedback import load_feedback_db
import json
for r in load_feedback_db()[:10]:
    print(json.dumps({'rating': r['rating'], 'user': r['user'], 'q': r['question'][:60]}))
"
```

#### Docker (with bind mount)

`docker-compose-remote.yml` mounts `./channels/teams/data` as a **bind mount** — feedback files appear directly on the host in the project folder. Same commands above work, plus:

```bash
# Tail logs while file updates live in the IDE
tail -f channels/teams/data/feedback.jsonl

# Or exec into the container if needed
docker compose -f docker-compose-remote.yml exec bot \
  sqlite3 channels/teams/data/feedback.db \
  "SELECT rating, COUNT(*) FROM feedback GROUP BY rating;"
```

#### Migrating from named volume to bind mount

If you previously ran with a named volume (older compose), copy the data out before switching:

```bash
mkdir -p channels/teams/data
docker compose -f docker-compose-remote.yml cp bot:/app/channels/teams/data/. ./channels/teams/data/
docker compose -f docker-compose-remote.yml up -d --force-recreate bot

# Optional: remove the now-unused named volume
docker volume rm compliance-bot_teams_data
```

---

## Step 11 — Run Evaluation

### Upload datasets to Phoenix

```bash
python scripts/make_dataset.py eval/datasets/retrieval_test.json
python scripts/make_dataset.py eval/datasets/e2e_test.json
python scripts/make_dataset.py eval/datasets/chatbot_test_cases.json
```

### Run experiments

```bash
# Tier 1 — retrieval only (fast, no LLM)
python eval/run_experiment.py --tier tier1 --name baseline-retrieval

# Tier 2 — full agent e2e
python eval/run_experiment.py --tier tier2 --name agentic-baseline

# Chatbot — realistic user questions
python eval/run_experiment.py --tier chatbot --name chatbot-baseline
```

Auto-generated experiment names include backend + reranker config. Metadata captures infra (`local`/`remote`) and URLs.

---

## Remote Stack (Spark @ 192.168.100.2)

For the production deployment, models run on Spark:
- Ollama (LLM + embedding) on port 11434
- Qdrant on port 6333
- vLLM (reranker) on port 8082

Switch via `.env`:

```
USE_REMOTE_OLLAMA=true
USE_REMOTE_QDRANT=true
EMBEDDING_SOURCE=ollama
OLLAMA_EMBEDDING_URL=http://192.168.100.2:11434
RERANKER_BACKEND=vllm
RERANKER_URL=http://192.168.100.2:8082
RERANKER_MODEL=Qwen/Qwen3-Reranker-4B
```

Re-ingest if embedding dimensions change. Verify connectivity to Spark before starting bot:

```bash
curl http://192.168.100.2:6333/collections
curl http://192.168.100.2:11434/api/tags
curl http://192.168.100.2:8082/v1/models
```

---

## Docker Deployment (Remote Bot Host)

For hosting the Teams bot on a separate machine that connects to Spark:

```bash
# On the bot host (Linux):
git clone <repo> compliance-bot && cd compliance-bot

# Copy a .env configured for remote stack (TEAMS_* + USE_REMOTE_*=true)
scp local:/path/to/.env .env

# Build & run
docker compose -f docker-compose-remote.yml up -d --build
```

The compose file runs:
- **bot** container (Python 3.12-slim) — code only, polls Graph API outbound
- **phoenix** container — observability, healthchecked

Volumes:
- **`./channels/teams/data` → `/app/channels/teams/data`** (bind mount — feedback DB, refresh token, bot state appear directly in the project folder so you can read them in your IDE)
- `phoenix_data` (named) → `/data` (traces)

`.env` is mounted via `env_file:`. Phoenix endpoint is overridden to `http://phoenix:6006/v1/traces` inside the compose network.

Logs: `docker compose -f docker-compose-remote.yml logs -f bot`

> **Ingestion is a one-time admin task** done before deploying. The bot image does not include `policies/` or `ingest/`.

---

## Quick-Start (All Steps Combined)

```bash
# 1. Setup
cp .env.example .env
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Infrastructure
docker compose up -d
ollama pull qwen2.5:32b-instruct-q8_0

# 3. Reranker (separate terminal, optional)
llama-server -hf Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp:Q8_0 \
  --reranking --pooling rank --embedding --port 8081

# 4. Documents + ingest
cp /path/to/policies/*.docx policies/
PYTHONPATH=. python scripts/ingest_all.py --folder ./policies

# 5. Test
PYTHONPATH=. python scripts/test_query.py -q "What is the policy on annual leave?"

# 6. View traces
open http://localhost:6006

# 7. Run Teams bot (after filling in TEAMS_* in .env)
PYTHONPATH=. python scripts/start_teams_bot.py
```

---

## Troubleshooting

### Reranker scores compressed (0.5-0.9 range)

Likely vLLM with default `/v1/rerank` not applying the Qwen3 chat template. Set `RERANKER_BACKEND=vllm` — our reranker module wraps query/documents with `<|im_start|>` + `<Document>:` + think suffix. Score discrimination should jump to 0.99 vs 0.0003.

### `OpenAILike` agent emits ReAct JSON instead of calling tools

In `rag/agent.py:get_llm()`, ensure `is_function_calling_model=True` for `OpenAILike`. Do NOT use `response_format={"type":"json_object"}` — it conflicts with tool calling.

### Teams bot can't load refresh token

Bot prefers `channels/teams/data/refresh_token.json` (rotated copy) over `.env` (initial seed). On first run with no file, it falls back to `TEAMS_REFRESH_TOKEN` from `.env`. After the first refresh, the file takes over.

### Docker bot can't reach Phoenix

Ensure compose overrides `PHOENIX_ENDPOINT=http://phoenix:6006/v1/traces` (already in `docker-compose-remote.yml`). Inside the bot container, `localhost:6006` is the bot itself.

### Embedding dim mismatch on re-ingestion

Different models produce different vector sizes (nemotron 2048, gemma 768, qwen3-embedding 4096). Delete the collection first:

```bash
python -c "from qdrant_client import QdrantClient; QdrantClient('http://localhost:6333').delete_collection('compliance_policies')"
```

Update `QDRANT_VECTOR_DIM` in `.env` and re-ingest.

### `\n` in env vars sent literally

`.env` stores `\n` as two chars. The reranker (`_build_query`) and Ollama embedding (`_ollama_embed`) convert `\\n` → `\n` at runtime. If you write a new place that reads `EMBEDDING_QUERY_PREFIX` or `RERANKER_QUERY_TEMPLATE` with `\n`, do the same conversion.

### Eval logs show stale results / `'results'` KeyError

`eval/agent_wrapper.py:_logged_search_policies` uses `import rag.tools.search_policies as sp` then reads `sp._last_search_results`. Don't switch to `from ... import _last_search_results` — that captures a stale reference.

### Phoenix UI not loading / no traces

```bash
docker compose ps phoenix
docker compose logs phoenix
grep PHOENIX .env
```

### Qdrant collection missing

```bash
PYTHONPATH=. python scripts/ingest_all.py --folder ./policies
```

### Python 3.14 / Pydantic errors

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Configuration Reference

All settings live in `.env`. See `config.py` for full schema. Notable groups:

### LLM & Backend
`LLM_BACKEND`, `LLM_MODEL`, `OPENAI_MODEL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `LLM_TEMPERATURE`, `USE_REMOTE_OLLAMA`, `OLLAMA_BASE_URL`, `OLLAMA_REMOTE_URL`, `LLM_REQUEST_TIMEOUT`, `LLM_REMOTE_REQUEST_TIMEOUT`

### Embeddings
`EMBEDDING_SOURCE`, `EMBEDDING_MODEL`, `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_PASSAGE_PREFIX`, `OLLAMA_EMBEDDING_URL`, `HF_TOKEN`

### Vector Store
`USE_REMOTE_QDRANT`, `QDRANT_URL`, `QDRANT_REMOTE_URL`, `QDRANT_COLLECTION`, `QDRANT_VECTOR_DIM`

### Search & Reranker
`RETRIEVAL_TOP_K`, `MIN_CONFIDENCE_SCORE`, `BM25_ENABLED`, `RERANKER_ENABLED`, `RERANKER_BACKEND`, `RERANKER_URL`, `RERANKER_MODEL`, `RERANKER_TOP_N`, `RERANKER_CANDIDATES`, `RERANKER_INSTRUCTION`, `RERANKER_QUERY_TEMPLATE`

### Agent
`AGENT_MAX_ITERATIONS`, `AGENT_TIMEOUT`

### Teams Bot
`TEAMS_TENANT_ID`, `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET`, `TEAMS_REFRESH_TOKEN`, `TEAMS_POLL_INTERVAL`

### Observability
`PHOENIX_ENABLED`, `PHOENIX_ENDPOINT`, `PHOENIX_PROJECT_NAME`

---

## Testing

Tier-A unit tests (pure logic, no services) plus an auto-skipping corpus layer.

```bash
pip install -r requirements-dev.txt

# Unit tests only — fast, no policy docs needed (CI-safe)
PYTHONPATH=. pytest -m "not corpus" -q

# Everything, including corpus parsing/numbering (needs local policies/*.docx)
PYTHONPATH=. pytest -q

# Parsing-coverage report over the local corpus
PYTHONPATH=. python scripts/parse_coverage.py
```

The `corpus` tests require the gitignored `policies/*.docx` and **auto-skip**
when they're absent (e.g. on CI or a fresh checkout).

### Manual: transient-infra resilience

Confirm a down backend yields a clean "unavailable" notice (not an escalation,
no raw error). Requires the local env; points embeddings at a dead port:

```bash
PYTHONPATH=. python -c "
from config import settings
settings.phoenix_enabled=False; settings.bm25_enabled=False
settings.ollama_embedding_url='http://127.0.0.1:1'
import channels.teams.bot as bot
print(bot._run_rag('remote access policy'))   # -> {'status': 'unavailable'}
"
```

And the LLM-backend-down path (leave embeddings/Qdrant up, kill only the LLM):

```bash
PYTHONPATH=. python -c "
from config import settings
settings.phoenix_enabled=False; settings.use_remote_ollama=False
settings.ollama_base_url='http://127.0.0.1:1'
import channels.teams.bot as bot
print(bot._run_rag('remote access policy'))   # -> {'status': 'unavailable'}
"
```

With Phoenix running, the event appears as an `infra_unavailable` span
(attributes: `failed_component`, `error_type`, `retries_attempted`).

---

## Project Status

### Completed
- DOCX ingestion with structure-aware chunking + cross-numId numbering fix
- Hybrid search + reranker (`/v1/rerank`, supports llama-server and vLLM with model-specific templates)
- Dual LLM backend (Ollama + OpenAI-compatible)
- Agentic RAG (LlamaIndex AgentWorkflow + structured `ComplianceAnswer` JSON with `source_number`)
- Microsoft Teams bot with feedback loop (-1, 0, 1, 2 ratings → JSONL + SQLite)
- Phoenix observability with infra metadata in experiments
- Eval system with `match_mode='any'` for multi-citation
- Remote stack support (Spark)
- Docker images & remote-host compose

### Not Yet Implemented
- Email escalation notifications
- React frontend
- Tier 3 escalation evaluators
- pytest test suite
