# Compliance Q&A Bot — Setup & Run Guide

## Prerequisites

| Dependency | Minimum Version | Check Command |
|------------|----------------|---------------|
| Python | 3.12.x (not 3.14) | `python3.12 --version` |
| Docker | 20+ | `docker --version` |
| Ollama | 0.3+ | `ollama --version` |
| uv (recommended) | any | `uv --version` |

> **Why Python 3.12?** LlamaIndex and Pydantic have compatibility issues with Python 3.14. If you have 3.14 installed as default, use `python3.12` explicitly or install it via `uv python install 3.12`.

### Installing prerequisites (macOS)

```bash
# Homebrew
brew install ollama docker

# Python 3.12 via uv
brew install uv
uv python install 3.12

# Start Ollama service (runs in background)
ollama serve
```

---

## Step 1 — Clone and Configure

```bash
cd compliance-bot

# Create .env from template (edit values as needed)
cp .env.example .env
```

### Key `.env` settings to review

`.env` is the **single source of truth** for all runtime configuration. `config.py` only has fallback defaults — `.env` always wins.

| Variable | Default | When to change |
|----------|---------|----------------|
| `LLM_MODEL` | `qwen3:14b` | Use a smaller model for dev/testing on limited hardware |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Any HuggingFace model (see Embedding Models below) |
| `POLICY_DOCS_FOLDER` | `./policies` | Change if your DOCX files are elsewhere |
| `POLICY_BASE_URL` | `http://intranet.company.com/policies` | Set to wherever documents are hosted internally |
| `RETRIEVAL_TOP_K` | `10` | Number of chunks returned per search |
| `BM25_ENABLED` | `true` | Set `false` for vector-only search |
| `MIN_CONFIDENCE_SCORE` | `0.45` | Only used in vector-only mode (not hybrid) |

---

## Step 2 — Create Virtual Environment and Install Dependencies

```bash
# Using uv (recommended — fast)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using plain Python
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "from config import settings; print('Config OK:', settings.llm_model, settings.embedding_model)"
```

Expected output:
```
Config OK: qwen3:14b nomic-ai/nomic-embed-text-v1.5
```

---

## Step 3 — Start Docker Services (Qdrant + Phoenix)

```bash
docker compose up -d
```

This starts two services:
- **Qdrant** (port 6333) — vector database for policy chunks
- **Phoenix** (port 6006) — observability UI for tracing agent queries + evaluation experiments

### Verify services are healthy

```bash
# Qdrant
curl http://localhost:6333/healthz

# Phoenix — open in browser
open http://localhost:6006
```

> Both services persist data in Docker volumes. To reset: `docker compose down -v && docker compose up -d`

---

## Step 4 — Pull Ollama LLM Model

Ollama is only used for the LLM (reasoning). Embeddings use HuggingFace (downloaded automatically on first use).

```bash
# LLM — pick ONE based on your hardware:

# Option A: Recommended (9 GB, good tool-calling + structured JSON output)
ollama pull qwen3:14b

# Option B: Smaller alternative
ollama pull qwen2.5:14b

# Option C: Best quality (40 GB, requires 48+ GB RAM)
ollama pull llama3.3:70b
```

### Verify model is available

```bash
ollama list
```

> **Important:** Make sure `LLM_MODEL` in `.env` matches the model you pulled.

### Hardware requirements by model

| Model | RAM Required | Disk | Response Time (M4 Pro) |
|-------|-------------|------|----------------------|
| `qwen3:14b` | 16 GB | 9 GB | ~10–25s |
| `qwen2.5:14b` | 16 GB | 9 GB | ~10–25s |
| `llama3.3:70b` | 48 GB | 40 GB | ~30–60s |

---

## Step 5 — Add Policy Documents

Place your DOCX policy files into the `policies/` folder:

```bash
# Create the folder if it doesn't exist
mkdir -p policies

# Copy your documents
cp /path/to/your/policies/*.docx policies/
```

### Supported document format

- **File type:** `.docx` only (not `.doc`, `.pdf`, or `.txt`)
- **Headings:** Use Word's built-in Heading 1, Heading 2, Heading 3 styles for section hierarchy
- **Numbered clauses:** Clauses like `4.2.1 Data Retention` are auto-detected via `NumberingResolver`
- **Tables:** Automatically converted to text

---

## Step 6 — Ingest Documents

This parses all DOCX files, generates embeddings, and stores them in Qdrant (+ BM25 index if enabled).

```bash
python scripts/ingest_all.py
```

### Expected output

```
Ingesting documents from: policies

Ingested Acceptable Use Policy [Internal].docx: 51 chunks
Ingested Access Management Policy [Internal].docx: 36 chunks
...

Done. Ingested 52 documents, 1514 total chunks.
```

### Verify chunks are in Qdrant

```bash
curl -s http://localhost:6333/collections/compliance_policies | python3 -m json.tool | grep points_count
```

### Re-ingesting documents

Running `ingest_all.py` again is safe — it deletes old chunks for each document before inserting new ones (both Qdrant and BM25). Use this after updating policy files or changing the embedding model.

---

## Step 7 — Test the Agent

### Single query

```bash
python scripts/test_query.py -q "What is forbidden to install for team members?"
```

### Interactive mode

```bash
python scripts/test_query.py -i
```

### What to expect

The agent returns **structured JSON** with `answer`, `citations`, and `escalation`:

```json
{
  "answer": "According to the Acceptable Use Policy...",
  "citations": [
    {
      "doc_title": "Acceptable Use Policy [Internal]",
      "section": "Corporate Workstation and Software Use",
      "clause": "Software Installation",
      "clause_number": "4.7",
      "quote": "Team Members are forbidden from installing..."
    }
  ],
  "escalation": {"needed": false, "reason": ""}
}
```

| Query Type | Expected Behavior |
|-----------|-------------------|
| Clear policy question | Answer with structured JSON citations |
| No matching policy | `NO_RELEVANT_POLICY_FOUND` → agent escalates |
| Multi-area question | Agent calls `search_policies` multiple times |

---

## Step 8 — View Traces in Phoenix

Every query is automatically traced via OpenTelemetry. Open the Phoenix UI:

```
http://localhost:6006
```

### What you'll see

- **Agent workflow** — the parent span covering the entire request
- **ReAct steps** — each Thought/Action/Observation iteration
- **LLM calls** — model name, full prompt, response text, token counts, latency
- **Tool calls** — which tools were invoked, inputs, and outputs

---

## Step 9 — Run Evaluation

### Upload test datasets to Phoenix

```bash
python scripts/make_dataset.py eval/datasets/retrieval_test.json
python scripts/make_dataset.py eval/datasets/e2e_test.json
python scripts/make_dataset.py eval/datasets/chatbot_test_cases.json
```

### Run experiments

```bash
# Tier 1 — retrieval only (fast, no LLM)
python eval/run_experiment.py --tier tier1 --name baseline-retrieval

# Tier 2 — full agent e2e (slow, calls LLM per question)
python eval/run_experiment.py --tier tier2 --name baseline-e2e

# Chatbot — realistic user questions
python eval/run_experiment.py --tier chatbot --name baseline-chatbot
```

Settings come from `.env` (`RETRIEVAL_TOP_K`, `BM25_ENABLED`). Override with `--top-k` flag if needed.

### View results

Open http://localhost:6006/datasets — experiments appear with per-evaluator metrics.

### After making changes

```bash
# Re-run with new experiment name for comparison
python eval/run_experiment.py --tier tier1 --name after-reranker-v1
```

---

## Embedding Models

Embeddings use **HuggingFace** (via `sentence-transformers`). Models are downloaded automatically on first use.

Change by editing `EMBEDDING_MODEL` in `.env`:

```bash
# Default (768d)
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Alternatives (768d — no Qdrant changes needed)
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Larger models (1024d — update QDRANT_VECTOR_DIM=1024)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_MODEL=intfloat/e5-large-v2
```

**After changing embedding model:** delete the Qdrant collection and re-ingest:

```bash
python -c "from qdrant_client import QdrantClient; QdrantClient('http://localhost:6333').delete_collection('compliance_policies')"
python scripts/ingest_all.py
```

If the new model has different dimensions, also update `QDRANT_VECTOR_DIM` in `.env`.

---

## Quick-Start (All Steps Combined)

```bash
# 1. Environment
cp .env.example .env
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Infrastructure
docker compose up -d
ollama pull qwen3:14b

# 3. Update .env to match your model
# Edit LLM_MODEL= to match what you pulled

# 4. Add documents and ingest
cp /path/to/policies/*.docx policies/
python scripts/ingest_all.py

# 5. Test
python scripts/test_query.py -q "What is the policy on annual leave?"

# 6. View traces
open http://localhost:6006

# 7. Run evaluation
python scripts/make_dataset.py eval/datasets/retrieval_test.json
python eval/run_experiment.py --tier tier1 --name baseline
```

---

## Troubleshooting

### Qdrant won't start

```bash
# Check if port 6333 is already in use
lsof -i :6333

# Check Docker logs
docker compose logs qdrant

# Reset Qdrant data
docker compose down -v && docker compose up -d
```

### Ollama connection refused

```bash
# Make sure Ollama is running
ollama serve

# Check it's responding
curl http://localhost:11434/api/tags

# If using a remote Ollama, update .env:
# OLLAMA_BASE_URL=http://your-server:11434
```

### "Model not found" error

```bash
# List available models
ollama list

# Pull the missing model
ollama pull qwen3:14b

# Make sure .env matches
grep LLM_MODEL .env
```

### Agent doesn't call tools (hallucinated answers)

This happens with smaller models — they skip tool calls and generate answers from general knowledge. Fix:
- Use `qwen3:14b` or larger
- Check `json_parse_success` evaluator in Phoenix experiments

### "Collection not found" error

Run ingestion first — it creates the Qdrant collection automatically:

```bash
python scripts/ingest_all.py
```

### Python 3.14 / Pydantic errors

```
TypeError: ForwardRef._evaluate() ...
```

You're running Python 3.14. Switch to 3.12:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Phoenix UI not loading / traces not appearing

```bash
# Check Phoenix container is running
docker compose ps phoenix

# Check Phoenix logs
docker compose logs phoenix

# If Phoenix is running but traces don't appear, check .env:
grep PHOENIX .env
# PHOENIX_ENABLED=true
# PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
```

### Search returns no results but documents are ingested

With hybrid search (`BM25_ENABLED=true`), there's no confidence threshold — results are always returned if any match exists.

With vector-only search (`BM25_ENABLED=false`), check the confidence threshold:

```bash
# In .env, try lowering:
MIN_CONFIDENCE_SCORE=0.3
```

### Embedding model download slow / fails

HuggingFace models are cached in `~/.cache/huggingface/`. First download may take a few minutes. If behind a firewall, set `HF_HUB_OFFLINE=1` after initial download.

---

## Configuration Reference

All settings are in `.env` and loaded via `config.py`. No code changes needed.

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `10` | Number of chunks returned per search |
| `BM25_ENABLED` | `true` | Toggle hybrid (vector+BM25) vs vector-only search |
| `HYBRID_VECTOR_CANDIDATES` | `20` | Vector candidates before RRF fusion |
| `HYBRID_BM25_CANDIDATES` | `20` | BM25 candidates before RRF fusion |
| `MIN_CONFIDENCE_SCORE` | `0.45` | Cosine threshold — only used in vector-only mode |

### Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_MIN_TOKENS` | `50` | Chunks smaller than this are merged with neighbors |
| `CHUNK_MAX_TOKENS` | `400` | Chunks larger than this are split at sentence boundaries |

### Agent

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_ITERATIONS` | `8` | Max ReAct loops before forced stop |
| `AGENT_TIMEOUT` | `120` | Seconds before agent times out |
| `LLM_TEMPERATURE` | `0.0` | **Must be 0.0** for deterministic compliance answers |

### Observability (Phoenix)

| Variable | Default | Description |
|----------|---------|-------------|
| `PHOENIX_ENABLED` | `true` | Set `false` to disable tracing entirely |
| `PHOENIX_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP collector endpoint |
| `PHOENIX_PROJECT_NAME` | `compliance-bot` | Project name in Phoenix UI |

---

## Project Status

The core RAG pipeline + evaluation system are complete. The system can:

- Parse DOCX files with structure-aware chunking (headings + ilvl numbering)
- Generate embeddings via HuggingFace models and store in Qdrant
- Search policies via hybrid (vector + BM25 with RRF) or vector-only
- Return structured JSON answers with citations (ComplianceAnswer schema)
- Escalate unanswerable questions
- Run a ReAct agent with 3 tools (search, get_section, escalate)
- Trace every query end-to-end via Phoenix
- Evaluate with Phoenix Datasets + Experiments (Tier 1, 2, Chatbot)

**Not yet implemented:**

- SQLite database for escalations and chat history
- FastAPI REST API endpoints
- Email notifications for escalations
- React frontend chat UI
- Automated tests
- Tier 3 escalation evaluators
