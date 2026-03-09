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

| Variable | Default | When to change |
|----------|---------|----------------|
| `LLM_MODEL` | `qwen2.5:14b` | Use `qwen2.5:1.5b` for fast dev/testing on limited hardware |
| `POLICY_DOCS_FOLDER` | `./policies` | Change if your DOCX files are elsewhere |
| `POLICY_BASE_URL` | `http://intranet.company.com/policies` | Set to wherever documents are hosted internally |
| `MIN_CONFIDENCE_SCORE` | `0.45` | Lower = more lenient answers, higher = more escalations |
| `RETRIEVAL_TOP_K` | `6` | Number of chunks returned per search |

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
python -c "from config import settings; print('Config OK:', settings.llm_model)"
```

Expected output:
```
Config OK: qwen2.5:1.5b
```

---

## Step 3 — Start Docker Services (Qdrant + Phoenix)

```bash
docker compose up -d
```

This starts two services:
- **Qdrant** (port 6333) — vector database for policy chunks
- **Phoenix** (port 6006) — observability UI for tracing agent queries

### Verify services are healthy

```bash
# Qdrant
curl http://localhost:6333/healthz

# Phoenix — open in browser
open http://localhost:6006
```

> Both services persist data in Docker volumes. To reset: `docker compose down -v && docker compose up -d`

---

## Step 4 — Pull Ollama Models

You need two models: an LLM for reasoning and an embedding model for vector search.

```bash
# Embedding model (required, ~274 MB)
ollama pull nomic-embed-text

# LLM — pick ONE based on your hardware:

# Option A: Small model for development (1 GB, fast, lower quality tool-calling)
ollama pull qwen2.5:1.5b

# Option B: Recommended for production (9 GB, good tool-calling)
ollama pull qwen2.5:14b

# Option C: Best quality (40 GB, requires 48+ GB RAM)
ollama pull llama3.3:70b
```

### Verify models are available

```bash
ollama list
```

You should see both your chosen LLM and `nomic-embed-text` listed.

> **Important:** Make sure `LLM_MODEL` in `.env` matches the model you pulled.

### Hardware requirements by model

| Model | RAM Required | Disk | Response Time (M4 Pro) |
|-------|-------------|------|----------------------|
| `qwen2.5:1.5b` | 4 GB | 1 GB | ~2–5s |
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
- **Numbered clauses:** Clauses like `4.2.1 Data Retention` are auto-detected
- **Tables:** Automatically converted to text

### Sample test document

If you don't have documents yet, create a test one:

```bash
python -c "
from docx import Document
doc = Document()
doc.add_heading('Test Policy', level=1)
doc.add_heading('1. Scope', level=2)
doc.add_paragraph('1.1 This policy applies to all employees of the company across all departments and locations.')
doc.add_heading('2. Rules', level=2)
doc.add_paragraph('2.1 Employees must complete mandatory training within 30 days of joining the company.')
doc.add_paragraph('2.2 All incidents must be reported to the compliance team within 24 hours of discovery.')
doc.save('policies/test_policy.docx')
print('Created policies/test_policy.docx')
"
```

---

## Step 6 — Ingest Documents

This parses all DOCX files, generates embeddings, and stores them in Qdrant.

```bash
python scripts/ingest_all.py
```

With custom options:

```bash
python scripts/ingest_all.py \
  --folder ./policies \
  --base-url http://intranet.company.com/policies
```

### Expected output

```
Ingesting documents from: policies
Base URL: http://intranet.company.com/policies

Ingested annual_leave_policy.docx: 6 chunks
Ingested data_privacy_policy.docx: 12 chunks
Ingested remote_work_policy.docx: 8 chunks

Done. Ingested 3 documents, 26 total chunks.
```

### Verify chunks are in Qdrant

```bash
curl -s http://localhost:6333/collections/compliance_policies | python3 -m json.tool | grep points_count
```

Expected: `"points_count": 26` (or however many chunks were created).

### Re-ingesting documents

Running `ingest_all.py` again is safe — it deletes old chunks for each document before inserting new ones. Use this after updating policy files.

---

## Step 7 — Test the Agent

### Single query

```bash
python scripts/test_query.py -q "How many days of annual leave do employees get?"
```

### Interactive mode

```bash
python scripts/test_query.py -i
```

```
Compliance Q&A Bot — Interactive Mode
Type 'quit' or 'exit' to stop.

You: How many days of annual leave do employees get?
Searching policies...

Bot: **Answer:** Full-time employees are entitled to 25 days...

**Policy Sources:**
- Annual Leave Policy | 2. Entitlement | Clause 2.1
  > "Full-time employees are entitled to 25 days of paid annual leave..."
  Link: http://intranet.company.com/policies/annual_leave_policy.docx

You: quit
Goodbye!
```

### What to expect

| Query Type | Expected Behavior |
|-----------|-------------------|
| Clear policy question | Answer with citations from policy documents |
| No matching policy | `"NO_RELEVANT_POLICY_FOUND"` → agent escalates automatically |
| Ambiguous question | Agent may ask for clarification first |
| Multi-area question | Agent calls `search_policies` multiple times |

> **Note on `qwen2.5:1.5b`:** This small model often skips tool calls and hallucmates answers. If you see answers without proper citations, switch to `qwen2.5:14b` in `.env`. The 14b model handles the ReAct tool-calling pattern correctly.

---

## Step 8 — View Traces in Phoenix

Every query is automatically traced via OpenTelemetry. Open the Phoenix UI to inspect them:

```
http://localhost:6006
```

### What you'll see

Phoenix captures the full execution trace for every agent query:

- **Agent workflow** — the parent span covering the entire request
- **ReAct steps** — each Thought/Action/Observation iteration
- **LLM calls** — model name, full prompt, response text, token counts, latency
- **Tool calls** — which tools were invoked, inputs, and outputs
- **Embedding calls** — queries embedded during ingestion and search

### Navigating the UI

1. Select the **compliance-bot** project in the sidebar
2. Click any trace to see the full span tree
3. Expand individual spans to see attributes (prompt text, token counts, etc.)
4. Use the **Traces** tab to filter by time, latency, or status

### Disabling tracing

If you don't need observability (e.g., CI environments):

```bash
# In .env
PHOENIX_ENABLED=false
```

The application runs normally without Phoenix — tracing is fully optional.

---

## Quick-Start (All Steps Combined)

For a fresh machine, run everything in sequence:

```bash
# 1. Environment
cp .env.example .env
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Infrastructure
docker compose up -d
ollama pull nomic-embed-text
ollama pull qwen2.5:14b      # or qwen2.5:1.5b for dev

# 3. Update .env to match your model
# Edit LLM_MODEL= to match what you pulled

# 4. Add documents and ingest
cp /path/to/policies/*.docx policies/
python scripts/ingest_all.py

# 5. Test
python scripts/test_query.py -q "What is the policy on annual leave?"

# 6. View traces
open http://localhost:6006
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
ollama pull qwen2.5:14b

# Make sure .env matches
grep LLM_MODEL .env
```

### Embedding takes too long

First ingestion may be slow as Ollama loads the embedding model. Subsequent runs are faster. For 50 documents, expect 2–5 minutes total.

### Agent doesn't call tools (hallucmates answers)

This happens with `qwen2.5:1.5b` — it's too small for reliable ReAct reasoning. Fix:

```bash
# In .env, change:
LLM_MODEL=qwen2.5:14b

# Pull the model if needed
ollama pull qwen2.5:14b
```

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

# Verify the OTLP endpoint is reachable
curl -s -X POST http://localhost:6006/v1/traces
# Should return "Unsupported content type: None" (not 404 or connection refused)

# If Phoenix is running but traces don't appear, check .env:
grep PHOENIX .env
# PHOENIX_ENABLED=true
# PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
```

### Search returns no results but documents are ingested

Check the confidence threshold. If your queries are very different from policy language, the cosine score may fall below 0.45:

```bash
# In .env, try lowering the threshold:
MIN_CONFIDENCE_SCORE=0.3
```

Or verify what's actually in Qdrant:

```bash
curl -s -X POST http://localhost:6333/collections/compliance_policies/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 3, "with_payload": true}' | python3 -m json.tool
```

---

## Configuration Reference

All settings are in `.env` and loaded via `config.py`. No code changes needed.

### Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_MIN_TOKENS` | `50` | Chunks smaller than this are merged with neighbors |
| `CHUNK_MAX_TOKENS` | `400` | Chunks larger than this are split at sentence boundaries |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `6` | Number of chunks returned per search |
| `MIN_CONFIDENCE_SCORE` | `0.45` | Below this cosine score → `NO_RELEVANT_POLICY_FOUND` |

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

Steps 0–8 (core RAG pipeline) are complete. The system can:

- Parse DOCX files with structure-aware chunking
- Generate embeddings and store in Qdrant
- Search policies via semantic similarity
- Fetch full clause text by exact identifier
- Escalate unanswerable questions with ticket IDs
- Run an agent that orchestrates all tools via ReAct loop
- Trace every query end-to-end via Phoenix (LLM calls, tool invocations, latency)

**Not yet implemented** (Steps 9–13):

- SQLite database for escalations and chat history (Step 9)
- FastAPI REST API endpoints (Step 10)
- Email notifications for escalations (Step 11)
- React frontend chat UI (Step 12)
- Automated tests (Step 13)
