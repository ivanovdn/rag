# Compliance Q&A Bot — Claude Code Instructions

## Project Overview

Internal Compliance Q&A Bot using **Agentic RAG** (default) or **Vanilla RAG** (single-call) architecture. Answers employee questions **strictly from approved internal policy documents (DOCX, 52 files)**. If an answer cannot be grounded in policy, it escalates to the Compliance team with full context.

**Channels**: Microsoft Teams (1:1 chat polling via Graph API). FastAPI HTTP endpoint also available.

**Inference**: Local (Ollama / llama-server) or remote (Spark @ `192.168.100.2`) — switchable via env vars. Embedding via HuggingFace or Ollama. Reranker via llama-server or vLLM.

**Current state**: Ingestion, agentic + vanilla pipelines, hybrid search + reranker, observability (Phoenix), Phoenix-based evaluation, FastAPI endpoint, Teams bot with feedback loop (JSONL + SQLite), and Docker deployment for remote hosts are all implemented.

---

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Ollama or llama-server / vLLM (OpenAI-compat) | Switchable via `LLM_BACKEND` (`ollama` or `openai-compatible`) |
| LLM model | `qwen2.5:32b-instruct-q8_0` typical | `LLM_MODEL` for Ollama, `OPENAI_MODEL` for openai-compat |
| Embedding | HuggingFace (`nvidia/llama-nemotron-embed-1b-v2`, 2048 dim) or Ollama (`embeddinggemma` 768, `qwen3-embedding` 4096) | `EMBEDDING_SOURCE=huggingface\|ollama` |
| Vector Store | Qdrant (Docker, port 6333) | Local or remote — `USE_REMOTE_QDRANT` toggle |
| Keyword Search | BM25 (pure Python, JSON persistence) | Toggleable via `BM25_ENABLED`, off by default |
| Search Fusion | Reciprocal Rank Fusion (RRF, k=60) | Combines vector + BM25 when both enabled |
| Reranker | Qwen3-Reranker-4B via llama-server `/v1/rerank` (local) or vLLM (remote) | `RERANKER_BACKEND=llama-server\|vllm`, model-agnostic API |
| RAG Framework | LlamaIndex `AgentWorkflow` | Agentic mode only |
| Vanilla pipeline | Pure Python — search → rerank → single LLM call | `rag/pipeline.py`, no LlamaIndex |
| Document Parsing | `python-docx` + `NumberingResolver` | Resolves Word auto-numbering across multiple `numId`s |
| Observability | Arize Phoenix (Docker, port 6006) | Auto-instruments LlamaIndex; manual spans for vanilla pipeline |
| Evaluation | Phoenix Datasets + Experiments | Tiers: tier1, tier2, chatbot (with multi-citation `match_mode`) |
| API | FastAPI on port 8000 | `POST /api/query`, `GET /health` |
| Teams Channel | Polls Microsoft Graph `/me/chats` every 5s | Direct RAG import (no HTTP between bot and pipeline) |
| Feedback storage | JSONL + SQLite | `channels/teams/data/feedback.{jsonl,db}` |

---

## Project Structure

```
compliance-bot/
├── CLAUDE.md
├── .env / .env.example
├── config.py                          # pydantic-settings, all toggles & URLs
├── requirements.txt                   # curated, minimal direct deps
├── docker-compose.yml                 # local: Qdrant + Phoenix
├── docker-compose-remote.yml          # remote: bot + Phoenix containers
├── Dockerfile                         # builds bot image (runtime-only code)
├── .dockerignore
│
├── ingest/
│   ├── numbering.py                   # NumberingResolver — cross-numId continuation
│   ├── docx_parser.py                 # Structure-aware chunker (headings + ilvl)
│   ├── chunk_models.py                # PolicyChunk pydantic model
│   └── pipeline.py                    # parse → embed → upsert
│
├── rag/
│   ├── embeddings.py                  # HuggingFace OR Ollama embedding (configurable)
│   ├── vector_store.py                # Qdrant client (active_qdrant_url)
│   ├── bm25_index.py                  # Pure-Python BM25
│   ├── hybrid_search.py               # RRF fusion
│   ├── reranker.py                    # /v1/rerank client (llama-server or vLLM)
│   ├── observability.py               # Phoenix init + tracer
│   ├── agent.py                       # AgentWorkflow + system prompt + ComplianceAnswer schema + get_llm()
│   ├── pipeline.py                    # Vanilla RAG (single LLM call, traced with manual spans)
│   ├── response.py                    # parse_agent_response() — JSON parser for agent output
│   └── tools/
│       ├── __init__.py
│       ├── search_policies.py         # search + rerank + format with [Source N] header
│       ├── get_section.py             # Fetch full section by doc_id + section_name
│       ├── escalate.py
│       └── clarify.py                 # NOT used (kept on disk for reference)
│
├── api/                               # FastAPI
│   ├── main.py                        # app + CORS + /health (initializes Phoenix)
│   ├── models.py                      # QueryRequest, QueryResponse pydantic models
│   └── routes/query.py                # POST /api/query — switches agentic/vanilla via PIPELINE_MODE
│
├── channels/
│   └── teams/
│       ├── auth.py                    # TokenRefresher (file → .env fallback)
│       ├── bot.py                     # TeamsBot — polls Graph, runs RAG directly, feedback loop
│       ├── renderer.py                # HTML renderers (answer / escalation / error / rating prompts)
│       ├── feedback.py                # save_feedback() → JSONL + SQLite
│       ├── utils.py                   # safe_get_nested, strip_html
│       └── data/                      # gitignored: bot_state.json, refresh_token.json, feedback.{jsonl,db}
│
├── eval/
│   ├── evaluators.py                  # shared evaluators with match_mode='any'/'all'
│   ├── agent_wrapper.py               # instrumented agent (logs tool calls); re-exports parse_agent_response
│   ├── pipeline_wrapper.py            # vanilla pipeline eval task
│   ├── run_experiment.py              # CLI: --mode agentic|vanilla, --tier
│   └── datasets/                      # gitignored: retrieval_test, e2e_test, chatbot_test_cases, escalation_test
│
└── scripts/
    ├── ingest_all.py
    ├── test_query.py                  # CLI: agentic agent testing
    ├── test_pipeline.py               # CLI: vanilla pipeline testing
    ├── make_dataset.py                # JSON → Phoenix Dataset
    ├── convert_eval_xlsx.py           # XLSX → JSON
    ├── start_api.sh                   # start FastAPI
    └── start_teams_bot.py             # start Teams bot (entry point)
```

---

## Document Ingestion Pipeline

### DOCX Parser — `ingest/docx_parser.py`

**Structure-aware chunking** — never splits by token count.

Key components:
- `NumberingResolver` (`ingest/numbering.py`): Simulates Word's `numbering.xml` counters. **Continues the counter across multiple `numId`s at level 0** (Word renders sequential sections continuously even when they reference different numIds — fixed previously when `External Communication` resolved as `5.` instead of `6.`).
- `extract_heading_level()`: Detects Heading 1/2/3 styles
- `extract_clause_name()`: Extracts bold-run label from ilvl=1 paragraphs
- `ilvl=0 + decimal` → section
- `ilvl=1 + decimal` → clause
- `ilvl=2+ / bullet` → content under clause
- Tables converted to `"Header: Value | Header: Value"` rows
- Min chunk: 50 tokens, Max: 400 tokens

### Chunk Metadata — `ingest/chunk_models.py`

```python
class PolicyChunk(BaseModel):
    chunk_id: str             # uuid
    doc_id: str               # slugified filename
    doc_title: str
    doc_filename: str
    doc_link: str
    section: str = ""         # name only (no number)
    section_number: str = ""
    clause: str = ""          # name only
    clause_number: str = ""
    section_display: str = "" # "7. Private Information > 7.5. Blogging..."
    text: str
    char_count: int = 0
    chunk_index: int = 0
    last_updated: str = ""
```

**Important:** `section` and `clause` store **names only**. Numbers are in `section_number`/`clause_number`. Eval matches against `section`/`clause`, not `section_display`.

**CLI:**
```bash
PYTHONPATH=. python scripts/ingest_all.py --folder ./policies
```

Current stats: **52 documents, ~1602 chunks**.

---

## Search Architecture

### Pipeline

```
Query → embed_query (with EMBEDDING_QUERY_PREFIX if set)
     → vector_search (top RERANKER_CANDIDATES if reranker on, else RETRIEVAL_TOP_K)
     → [optional: BM25 fusion via RRF]
     → [optional: rerank via /v1/rerank → top RERANKER_TOP_N]
     → format_sources() with [Source N] headers
```

### Reranker — `rag/reranker.py`

Universal `/v1/rerank` client. Two backend modes:

- **`llama-server`**: simple template — `<Instruct>: {instruction}\n<Query>: {query}` (set via `RERANKER_QUERY_TEMPLATE`)
- **`vllm`**: full Qwen3 chat template with `<|im_start|>system`/`user` wrapping for query and `<Document>:`+think suffix wrapping per document

Falls back to original ranking on connection error / timeout — never blocks the pipeline. Always sends `model` field (vLLM requires it; llama-server ignores).

### Format for the agent — `rag/tools/search_policies.py`

Replaces old `--- Result N ---` format. New format per source:
```
=== RETRIEVED POLICY SOURCES ===

[Source 1] Acceptable Use Policy [Internal]
Section: Corporate Workstation and Software Use
Clause 4.7: Software Installation
---
4.7. Software Installation: Team Members are forbidden to install...
```

`format_sources()` is called inside `search_policies()`. Module-level `_last_search_results` exposes structured results (with rerank scores) for eval logging — read via `import rag.tools.search_policies as sp; sp._last_search_results` (NOT `from ... import _last_search_results` — that would freeze a stale reference).

---

## The 3 Agent Tools

| Tool | File | Purpose |
|------|------|---------|
| `search_policies` | `rag/tools/search_policies.py` | Hybrid/vector search + rerank + format. Always call FIRST. |
| `get_section` | `rag/tools/get_section.py` | Fetch full section text by doc_id + section_name. New format: `=== FULL SECTION ===` |
| `escalate_to_compliance` | `rag/tools/escalate.py` | Escalate when no policy found |

`ask_clarification` (`rag/tools/clarify.py`) exists on disk but is **not imported or used**.

---

## The Agent — `rag/agent.py`

### LLM backend — `get_llm()`

```python
if settings.llm_backend == "openai-compatible":
    OpenAILike(api_base, api_key, model=settings.openai_model,
               is_chat_model=True, is_function_calling_model=True, ...)
else:
    Ollama(model=settings.llm_model, base_url=settings.active_ollama_url, ...)
```

`is_function_calling_model=True` is critical for `OpenAILike` — without it, AgentWorkflow falls back to ReAct text parsing which breaks tool calling. Do NOT use `response_format: json_object` — it interferes with tool call emission.

`temperature=0.0` is mandatory (deterministic compliance answers).

### Response Schema — `ComplianceAnswer`

```python
class Citation(BaseModel):
    source_number: int = 0   # matches [Source N] from search results (default 0 for backwards compat)
    doc_title: str           # copied verbatim from source header
    section: str
    clause: str              # empty string if no clause
    clause_number: str
    quote: str               # verbatim from policy text

class Escalation(BaseModel):
    needed: bool
    reason: str

class ComplianceAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    escalation: Escalation
```

Final response must be **pure JSON**. Parsed by `rag/response.py:parse_agent_response()` (handles code fences, embedded JSON, fallback).

System prompt enforces:
- Pass user's ORIGINAL question to `search_policies` (no keyword extraction)
- Quote policy text VERBATIM (character-for-character clause names)
- Cite ALL relevant sources for multi-source questions
- Escalate only after reading ALL sources

---

## Vanilla Pipeline — `rag/pipeline.py`

Single LLM call, no agent loop. Traced with manual Phoenix spans (`vanilla_rag_pipeline` → `search_policies` → `llm_call`).

```python
def run_query(question: str) -> dict:
    sources = search_policies(question)  # search + rerank + format
    if "NO_RELEVANT_POLICY_FOUND" in sources:
        return {escalation: needed=True}
    response = httpx.post(ollama/api/chat, ...)  # single call
    return parse_response(response)
```

Note: vanilla pipeline calls Ollama directly via `httpx`. To use llama-server with vanilla mode, the pipeline would need to be updated.

Switch globally via `PIPELINE_MODE=agentic|vanilla` (used by FastAPI route and Teams bot).

---

## Microsoft Teams Bot — `channels/teams/`

### Architecture

Polls `https://graph.microsoft.com/v1.0/me/chats` every 5 seconds. On new message:
1. Show loading indicator
2. Run RAG **directly** (no HTTP — `from rag.agent import build_agent` or `from rag.pipeline import run_query`)
3. Render HTML reply (Teams supports limited tags — use `<p>`, `<b>`, `<i>`, `<ul>/<li>`, `<hr>`. Avoid `<div style=...>`)
4. Send rating prompt; store pending rating context
5. Next message: if `"-1"`, `"0"`, `"1"`, `"2"` exactly → save feedback, clear state. Otherwise process as new question.

### Auth — `channels/teams/auth.py`

`TokenRefresher` prefers saved file (`channels/teams/data/refresh_token.json` — rotated copy) over `.env` (initial seed). Refresh token rotation is automatic.

Required `.env` vars (one-time): `TEAMS_TENANT_ID`, `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET`, `TEAMS_REFRESH_TOKEN`. To get the initial refresh token, the original `legal-compliance-qa-agent` repo has `scripts/get_refresh_token.py` (device code flow).

### Rating values

- **-1** = bot answered, but should have been escalated
- **0** = wrong
- **1** = partially correct
- **2** = correct

Detection rule: `message.strip() in {"-1","0","1","2"}` exactly. Anything else (typos, sentences, Cyrillic) → treat as new question, drop pending state.

### Feedback storage — `channels/teams/feedback.py`

Writes to BOTH:
- `channels/teams/data/feedback.jsonl` (append-only, easy to load with pandas)
- `channels/teams/data/feedback.db` (SQLite with indexes on `rating`, `timestamp`)

Schema: `id, timestamp, user, chat_id, question, answer, citations (JSON), rating`.

### State files

`channels/teams/data/`:
- `bot_state.json` — last_check timestamp + processed message IDs
- `refresh_token.json` — current refresh token (rotated)
- `bot.pid` — PID lock (prevents multiple instances)
- `feedback.jsonl` / `feedback.db`

All gitignored.

### Entry point

```bash
PYTHONPATH=. python scripts/start_teams_bot.py
```

---

## FastAPI — `api/`

`api/main.py`: initializes Phoenix tracing, sets up CORS, registers `/health` and routes.

`api/routes/query.py`: `POST /api/query` switches between agentic and vanilla based on `PIPELINE_MODE`.

```bash
./scripts/start_api.sh      # → uvicorn on port 8000
curl http://localhost:8000/health
curl -s http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the policy on software installation?"}'
```

Note: the Teams bot does **not** call this API. It imports the RAG layer directly. The API is for other clients (or testing).

---

## Observability — Phoenix

- Docker service on port 6006
- `init_observability()` must be called once before any LlamaIndex / Ollama imports
  - Called by: `api/main.py`, `scripts/test_query.py`, `scripts/start_teams_bot.py`, `eval/run_experiment.py`
- Auto-instruments: LLM, embeddings, tool calls (LlamaIndex)
- Manual spans: `hybrid_search`, `vanilla_rag_pipeline`, `search_policies`, `llm_call`
- In Docker remote: bot uses `PHOENIX_ENDPOINT=http://phoenix:6006/v1/traces` (override in compose)

---

## Evaluation System — Phoenix Datasets + Experiments

### Tiers

| Tier | Dataset | Cases | What it tests |
|------|---------|-------|---------------|
| Tier 1 | `retrieval-test-v1` | 70 | Retrieval — chunk in top-k by doc+section+clause |
| Tier 2 | `e2e-test-v1` | 25 | Full agent Q&A — citation + answer coverage |
| Chatbot | `chatbot-test-v1` | 61 | Realistic user questions (CB-NNN) |

### Multi-citation support — `match_mode`

Test cases can have `"match_mode": "any"` — eval passes if **any** expected citation matches. Useful when multiple policies legitimately address the same question. Default is `"all"` (existing behavior).

Applies to: `hit_evaluator`, `citation_doc_accuracy`, `citation_section_accuracy`, `citation_clause_accuracy`. `mrr_evaluator` already takes the best rank, so no changes needed.

### Running

```bash
# Upload datasets (first time / after changes)
python scripts/make_dataset.py eval/datasets/<file>.json

# Run experiments
python eval/run_experiment.py --tier chatbot --mode agentic --name baseline-v1
python eval/run_experiment.py --tier chatbot --mode vanilla --name vanilla-v1
```

`--mode` defaults to `agentic`. Auto-generated experiment name includes mode + embedding + reranker config.

Metadata captured per experiment includes infra info (`infra: local|remote`, `llm_url`, `embedding_url`, `qdrant_url`, `reranker_backend`, `reranker_url`) — visible as columns in Phoenix.

---

## Configuration — `.env`

Key toggles:

```bash
# LLM backend & infrastructure
LLM_BACKEND=ollama                   # or openai-compatible
USE_REMOTE_OLLAMA=false              # true → use OLLAMA_REMOTE_URL
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REMOTE_URL=http://192.168.100.2:11434
LLM_MODEL=qwen2.5:32b-instruct-q8_0  # for ollama backend
OPENAI_API_BASE=http://localhost:8082/v1
OPENAI_MODEL=qwen2.5-32b             # for openai-compatible backend

# Embeddings (HuggingFace OR Ollama)
EMBEDDING_SOURCE=huggingface         # or ollama
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-1b-v2  # or embeddinggemma, qwen3-embedding
EMBEDDING_QUERY_PREFIX=query:        # model-specific (\n converted at runtime)
EMBEDDING_PASSAGE_PREFIX=passage:
OLLAMA_EMBEDDING_URL=http://localhost:11434

# Qdrant — local or remote
USE_REMOTE_QDRANT=false
QDRANT_URL=http://localhost:6333
QDRANT_REMOTE_URL=http://192.168.100.2:6333
QDRANT_VECTOR_DIM=2048               # 2048 nemotron, 768 gemma, 4096 qwen3

# Reranker
RERANKER_ENABLED=true
RERANKER_BACKEND=llama-server        # or vllm
RERANKER_URL=http://localhost:8081
RERANKER_MODEL=qwen3-reranker-4b-q8
RERANKER_TOP_N=6
RERANKER_CANDIDATES=25
RERANKER_QUERY_TEMPLATE=<Instruct>: {instruction}\n<Query>: {query}

# Pipeline mode (selects agentic vs vanilla everywhere)
PIPELINE_MODE=agentic                # or vanilla

# Teams bot
TEAMS_TENANT_ID=...
TEAMS_CLIENT_ID=...
TEAMS_CLIENT_SECRET=...
TEAMS_REFRESH_TOKEN=...

# Observability
PHOENIX_ENDPOINT=http://localhost:6006/v1/traces  # overridden in remote compose to http://phoenix:6006/v1/traces
```

`\n` in `.env` is stored literally — code converts `\\n` → `\n` for `RERANKER_QUERY_TEMPLATE` and `EMBEDDING_QUERY_PREFIX`.

---

## Docker Deployment

### Local dev

```bash
docker compose up -d   # Qdrant + Phoenix
```

### Remote host (production)

```bash
# Bot polls Microsoft Graph (outbound only — no inbound port needed)
# Connects to Spark for: LLM (Ollama), embedding (Ollama), reranker (vLLM), Qdrant
docker compose -f docker-compose-remote.yml up -d --build
```

**Files**:
- `Dockerfile` — Python 3.12-slim base, copies only runtime code (config, rag, channels, scripts/start_teams_bot.py). No `eval/`, no `ingest/`, no `policies/`. Ingestion is a one-time admin task done before deploying.
- `.dockerignore` — excludes `.venv`, datasets, policies, dev tooling, `.env`
- `docker-compose-remote.yml` — bot + phoenix services; phoenix has healthcheck (Python urllib check) so bot only starts after phoenix is ready

Volumes:
- `teams_data` → `/app/channels/teams/data` (persists feedback DB, refresh token, bot state)
- `phoenix_data` → `/data` (traces)

Ingestion before first deploy:
```bash
PYTHONPATH=. python scripts/ingest_all.py --folder ./policies
```

---

## Implementation Status

### Completed
- [x] Docker: Qdrant + Phoenix (local), bot + Phoenix (remote)
- [x] Ingestion pipeline with cross-numId numbering fix
- [x] Hybrid search + reranker (universal `/v1/rerank` client supporting llama-server and vLLM)
- [x] Dual LLM backend (Ollama + OpenAI-compatible)
- [x] Remote stack support (Spark)
- [x] Agentic RAG (LlamaIndex AgentWorkflow + structured JSON ComplianceAnswer)
- [x] Vanilla RAG (single LLM call, no LlamaIndex, manually traced)
- [x] FastAPI `/api/query` and `/health`
- [x] Phoenix observability (auto + manual spans, infra metadata in experiments)
- [x] Eval system: tier1, tier2, chatbot — with `match_mode='any'` for multi-citation
- [x] Microsoft Teams bot with feedback loop (-1, 0, 1, 2) → JSONL + SQLite
- [x] Docker images and remote deployment

### Not Yet Implemented
- [ ] Email escalation notifications
- [ ] Frontend (React)
- [ ] Tier 3 escalation evaluators (dataset exists)
- [ ] pytest test suite

---

## Critical Requirements & Constraints

### Must Never Violate
- `temperature=0.0` on LLM at all times
- Agent must never answer without citing a retrieved chunk
- If `search_policies` returns `NO_RELEVANT_POLICY_FOUND` → escalate
- Citations must only come from retrieved chunk metadata (no hallucinations)
- For `OpenAILike`, set `is_function_calling_model=True` (NOT `response_format=json_object`)

### Performance Targets
- Local M4 Pro, qwen2.5:32b-q8 via Ollama: ~50-90s per agentic query (multi-step)
- Vanilla pipeline: ~30-40s (single LLM call)
- Remote Spark: faster wall-clock for big models
- Qdrant search latency: <200ms

### Security
- `.env` is gitignored; `channels/teams/data/` is gitignored
- Bot has no inbound network exposure (polls outbound only)
- Refresh token rotation is automatic; persisted to file

---

## Common Pitfalls — Avoid These

| Pitfall | Fix |
|---|---|
| Word numbering jumps between sections | `NumberingResolver` continues counter across decimal numIds at level 0 |
| `from ... import _last_search_results` returns stale results | Use `import rag.tools.search_policies as sp` then `sp._last_search_results` (rebinding via `global` doesn't update prior `from ... import ...` references) |
| Reranker scores compressed when using vLLM with Qwen3 | Use `RERANKER_BACKEND=vllm` — wraps query+documents with chat template (`<\|im_start\|>` + `<Document>:` + think suffix) |
| `\n` in `.env` sent literally to reranker / embedding | Code converts `\\n` → `\n` in `_build_query` and `_ollama_embed` |
| `OpenAILike` agent emits ReAct JSON instead of tool calls | Set `is_function_calling_model=True`. Do NOT enable `response_format=json_object`. |
| Eval logs wrong (unranked) results | `_logged_search_policies` calls real `search_policies` then reads `sp._last_search_results` via module attribute |
| Teams bot rendering looks bad / duplicates | Teams supports limited HTML — use `<p>`, `<ul>/<li>`, `<hr>`. Don't repeat the answer text in citations |
| Bot can't reach Phoenix in Docker | Override `PHOENIX_ENDPOINT=http://phoenix:6006/v1/traces` in compose |
| Embedding dim mismatch on re-ingestion | Delete Qdrant collection before re-ingesting with a different embedding model |
| Teams refresh token not loading | Bot prefers `data/refresh_token.json` (rotated) over `.env` (seed) |
| Multi-citation eval flagging valid answers | Add alternative `expected_citations` + `"match_mode": "any"` to test case |
| `parse_agent_response` import path | Now in `rag/response.py`; `eval/agent_wrapper.py` re-exports for backwards compat |
| Eval's `pipeline_wrapper.run_pipeline_task` is async | Phoenix's sync `run_experiment` requires sync tasks — keep it `def`, not `async def` |
| Eval's `match_mode` skipped per-test-case | Set `"match_mode": "any"` at the top level of the test case JSON, not inside `expected_citations` |
| Phoenix not initialized before LlamaIndex | Always `init_observability()` first in entry points |
| Reranker model name doesn't match served name | vLLM: model name must match `--served-model-name` or HF id. llama-server: any non-empty value works. |
