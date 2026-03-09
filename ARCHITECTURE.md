# Compliance Q&A Bot — Architecture

## System Overview

An **Agentic RAG** (Retrieval-Augmented Generation) system that answers employee compliance questions strictly from approved internal DOCX policy documents. All inference runs locally via Ollama — zero external API calls.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  DOCX Files │────▶│  Ingestion   │────▶│  Qdrant (6333)  │
│  /policies/ │     │  Pipeline    │     │  Vector Store    │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
┌─────────────┐     ┌──────────────┐               │
│    User     │────▶│  ReAct Agent │◀──────────────┘
│   Query     │     │  (4 tools)   │
└─────────────┘     └──────┬───────┘
                           │
                    ┌──────▼───────┐     ┌─────────────────┐
                    │   Ollama     │     │ Phoenix (6006)   │
                    │  qwen2.5:14b │     │ Observability    │
                    └──────────────┘     └─────────────────┘
                                                ▲
                           OpenTelemetry spans  │
                    (auto-instrumented via ─────┘
                     LlamaIndex instrumentor)
```

---

## Component Map

```
compliance-bot/
├── config.py                          # Centralized settings (singleton)
│
├── ingest/                            # DOCUMENT INGESTION
│   ├── chunk_models.py                #   PolicyChunk Pydantic model
│   ├── docx_parser.py                 #   Structure-aware DOCX → chunks
│   └── pipeline.py                    #   Parse → embed → upsert orchestrator
│
├── rag/                               # RAG + AGENT
│   ├── observability.py               #   Phoenix tracing initialization
│   ├── embeddings.py                  #   Ollama nomic-embed-text wrapper
│   ├── vector_store.py                #   Qdrant client + operations
│   ├── agent.py                       #   ReAct agent + system prompt
│   └── tools/                         #   4 agent tools
│       ├── search_policies.py         #     Tool 1: semantic search
│       ├── get_section.py             #     Tool 2: exact clause fetch
│       ├── clarify.py                 #     Tool 3: ask user for clarity
│       └── escalate.py                #     Tool 4: escalate to team
│
├── scripts/                           # CLI UTILITIES
│   ├── ingest_all.py                  #   Batch ingest documents
│   └── test_query.py                  #   Test agent (single / interactive)
│
├── docker-compose.yml                 #   Qdrant + Phoenix containers
├── .env                               #   Runtime config values
└── requirements.txt                   #   Python dependencies
```

---

## Data Flow

### 1. Ingestion Pipeline

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    scripts/ingest_all.py                      │
  │              (--folder ./policies --base-url ...)             │
  └────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    ingest/pipeline.py                         │
  │                                                              │
  │  ingest_folder(folder, base_url)                             │
  │    └─▶ for each .docx:                                       │
  │          ingest_document(filepath, doc_link)                  │
  │            ├─▶ parse_docx()          → list[PolicyChunk]     │
  │            ├─▶ delete_document()     → remove old version    │
  │            ├─▶ embed_texts()         → list[list[float]]     │
  │            └─▶ upsert_chunks()       → write to Qdrant      │
  └──────────────────────────────────────────────────────────────┘
```

**Step-by-step:**

1. `scripts/ingest_all.py` takes `--folder` and `--base-url` CLI args (defaults from `.env`)
2. `pipeline.ingest_folder()` iterates all `.docx` files
3. For each file, `pipeline.ingest_document()`:
   - Calls `docx_parser.parse_docx()` → returns `list[PolicyChunk]`
   - Calls `vector_store.delete_document(doc_id)` → removes stale chunks (supports re-ingestion)
   - Calls `embeddings.embed_texts()` → generates 768-dim vectors via Ollama
   - Calls `vector_store.upsert_chunks()` → writes points to Qdrant in batches of 100

### 2. Query Pipeline

```
  ┌─────────────┐
  │  User Query  │
  └──────┬──────┘
         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                      rag/agent.py                            │
  │                                                              │
  │  AgentWorkflow (ReAct loop, max 8 iterations)                │
  │  ┌────────────────────────────────────────────────────┐      │
  │  │ LLM: Ollama qwen2.5:14b (temp=0.0)                │      │
  │  │                                                    │      │
  │  │ Thought → Action → Observation → ... → Answer      │      │
  │  │                                                    │      │
  │  │ Available actions (tools):                         │      │
  │  │   1. search_policies(query, top_k)                 │      │
  │  │   2. get_section(doc_id, clause_number)            │      │
  │  │   3. ask_clarification(question_to_user)           │      │
  │  │   4. escalate_to_compliance(reason, question)      │      │
  │  └────────────────────────────────────────────────────┘      │
  └──────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
                    Cited answer or escalation
```

**Typical happy-path sequence:**

```
User: "How many annual leave days do employees get?"
  │
  ▼
Agent (Thought): I need to search the policies
  │
  ├──▶ Tool: search_policies("annual leave entitlement days")
  │      │
  │      ├──▶ embeddings.embed_query(query)
  │      │      └──▶ Ollama nomic-embed-text → 768-dim vector
  │      │
  │      ├──▶ vector_store.search_vectors(vector, top_k=6)
  │      │      └──▶ Qdrant query_points (cosine similarity)
  │      │
  │      └──▶ Returns: "[SCORE: 0.83] Document: Annual Leave Policy
  │                      | Section: 2. Entitlement | Clause: 2.1 ..."
  │
  ▼
Agent (Thought): I have relevant policy text, formulate answer
  │
  ▼
Agent (Answer):
  **Answer:** Full-time employees are entitled to 25 days...
  **Policy Sources:**
  - Annual Leave Policy | 2. Entitlement | Clause 2.1
    > "Full-time employees are entitled to 25 days..."
```

**Escalation path (no relevant policy found):**

```
User: "Can we share client data with our parent company in Germany?"
  │
  ├──▶ Tool: search_policies("sharing client data parent company Germany")
  │      └──▶ Returns: "NO_RELEVANT_POLICY_FOUND" (top score < 0.45)
  │
  ├──▶ Tool: escalate_to_compliance(
  │      reason="No relevant policy found for cross-border data sharing",
  │      unanswered_question="Can we share client data with..."
  │    )
  │      └──▶ Creates ticket ESC-2026-0001, stores in memory
  │
  ▼
Agent: "I was unable to find a confirmed answer...
        forwarded to the Compliance team (Ticket #ESC-2026-0001)"
```

---

## Component Details

### config.py — Centralized Settings

Single `Settings` class (pydantic-settings), loaded once as a singleton via `@lru_cache`. Every module imports `from config import settings`.

```
.env  ──▶  Settings  ──▶  All modules
```

| Group | Key Settings | Defaults |
|-------|-------------|----------|
| LLM | `llm_model`, `llm_temperature`, `llm_request_timeout` | `qwen2.5:14b`, `0.0`, `120` |
| Embedding | `embedding_model` | `nomic-embed-text` |
| Qdrant | `qdrant_url`, `qdrant_collection`, `qdrant_vector_dim` | `localhost:6333`, `compliance_policies`, `768` |
| Chunking | `chunk_min_tokens`, `chunk_max_tokens` | `50`, `400` |
| Retrieval | `retrieval_top_k`, `min_confidence_score` | `6`, `0.45` |
| Agent | `agent_max_iterations`, `agent_timeout` | `8`, `120` |
| Escalation | `escalation_ticket_prefix`, `smtp_*`, `compliance_team_email` | `ESC`, SMTP placeholders |

---

### ingest/docx_parser.py — Structure-Aware DOCX Chunking

The most critical component. Naive token-splitting destroys compliance document structure. This parser preserves the heading hierarchy and clause boundaries.

**Parsing strategy:**

```
DOCX document body
  │
  ├── Paragraph (Heading 1)  ──▶  Update current_headings[0], flush buffer
  ├── Paragraph (Heading 2)  ──▶  Update current_headings[1], flush buffer
  ├── Paragraph (normal)     ──▶  Check for clause number, buffer text
  ├── Paragraph (clause 2.1) ──▶  Flush prev clause, start new buffer
  ├── Paragraph (clause 2.2) ──▶  Flush prev clause, start new buffer
  ├── Table                  ──▶  Convert to "Header: Value" text, append to buffer
  └── ...
```

**Key design decisions:**

1. **Heading hierarchy tracking** — maintains a 3-level stack `[h1, h2, h3]`. When a Heading N appears, lower levels are cleared. This builds the `section_path` metadata.

2. **Clause boundary detection** — regex `^\d+(\.\d+)*\.?\s` identifies numbered clauses (1.1, 4.2.1, etc.). Each new clause flushes the previous buffer.

3. **Small chunk accumulation** — chunks under `chunk_min_tokens` (50) are accumulated with neighboring content under the same section heading, rather than being discarded. This prevents loss of short but meaningful clauses.

4. **Oversized chunk splitting** — chunks exceeding `chunk_max_tokens` (400) are split at sentence boundaries (regex `(?<=[.!?])\s+`).

5. **Table handling** — tables are converted to text rows (`"Column: Value | Column: Value"`) and attached to the current section's buffer.

6. **Document order iteration** — iterates `doc.element.body` directly (not `doc.paragraphs`) to process paragraphs and tables in their actual document order.

**Output:** flat list of `PolicyChunk` objects, each with:

```python
PolicyChunk(
    chunk_id="uuid",
    doc_id="annual-leave-policy",         # slugified filename
    doc_title="Annual Leave Policy",       # prettified filename
    doc_filename="annual_leave_policy.docx",
    doc_link="http://.../annual_leave_policy.docx",
    section_path=["Annual Leave Policy", "2. Entitlement"],
    section_display="Annual Leave Policy > 2. Entitlement",
    clause_number="2.1",
    text="2.1 Full-time employees are entitled to...",
    char_count=342,
    chunk_index=1,
    last_updated="2026-03-04T..."
)
```

---

### rag/embeddings.py — Ollama Embedding Wrapper

Thin wrapper around LlamaIndex's `OllamaEmbedding`. Singleton pattern — the model is loaded once on first use.

```
embed_texts(["chunk1", "chunk2", ...])  →  [[768 floats], [768 floats], ...]
embed_query("user question")            →  [768 floats]
```

Uses `nomic-embed-text` model (768 dimensions) running locally via Ollama at `http://localhost:11434`.

---

### rag/vector_store.py — Qdrant Operations

Manages the Qdrant collection lifecycle and all read/write operations. Singleton client.

```
                    ┌────────────────────────────────────┐
                    │   Qdrant Collection:               │
                    │   "compliance_policies"             │
                    │                                    │
                    │   Vector: 768-dim, cosine distance │
                    │                                    │
                    │   Payload indexes:                 │
                    │     - doc_id (keyword)             │
                    │     - clause_number (keyword)      │
                    │                                    │
                    │   Each point:                      │
                    │     id = chunk_id (UUID)           │
                    │     vector = [768 floats]          │
                    │     payload = PolicyChunk fields   │
                    └────────────────────────────────────┘
```

**Operations:**

| Function | Used By | Qdrant API |
|----------|---------|------------|
| `init_collection()` | `pipeline.py` | `create_collection` + `create_payload_index` |
| `upsert_chunks(chunks, embeddings)` | `pipeline.py` | `upsert` (batches of 100) |
| `delete_document(doc_id)` | `pipeline.py` | `delete` with filter on `doc_id` |
| `search_vectors(vector, top_k)` | `search_policies` tool | `query_points` (cosine similarity) |
| `scroll_by_filter(filter, limit)` | `get_section` tool | `scroll` with exact match filter |

---

### rag/tools/ — The 4 Agent Tools

The agent decides which tool to call based on each tool's docstring. The docstrings are detailed and prescriptive — the LLM reads them at runtime.

#### Tool 1: search_policies

```
Purpose:   Primary retrieval — semantic search over all policy chunks
Input:     query (str), top_k (int, default 6)
Output:    Formatted chunks with scores, or "NO_RELEVANT_POLICY_FOUND"
Calls:     embeddings.embed_query() → vector_store.search_vectors()
Threshold: top result score < 0.45 → returns NO_RELEVANT_POLICY_FOUND
```

Output format per result:
```
[SCORE: 0.83] Document: Annual Leave Policy | Section: ... | Clause: 2.1 | Link: ...
Text: 2.1 Full-time employees are entitled to...
```

#### Tool 2: get_section

```
Purpose:   Fetch complete clause text by exact doc_id + clause_number
Input:     doc_id (str), clause_number (str)
Output:    Full clause text with metadata, or "No section found"
Calls:     vector_store.scroll_by_filter()
Use case:  When search returned a partial chunk and agent needs full wording
```

#### Tool 3: ask_clarification

```
Purpose:   Ask user to clarify an ambiguous question before searching
Input:     question_to_user (str)
Output:    "CLARIFICATION_NEEDED: {question}"
Use case:  "reporting" could mean incident, regulatory, or colleague
```

#### Tool 4: escalate_to_compliance

```
Purpose:   Forward unanswerable questions to the Compliance team
Input:     reason (str), unanswered_question (str), search_attempted (bool)
Output:    "ESCALATED: ... Ticket #ESC-2026-XXXX ... Reason: ..."
Storage:   In-memory dict (will be replaced by SQLite in Step 9)
Use case:  No policy found, ambiguous policies, legal interpretation needed
```

**Tool selection flow:**

```
                    ┌──────────────────┐
                    │   User Question  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Is question    │──── Yes ───▶ ask_clarification
                    │   ambiguous?     │
                    └────────┬─────────┘
                             │ No
                    ┌────────▼─────────┐
                    │ search_policies  │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
         score ≥ 0.45   score < 0.45   need full
                │            │          clause
                ▼            ▼            ▼
          Cite answer    escalate     get_section
                         _to_          then cite
                       compliance
```

---

### rag/agent.py — ReAct Agent

Builds a LlamaIndex `AgentWorkflow` with all 4 tools and a strict system prompt.

**System prompt rules (enforced by LLM):**

1. ONLY answer from approved internal policy documents
2. NEVER use general knowledge
3. EVERY answer must cite: Document, Section, Clause, Link
4. `NO_RELEVANT_POLICY_FOUND` → MUST call `escalate_to_compliance`
5. Ambiguous/contradictory → escalate
6. Multi-area questions → multiple `search_policies` calls
7. Vague questions → `ask_clarification` first
8. Not a lawyer — cite verbatim, no interpretation

**LLM configuration:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `model` | `qwen2.5:14b` | Best local model for tool-calling |
| `temperature` | `0.0` | **Mandatory** — deterministic compliance answers |
| `request_timeout` | `120s` | Large model inference time |
| `verbose` | `True` | Logs Thought/Action/Observation steps |

**ReAct loop:** The agent follows a Thought → Action → Observation cycle up to 8 iterations, then produces a final answer.

---

## Observability — Arize Phoenix

### rag/observability.py — Tracing Initialization

This module must be called **once at startup, before any LlamaIndex imports**. It connects to Phoenix via OpenTelemetry and auto-instruments all LlamaIndex calls.

```
init_observability()  ──▶  phoenix.otel.register()  ──▶  LlamaIndexInstrumentor
         │                        │                              │
         │                  Sets up OTLP             Auto-instruments:
         │                  HTTP exporter            - Agent workflow steps
         │                  to Phoenix               - LLM calls (prompt/response/tokens)
         │                                           - Embedding calls
         │                                           - Tool invocations
         ▼
   Called from:
   - scripts/test_query.py (before agent import)
   - scripts/ingest_all.py (before pipeline import)
   - FastAPI app startup (future)
```

**What gets traced automatically:**

| Span Type | What's Captured |
|-----------|-----------------|
| `AgentWorkflow.run` | Full agent execution (parent span) |
| `AgentWorkflow.init_run` | Agent initialization |
| `AgentWorkflow.setup_agent` | Tool + LLM setup |
| `AgentWorkflow.run_agent_step` | Each ReAct iteration |
| `AgentWorkflow.parse_agent_output` | Output parsing |
| `Ollama.astream_chat` | LLM generation (model, tokens, latency) |
| `Ollama._prepare_chat_with_tools` | Tool preparation for LLM |

**Manual spans** via `get_tracer()` — for custom instrumentation around search scoring, confidence gates, escalation events.

**Configuration:**

| Setting | Default | Description |
|---------|---------|-------------|
| `PHOENIX_ENABLED` | `true` | Set `false` to disable all tracing |
| `PHOENIX_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP collector endpoint |
| `PHOENIX_PROJECT_NAME` | `compliance-bot` | Groups traces in Phoenix UI |

**Graceful degradation:** If Phoenix is unreachable or packages aren't installed, the app continues without tracing (warning logged, no crash).

---

## Infrastructure

### Docker Services

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"       # REST API
    volumes:
      - qdrant_data:/qdrant/storage   # persistent

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"       # UI + OTLP collector
    volumes:
      - phoenix_data:/data            # persistent traces
    environment:
      - PHOENIX_WORKING_DIR=/data
    restart: unless-stopped
```

Health checks:
- Qdrant: `curl localhost:6333/healthz`
- Phoenix UI: `http://localhost:6006`

> The `phoenix_data` volume persists all traces, experiments, and evaluation scores across container restarts. Only `docker compose down -v` destroys it.

### Ollama (Local)

Two models required:

| Model | Purpose | Dimensions | Pulled via |
|-------|---------|-----------|------------|
| `qwen2.5:14b` | LLM inference (ReAct agent) | — | `ollama pull qwen2.5:14b` |
| `nomic-embed-text` | Text embedding | 768 | `ollama pull nomic-embed-text` |

Both accessed at `http://localhost:11434`.

---

## Module Dependency Graph

```
config.py ◀──────────────────────────────────────────────────────┐
    │                                                            │
    ├──▶ ingest/chunk_models.py                                  │
    │         │                                                  │
    │         ▼                                                  │
    ├──▶ ingest/docx_parser.py ──▶ chunk_models.py               │
    │         │                                                  │
    │         ▼                                                  │
    ├──▶ rag/embeddings.py                                       │
    │         │                                                  │
    │         ▼                                                  │
    ├──▶ rag/vector_store.py ──▶ chunk_models.py                 │
    │         │                                                  │
    │         ▼                                                  │
    ├──▶ ingest/pipeline.py ──▶ docx_parser + embeddings         │
    │    │                      + vector_store                   │
    │    │                                                       │
    │    ▼                                                       │
    │  scripts/ingest_all.py                                     │
    │                                                            │
    ├──▶ rag/tools/search_policies.py ──▶ embeddings             │
    │    │                                + vector_store          │
    │    │                                                       │
    ├──▶ rag/tools/get_section.py ──▶ vector_store               │
    │    │                                                       │
    ├──▶ rag/tools/clarify.py          (no project deps)         │
    │    │                                                       │
    ├──▶ rag/tools/escalate.py                                   │
    │    │                                                       │
    │    ▼                                                       │
    ├──▶ rag/tools/__init__.py ──▶ all 4 tools → ALL_TOOLS       │
    │         │                                                  │
    │         ▼                                                  │
    ├──▶ rag/agent.py ──▶ tools/__init__.ALL_TOOLS               │
    │         │                                                  │
    │         ▼                                                  │
    ├──▶ rag/observability.py     (phoenix.otel + OTel)          │
    │         │                                                  │
    │         ▼                                                  │
    └──▶ scripts/test_query.py ──▶ observability.init()          │
                                   + agent.build_agent()         │
                                                                 │
    All modules ─────────────────────────────────────────────────┘
                 import `from config import settings`
```

---

## External Dependencies

| Package | Version | Used By | Purpose |
|---------|---------|---------|---------|
| `pydantic-settings` | ≥2.0 | `config.py` | `.env` → typed settings |
| `python-docx` | ≥1.1 | `docx_parser.py` | Parse DOCX structure |
| `python-slugify` | ≥8.0 | `docx_parser.py` | Filename → doc_id slug |
| `llama-index-core` | ≥0.12 | `agent.py`, tools | Agent framework, FunctionTool |
| `llama-index-llms-ollama` | ≥0.5 | `agent.py` | Ollama LLM integration |
| `llama-index-embeddings-ollama` | ≥0.5 | `embeddings.py` | Ollama embedding integration |
| `qdrant-client` | ≥1.12 | `vector_store.py`, tools | Vector DB client |
| `arize-phoenix` | ≥8.0 | `observability.py` | Tracing UI + OTLP collector |
| `openinference-instrumentation-llama-index` | ≥3.0 | `observability.py` | Auto-instrument LlamaIndex |
| `opentelemetry-api` | ≥1.0 | `observability.py` | Tracing API |
| `opentelemetry-sdk` | ≥1.0 | `observability.py` | Tracing SDK |
| `fastapi` | ≥0.115 | (Step 10) | REST API — not yet implemented |
| `sqlalchemy` | ≥2.0 | (Step 9) | DB models — not yet implemented |

---

## What's Implemented vs Remaining

| Step | Component | Status |
|------|-----------|--------|
| 0 | Scaffolding, config, venv | Done |
| 1 | Docker Compose (Qdrant) | Done |
| 2 | Ollama models pulled | Done |
| 3 | DOCX parser | Done |
| 4 | Embeddings + vector store | Done |
| 5 | Ingestion pipeline + CLI | Done |
| 6 | Tool 1: search_policies | Done |
| 7 | Tools 2–4: get_section, clarify, escalate | Done |
| 8 | ReAct agent + test script | Done |
| 9 | SQLite DB models (SQLAlchemy) | **Not started** |
| 10 | FastAPI routes | **Not started** |
| 11 | Email notifications | **Not started** |
| 12 | React frontend | **Not started** |
| 13 | Tests | **Not started** |

**Current limitation:** Escalation tickets are stored in-memory (`rag/tools/escalate.py:_escalations` dict). Step 9 will replace this with SQLite via SQLAlchemy.
