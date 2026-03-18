# Compliance Q&A Bot — Claude Code Instructions

## Project Overview

Internal Compliance Q&A Bot using **Agentic RAG** architecture.
Answers employee questions **strictly from approved internal policy documents (DOCX, 52 files)**.
If an answer cannot be grounded in policy, it escalates to the Compliance team with full context.

**LLM: Local only via Ollama. No external API calls for inference.**

**Current state:** Ingestion pipeline, agent with 4 tools, hybrid search (vector + BM25), observability (Phoenix), and evaluation harness are all implemented and working. API, DB, and frontend are not yet built.

---

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Ollama (`qwen2.5:14b` default) | Local, no cloud |
| Embedding | `nomic-embed-text` via Ollama | 768-dim vectors |
| Vector Store | Qdrant (Docker, port 6333) | Cosine similarity |
| Keyword Search | BM25 (pure Python, JSON persistence) | Toggleable via `BM25_ENABLED` |
| Search Fusion | Reciprocal Rank Fusion (RRF, k=60) | Combines vector + BM25 |
| RAG Framework | LlamaIndex | `AgentWorkflow` (ReAct) |
| Agent | LlamaIndex `AgentWorkflow` | 4 tools, `temperature=0.0` |
| Document Parsing | `python-docx` + `NumberingResolver` | Resolves Word auto-numbering |
| Observability | Arize Phoenix (Docker, port 6006) | Traces all LLM/tool/retrieval calls |
| Evaluation | 4-tier harness (`scripts/run_eval.py`) | Retrieval, E2E, Escalation, Chatbot |
| Backend API | FastAPI | **Not yet implemented** |
| Frontend | React + Tailwind CSS | **Not yet implemented** |
| Metadata DB | SQLite (via SQLAlchemy) | **Not yet implemented** |

---

## Project Structure

```
compliance-bot/
├── CLAUDE.md                         # This file
├── .env / .env.example               # Environment variables
├── config.py                         # pydantic-settings, @lru_cache singleton
├── docker-compose.yml                # Qdrant + Phoenix
├── requirements.txt
│
├── ingest/
│   ├── __init__.py
│   ├── numbering.py                  # NumberingResolver — resolves Word auto-numbering
│   ├── docx_parser.py                # Structure-aware DOCX chunker (headings + ilvl)
│   ├── chunk_models.py               # PolicyChunk pydantic model
│   └── pipeline.py                   # parse → embed → upsert (Qdrant + BM25)
│
├── rag/
│   ├── __init__.py
│   ├── embeddings.py                 # Ollama nomic-embed-text wrapper
│   ├── vector_store.py               # Qdrant collection + CRUD
│   ├── bm25_index.py                 # Pure-Python BM25 index (JSON persistence)
│   ├── hybrid_search.py              # RRF fusion (vector + BM25)
│   ├── observability.py              # Phoenix/OpenTelemetry init + tracer
│   ├── agent.py                      # AgentWorkflow + system prompt
│   └── tools/
│       ├── __init__.py               # ALL_TOOLS list
│       ├── search_policies.py        # Tool 1: hybrid/vector search
│       ├── get_section.py            # Tool 2: fetch full section by doc_id
│       ├── clarify.py                # Tool 3: ask user clarification
│       └── escalate.py               # Tool 4: escalate to Compliance
│
├── eval/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── retrieval_test.json       # 70 test cases — retrieval accuracy
│   │   ├── e2e_test.json             # 25 test cases — full pipeline Q&A
│   │   ├── escalation_test.json      # 6 test cases — should-escalate
│   │   └── chatbot_test_cases.json   # 120 test cases — positive/negative pairs
│   └── results/                      # Auto-generated eval run outputs
│
├── scripts/
│   ├── ingest_all.py                 # CLI: ingest all DOCX from folder
│   ├── test_query.py                 # CLI: test agent with a query
│   ├── run_eval.py                   # CLI: run evaluation harness (4 tiers)
│   ├── convert_eval_xlsx.py          # XLSX → JSON for 3-tier eval datasets
│   └── convert_chatbot_xlsx.py       # XLSX → JSON for chatbot test cases
│
├── notebooks/
│   ├── test_chunking_retrieval.ipynb  # Interactive chunking + search testing
│   ├── eval.ipynb
│   └── test.ipynb
│
├── api/                              # FastAPI — NOT YET IMPLEMENTED
│   ├── __init__.py
│   └── routes/__init__.py
│
├── db/                               # SQLAlchemy — NOT YET IMPLEMENTED
│   └── __init__.py
│
├── notification/                     # Email escalation — NOT YET IMPLEMENTED
│   └── __init__.py
│
└── tests/
    └── __init__.py
```

---

## Document Ingestion Pipeline

### DOCX Parser — `ingest/docx_parser.py`

**Structure-aware chunking** — never splits by token count.

Key components:
- `NumberingResolver` (`ingest/numbering.py`): Simulates Word's `numbering.xml` counters to resolve auto-generated numbers (e.g. `7.5.`) that are NOT in `para.text`
- `extract_heading_level()`: Detects Heading 1/2/3 styles
- `extract_clause_name()`: Extracts bold-run label from ilvl=1 paragraphs (e.g. "Blogging and Social Media" from bold text before colon)
- `ilvl=0 + decimal` → section (sets `section`, `section_number`)
- `ilvl=1 + decimal` → clause (sets `clause`, `clause_number`)
- `ilvl=2+ / bullet` → content under clause
- Tables converted to `"Header: Value | Header: Value"` rows
- Min chunk: 50 tokens, Max: 400 tokens (split at sentence boundary if oversized)
- Undersized chunks accumulated and merged within same section

### Chunk Metadata — `ingest/chunk_models.py`

```python
class PolicyChunk(BaseModel):
    chunk_id: str             # uuid
    doc_id: str               # slugified filename: "acceptable-use-policy-internal"
    doc_title: str            # "Acceptable Use Policy [Internal]"
    doc_filename: str         # "Acceptable Use Policy [Internal].docx"
    doc_link: str             # URL to source document

    section: str = ""         # "Private Information" (name only, no number)
    section_number: str = ""  # "7"
    clause: str = ""          # "Blogging and Social Media" (name only)
    clause_number: str = ""   # "7.5"

    section_display: str = "" # "7. Private Information > 7.5. Blogging and Social Media"

    text: str                 # chunk content (includes prepended number)
    char_count: int = 0
    chunk_index: int = 0
    last_updated: str = ""
```

### Ingestion Pipeline — `ingest/pipeline.py`

```
parse_docx() → embed_texts() → upsert_chunks() to Qdrant
                              → add_chunks_to_bm25() (if BM25_ENABLED)
```

On re-ingestion: old chunks deleted from both Qdrant and BM25 before inserting new ones.

**CLI:**
```bash
PYTHONPATH=. python scripts/ingest_all.py --folder ./policies
```

Current stats: **52 documents, 1878 chunks**.

---

## Search Architecture

### Hybrid Search — `rag/hybrid_search.py`

When `BM25_ENABLED=true` (default):
1. Vector search via Qdrant (top 20 candidates)
2. BM25 keyword search (top 20 candidates)
3. Reciprocal Rank Fusion (k=60) merges results
4. Return top-k by RRF score

When `BM25_ENABLED=false`: vector-only search (original behavior).

**BM25 Index** (`rag/bm25_index.py`):
- Pure Python, no external dependencies
- Okapi BM25 (k1=1.2, b=0.75)
- Persisted to `.bm25_index.json` (1.4MB)
- Stop words removed, tokens = lowercase alphanumeric

**Confidence threshold:** If no result has `vector_score >= 0.45` → returns `NO_RELEVANT_POLICY_FOUND`.

### Search Output Format

```
Document: Acceptable Use Policy [Internal] | Section: 4. Corporate Workstation and Software Use | Clause: 4.7. Software Installation | Doc ID: acceptable-use-policy-internal | Link: ...
Text: 4.7. Software Installation: Team Members are forbidden to install...
```

---

## The 4 Agent Tools

| Tool | File | Purpose |
|------|------|---------|
| `search_policies` | `rag/tools/search_policies.py` | Hybrid/vector search over policy docs. Always call FIRST. |
| `get_section` | `rag/tools/get_section.py` | Fetch full section text by doc_id + section_name |
| `ask_clarification` | `rag/tools/clarify.py` | Ask user for clarity when question is ambiguous |
| `escalate_to_compliance` | `rag/tools/escalate.py` | Escalate when no policy found or answer is uncertain |

---

## The Agent — `rag/agent.py`

- Uses `AgentWorkflow.from_tools_or_functions()` (LlamaIndex)
- LLM: `Ollama(model=settings.llm_model, temperature=0.0)`
- **temperature=0.0 is mandatory** — compliance answers must be deterministic
- System prompt enforces: search first → cite verbatim → escalate if unsure → respond in English

---

## Observability — Phoenix

- Docker service on port 6006 (`docker-compose.yml`)
- `rag/observability.py`: call `init_observability()` before any LlamaIndex imports
- Auto-instruments: all LLM calls, tool calls, embeddings via `LlamaIndexInstrumentor`
- Custom spans: `hybrid_search` (vector/BM25 scores), `escalation` events
- Phoenix UI: http://localhost:6006

---

## Evaluation Harness — `scripts/run_eval.py`

4-tier evaluation system:

| Tier | Cases | What it tests | LLM needed? |
|------|-------|---------------|-------------|
| `retrieval` | 70 | Correct chunk in top-k by doc_title, section, clause, text match | No |
| `e2e` | 25 | Full agent Q&A — citation accuracy + fact coverage | Yes |
| `escalation` | 6 | Bot correctly escalates (doesn't answer) | Yes |
| `chatbot` | 60 unique Qs | Positive vs negative answer scoring (pass if pos > neg) | Yes |

**Usage:**
```bash
PYTHONPATH=. python scripts/run_eval.py --tier retrieval  --tag "baseline"
PYTHONPATH=. python scripts/run_eval.py --tier all        --tag "baseline"
```

Results saved to `eval/results/{tier}_{tag}_{timestamp}.json` with config snapshot.

**Important matching rules (retrieval tier):**
- `expected_doc_id` in test JSON contains display title (e.g. "Whistleblowing Policy [Internal]")
- Matched against `doc_title` from Qdrant using normalized slugification
- `expected_text_contains` can be string OR list (any item match = pass)

**XLSX converters:**
```bash
python scripts/convert_eval_xlsx.py <3-tier.xlsx>       # → retrieval, e2e, escalation JSONs
python scripts/convert_chatbot_xlsx.py <chatbot.xlsx>    # → chatbot_test_cases.json
```

---

## Configuration — `.env`

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:14b
EMBEDDING_MODEL=nomic-embed-text
LLM_TEMPERATURE=0.0
LLM_REQUEST_TIMEOUT=120

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=compliance_policies
QDRANT_VECTOR_DIM=768

# Documents
POLICY_DOCS_FOLDER=./policies
POLICY_BASE_URL=http://intranet.company.com/policies

# Retrieval
RETRIEVAL_TOP_K=6
MIN_CONFIDENCE_SCORE=0.45

# Hybrid Search (BM25)
BM25_ENABLED=true
HYBRID_VECTOR_CANDIDATES=20
HYBRID_BM25_CANDIDATES=20

# Agent
AGENT_MAX_ITERATIONS=8
AGENT_TIMEOUT=120

# Chunking
CHUNK_MIN_TOKENS=50
CHUNK_MAX_TOKENS=400

# Escalation Email
SMTP_HOST=smtp.company.com
SMTP_PORT=587
SMTP_USER=bot@company.com
SMTP_PASSWORD=
COMPLIANCE_TEAM_EMAIL=compliance@company.com
ESCALATION_TICKET_PREFIX=ESC

# API
API_SECRET_KEY=changeme
ADMIN_API_KEY=changeme

# SQLite
DATABASE_URL=sqlite:///./compliance_bot.db

# Observability (Phoenix)
PHOENIX_ENABLED=true
PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_PROJECT_NAME=compliance-bot
EVAL_DATASET_PATH=eval/datasets
```

---

## Qdrant Payload Indexes

Collection `compliance_policies` has these indexed fields:
- `doc_id` (KEYWORD)
- `section` (KEYWORD)
- `section_number` (KEYWORD)
- `clause` (KEYWORD)
- `clause_number` (KEYWORD)
- `section_display` (TEXT — full-text search)

If schema changes, delete collection and re-ingest:
```bash
PYTHONPATH=. python -c "from qdrant_client import QdrantClient; QdrantClient('http://localhost:6333').delete_collection('compliance_policies')"
PYTHONPATH=. python scripts/ingest_all.py
```

---

## Implementation Status

### Completed
- [x] Docker: Qdrant + Phoenix (`docker-compose up -d`)
- [x] Ollama: `qwen2.5:14b` + `nomic-embed-text` models
- [x] `ingest/numbering.py` — Word auto-numbering resolver
- [x] `ingest/docx_parser.py` — structure-aware chunking with ilvl hierarchy
- [x] `ingest/pipeline.py` — parse → embed → Qdrant + BM25
- [x] `rag/embeddings.py` + `rag/vector_store.py`
- [x] `rag/bm25_index.py` — pure Python BM25
- [x] `rag/hybrid_search.py` — RRF fusion
- [x] 4 agent tools (`search_policies`, `get_section`, `clarify`, `escalate`)
- [x] `rag/agent.py` — AgentWorkflow with system prompt
- [x] `rag/observability.py` — Phoenix tracing
- [x] `scripts/ingest_all.py` — 52 docs, 1878 chunks ingested
- [x] `scripts/test_query.py` — interactive agent testing
- [x] `scripts/run_eval.py` — 4-tier evaluation harness
- [x] XLSX → JSON converter scripts
- [x] Notebooks for interactive testing

### Not Yet Implemented
- [ ] `api/` — FastAPI REST endpoints
- [ ] `db/` — SQLAlchemy models (Escalation, Session, Message)
- [ ] `notification/email.py` — SMTP escalation emails
- [ ] `frontend/` — React chat UI
- [ ] `tests/` — pytest suite

---

## Critical Requirements & Constraints

### Must Never Violate
- `temperature=0.0` on LLM at all times
- Agent must never answer without citing a retrieved chunk
- If `search_policies` returns `NO_RELEVANT_POLICY_FOUND` → next call must be `escalate_to_compliance`
- No external HTTP calls during inference (no web search, no remote APIs)
- Escalation must save the **full conversation context**, not just the last message

### Performance Targets
- Single-query response: < 30s on M4 Pro with `qwen2.5:14b`
- Ingestion of 52 documents: ~2 minutes
- Qdrant search latency: < 200ms

### Security
- Admin endpoints require `X-Admin-Key` header
- User endpoints require `session_id` (UUID, client-generated)
- No PII stored beyond what's in the conversation
- Qdrant not exposed outside localhost

---

## Common Pitfalls — Avoid These

| Pitfall | Fix |
|---|---|
| Chunking by token count | Always chunk by document structure (headings + ilvl numbering) |
| Missing clause numbers in DOCX | Use `NumberingResolver` — Word auto-numbers are NOT in `para.text` |
| `expected_doc_id` mismatch in eval | Test JSONs use display title; match against `doc_title` with `normalize_doc_id()` |
| Single retrieval call per question | For complex questions, agent must call `search_policies` multiple times |
| Hallucinated citations | Citations must only come from retrieved chunk metadata |
| Large DOCX tables broken | Convert all table cells to key:value text before chunking |
| Ollama timeout on 70B model | Set `request_timeout=180`, use streaming response |
| Qdrant empty results on first run | Always run `ingest_all.py` before starting |
| BM25 index stale after re-ingestion | Pipeline auto-syncs BM25 on ingest (if `BM25_ENABLED=true`) |
| PYTHONPATH not set for scripts | Run all scripts with `PYTHONPATH=. python scripts/...` |
