# Compliance Q&A Bot — Claude Code Instructions

## Project Overview

Internal Compliance Q&A Bot using **Agentic RAG** architecture.
Answers employee questions **strictly from approved internal policy documents (DOCX, 52 files)**.
If an answer cannot be grounded in policy, it escalates to the Compliance team with full context.

**LLM: Local only via Ollama. No external API calls for inference.**

**Current state:** Ingestion pipeline, agent with 3 tools (structured JSON output), hybrid search (vector + BM25), observability (Phoenix), and Phoenix-based evaluation (Datasets + Experiments) are all implemented and working. API, DB, and frontend are not yet built.

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
| Agent | LlamaIndex `AgentWorkflow` | 3 tools, structured JSON output, `temperature=0.0` |
| Document Parsing | `python-docx` + `NumberingResolver` | Resolves Word auto-numbering |
| Observability | Arize Phoenix (Docker, port 6006) | Traces all LLM/tool/retrieval calls |
| Evaluation | Phoenix Datasets + Experiments | 3 active tiers (Tier 1, 2, Chatbot) |
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
│   ├── evaluators.py                 # Shared evaluator functions (all tiers)
│   ├── agent_wrapper.py              # Instrumented agent (logs tool calls)
│   ├── run_experiment.py             # CLI: runs Phoenix experiments
│   └── datasets/
│       ├── retrieval_test.json       # 70 test cases — retrieval accuracy
│       ├── e2e_test.json             # 25 test cases — full pipeline Q&A
│       ├── escalation_test.json      # 6 test cases — should-escalate (Tier 3, TODO)
│       └── chatbot_test_cases.json   # 61 test cases — chatbot Q&A (same format as e2e)
│
├── scripts/
│   ├── ingest_all.py                 # CLI: ingest all DOCX from folder
│   ├── test_query.py                 # CLI: test agent with a query
│   ├── run_eval.py                   # CLI: legacy eval harness (kept for reference)
│   ├── make_dataset.py               # CLI: upload JSON → Phoenix Dataset
│   └── convert_eval_xlsx.py          # XLSX → JSON for all 4 tiers
│
├── notebooks/
│   └── test_chunking_retrieval.ipynb # Interactive chunking + search testing
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

**Important:** `section` and `clause` store **names only** (no numbers). Numbers are in `section_number` and `clause_number`. The `section_display` field is a formatted combination for display purposes. Evaluation matching uses `section` and `clause` (not `section_display`).

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

Current stats: **52 documents, ~1514 chunks**.

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

**Metadata in search results:** Both hybrid and vector-only paths return `section`, `section_number`, `clause`, `clause_number` as separate fields (not just `section_display`). This is required for evaluation matching and for the agent to copy into structured JSON citations.

---

## The 3 Agent Tools

| Tool | File | Purpose |
|------|------|---------|
| `search_policies` | `rag/tools/search_policies.py` | Hybrid/vector search over policy docs. Always call FIRST. |
| `get_section` | `rag/tools/get_section.py` | Fetch full section text by doc_id + section_name |
| `escalate_to_compliance` | `rag/tools/escalate.py` | Escalate when no policy found or answer is uncertain |

`ask_clarification` (`rag/tools/clarify.py`) exists on disk but is **not imported or used** — agent should always search first.

### Search Output Format

Each result has separate fields the agent copies directly into citations:
```
--- Result 1 [RRF Score: 0.0328] ---
Document: Acceptable Use Policy [Internal]
Section: Corporate Workstation and Software Use
Clause: Software Installation
Clause Number: 4.7
Doc ID: acceptable-use-policy-internal
Text: 4.7. Software Installation: Team Members are forbidden to install...
```

---

## The Agent — `rag/agent.py`

- Uses `AgentWorkflow.from_tools_or_functions()` (LlamaIndex)
- LLM: `Ollama(model=settings.llm_model, temperature=0.0)`
- **temperature=0.0 is mandatory** — compliance answers must be deterministic
- **Structured JSON output** — agent returns `ComplianceAnswer` schema (answer + citations + escalation)
- System prompt enforces: search first → cite verbatim → escalate if unsure → respond in English → final response must be valid JSON
- `ask_clarification` tool removed — agent always searches first

### Response Schema — `ComplianceAnswer`

```python
class Citation(BaseModel):
    doc_title: str    # copied from search results
    section: str      # copied from search results
    clause: str       # copied from search results
    clause_number: str  # e.g. "4.3"
    quote: str        # verbatim from policy text

class Escalation(BaseModel):
    needed: bool      # True only if NO_RELEVANT_POLICY_FOUND
    reason: str

class ComplianceAnswer(BaseModel):
    answer: str                # direct answer pointing to policy
    citations: list[Citation]  # one or more policy sources
    escalation: Escalation     # needed=true only if no policy found
```

The agent's final response must be **pure JSON** matching this schema. The eval system parses it via `eval/agent_wrapper.py:parse_agent_response()`.

---

## Observability — Phoenix

- Docker service on port 6006 (`docker-compose.yml`)
- `rag/observability.py`: call `init_observability()` before any LlamaIndex imports
- Auto-instruments: all LLM calls, tool calls, embeddings via `LlamaIndexInstrumentor`
- Custom spans: `hybrid_search` (vector/BM25 scores), `escalation` events
- Phoenix UI: http://localhost:6006

---

## Evaluation System — Phoenix Datasets + Experiments

### Architecture

```
eval/datasets/*.json          ← Source of truth (from XLSX)
        ↓
scripts/make_dataset.py       ← Uploads to Phoenix as a Dataset
        ↓
Phoenix Dataset               ← Stores examples (input/output/metadata)
        ↓
eval/run_experiment.py        ← Runs task + evaluators against dataset
        ↓
Phoenix Experiment            ← Stores results, metrics, traces
```

### Active Tiers

| Tier | Dataset | Cases | What it tests | LLM? |
|------|---------|-------|---------------|------|
| Tier 1 (`tier1`) | `retrieval-test-v1` | 70 | Correct chunk in top-k by doc+section+clause | No |
| Tier 2 (`tier2`) | `e2e-test-v1` | 25 | Full agent Q&A — citation + answer coverage | Yes |
| Chatbot (`chatbot`) | `chatbot-test-v1` | 61 | Same as Tier 2, realistic user questions | Yes |
| Tier 3 (`escalation`) | — | 6 | Bot correctly escalates | **TODO** |

**Tier 3 is not yet implemented** — dataset exists but evaluators are not written.

### Evaluators — `eval/evaluators.py`

**Retrieval evaluators** (all tiers):
| Evaluator | What it measures |
|---|---|
| `hit_evaluator` | Did ANY result match expected doc+section+clause? (1.0 or 0.0) |
| `mrr_evaluator` | 1/rank of first match (1.0=rank1, 0.5=rank2, ...) |
| `retrieval_doc_hit` | Right document found? (loose) |
| `retrieval_section_hit` | Right doc+section found? |

**Generation evaluators** (Tier 2 + Chatbot):
| Evaluator | What it measures |
|---|---|
| `answer_coverage` | Fraction of expected items found in response (≥50% word overlap) |
| `citation_doc_accuracy` | Did JSON citations reference the correct document? |
| `citation_section_accuracy` | Did JSON citations reference the correct doc+section? |
| `citation_clause_accuracy` | Did JSON citations reference the correct doc+section+clause? (skips if no expected clause) |
| `json_parse_success` | Did agent return valid JSON matching ComplianceAnswer schema? |

**Agent behavior evaluators** (Tier 2 + Chatbot):
| Evaluator | What it measures |
|---|---|
| `agent_search_count` | 0=hallucinated, 1-3=good, 4+=thrashing |
| `agent_used_get_section` | Did agent fetch full section? (0.5 if skipped, not penalized) |

### Parallel Evaluator Structure

| Level | Retrieval layer | Citation layer |
|---|---|---|
| Document only | `retrieval_doc_hit` | `citation_doc_accuracy` |
| Doc + Section | `retrieval_section_hit` | `citation_section_accuracy` |
| Doc + Section + Clause | `hit_evaluator` | `citation_clause_accuracy` |

### Eval Matching Rules

| Expected field | Compared against | Method |
|---|---|---|
| `expected_doc` / `doc_id` | `doc_title` in Qdrant payload | Case-insensitive exact match |
| `expected_section` / `section` | `section` in Qdrant payload | Case-insensitive substring (`in`) |
| `expected_clause` / `clause` | `clause` in Qdrant payload | Case-insensitive substring (`in`) |

**No slugification or normalization.** Test JSON uses display titles as-is.

### Instrumented Agent — `eval/agent_wrapper.py`

All 3 tools are wrapped with logging to capture the agent's actual behavior:
- `search_policies` → logs query + raw hybrid search results
- `get_section` → logs doc_id, section_name, found status
- `escalate_to_compliance` → logs reason

Also includes `parse_agent_response()` — extracts JSON from the agent's final response (handles code fences, embedded JSON, fallback to raw text). Returns `parse_success` flag used by `json_parse_success` evaluator.

**Critical:** Wrapped tool `name=` parameter must match system prompt references.
Fresh agent built per question to avoid state leakage.

### Test Case Formats

**Tier 1** (`retrieval_test.json`):
```json
{"id": "RET-001", "question": "...", "expected_doc_id": "...", "expected_section_contains": "...", "expected_clause": "..."}
```

**Tier 2 / Chatbot** (`e2e_test.json`, `chatbot_test_cases.json`) — **same format**:
```json
{"id": "E2E-001", "question": "...", "expected_answer": ["fact1", "fact2"], "expected_citations": [{"doc_id": "...", "section": "...", "clause": "..."}]}
```
Chatbot IDs use `CB-` prefix, e2e uses `E2E-` prefix.

### Running Evaluations

```bash
# 1. Upload datasets to Phoenix (first time / after changes)
python scripts/make_dataset.py eval/datasets/retrieval_test.json
python scripts/make_dataset.py eval/datasets/e2e_test.json
python scripts/make_dataset.py eval/datasets/chatbot_test_cases.json

# 2. Run experiments
python eval/run_experiment.py --tier tier1 --name baseline-hybrid-v1
python eval/run_experiment.py --tier tier2 --name baseline-e2e-v1
python eval/run_experiment.py --tier chatbot --name baseline-chatbot-v1

# After changes, re-run with new name for comparison
python eval/run_experiment.py --tier tier1 --name reranker-test-v1
```

### XLSX → JSON Conversion

Single converter for all 4 tiers:
```bash
python scripts/convert_eval_xlsx.py eval/eval_dataset_template.xlsx
```

Reads 4 sheets from XLSX:
1. Tier 1 (retrieval): columns `id, question, expected_doc, expected_section, expected_clause, expected_text`
2. Tier 2 (e2e): columns `id, question, expected_answer, expected_doc, expected_section, expected_clause`
3. Tier 3 (escalation): columns `id, question, reason, category, should_escalate`
4. Tier 4 (chatbot): **reads columns by header name** (no id column — auto-generates `CB-001`, `CB-002`, etc.)
   - Headers: `Expected Document, Expected Section, Expected Clause, Policy Rule, User Goal, Question, Expected Answer`

### Baseline Results (as of 2026-03-24)

**Tier 1 — Retrieval:** Hit Rate 81%, MRR 0.67, Doc Hit 95%
**Tier 2 — E2E:** Answer Coverage 88%, Citation Accuracy 96%

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
- [x] `rag/hybrid_search.py` — RRF fusion with full metadata fields
- [x] 3 agent tools (`search_policies`, `get_section`, `escalate`) — `ask_clarification` disabled
- [x] `rag/agent.py` — AgentWorkflow with structured JSON output (ComplianceAnswer schema)
- [x] `rag/observability.py` — Phoenix tracing
- [x] `scripts/ingest_all.py` — 52 docs, ~1514 chunks ingested
- [x] `scripts/test_query.py` — interactive agent testing
- [x] `scripts/convert_eval_xlsx.py` — XLSX → JSON for all 4 tiers (header-based chatbot parser)
- [x] `scripts/make_dataset.py` — JSON → Phoenix Dataset uploader
- [x] `eval/evaluators.py` — shared evaluators (retrieval, citation accuracy, generation, agent behavior)
- [x] `eval/agent_wrapper.py` — instrumented agent for eval
- [x] `eval/run_experiment.py` — Phoenix Experiments runner (Tier 1, 2, Chatbot)
- [x] Notebooks for interactive testing

### Not Yet Implemented
- [ ] `api/` — FastAPI REST endpoints
- [ ] `db/` — SQLAlchemy models (Escalation, Session, Message)
- [ ] `notification/email.py` — SMTP escalation emails
- [ ] `frontend/` — React chat UI
- [ ] `tests/` — pytest suite
- [ ] Tier 3 escalation evaluators

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
| Eval matching uses `section_display` | Match against `section` and `clause` fields (name only, no numbers) |
| Eval matching uses slugification | Simple case-insensitive compare: exact for doc, substring for section/clause |
| Single retrieval call per question | For complex questions, agent must call `search_policies` multiple times |
| Hallucinated citations | Citations must only come from retrieved chunk metadata |
| Large DOCX tables broken | Convert all table cells to key:value text before chunking |
| Ollama timeout on 70B model | Set `request_timeout=180`, use streaming response |
| Qdrant empty results on first run | Always run `ingest_all.py` before starting |
| BM25 index stale after re-ingestion | Pipeline auto-syncs BM25 on ingest (if `BM25_ENABLED=true`) |
| PYTHONPATH not set for scripts | Run all scripts with `PYTHONPATH=. python scripts/...` |
| Agent tool names in eval wrapper | `name=` parameter must match system prompt references exactly |
| nest_asyncio missing for eval | Install `pip install nest_asyncio` — required for Phoenix experiment runner |
| Chatbot XLSX has no ID column | `convert_eval_xlsx.py` reads by header name, auto-generates `CB-NNN` IDs |
