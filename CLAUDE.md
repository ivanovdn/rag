# Compliance Q&A Bot — Claude Code Instructions

Internal Compliance Q&A bot using **Agentic RAG** (LlamaIndex `AgentWorkflow`, multi-tool). Answers employee questions **strictly from approved internal policy DOCX files**; if an answer can't be grounded in policy, it escalates to Compliance with full context.

**Channels:** Microsoft Teams (polls Graph `/me/chats` every 5s, imports RAG directly — no HTTP) + FastAPI `/api/query`.
**Deployment:** runs in Docker on a remote **Linux** host (`docker-compose-remote.yml`). All models + Qdrant live on an **NVIDIA Spark** box (`192.168.100.2`); the bot connects out to them. Local dev (everything on localhost) is still supported via env toggles.

## Commands

```bash
# Ingest policies (parse → embed → upsert to Qdrant)
PYTHONPATH=. python scripts/ingest_all.py --folder ./policies
# Test a query
PYTHONPATH=. python scripts/test_query.py
# Eval (upload dataset first, then run)
python scripts/make_dataset.py eval/datasets/<file>.json
python eval/run_experiment.py --tier chatbot --name baseline-v1
# Serve
./scripts/start_api.sh                            # FastAPI :8000  (GET /health, POST /api/query)
PYTHONPATH=. python scripts/start_teams_bot.py    # Teams bot
docker compose up -d                              # local dev: Qdrant :6333 + Phoenix :6006
docker compose -f docker-compose-remote.yml up -d --build   # PRODUCTION: bot + Phoenix on Linux host
```

## Architecture

```
config.py            # pydantic-settings — ALL toggles & URLs (defaults are lighter dev fallbacks; see Config)
ingest/
  numbering.py       # NumberingResolver — simulates Word numbering.xml counters
  docx_parser.py     # structure-aware chunker (headings + ilvl), NOT token-count splitting
  chunk_models.py    # PolicyChunk: section/clause hold NAMES only; numbers + section_display separate
  pipeline.py        # parse → embed → upsert
rag/
  embeddings.py      # HuggingFace OR Ollama (EMBEDDING_SOURCE)
  vector_store.py    # Qdrant client       bm25_index.py   # pure-Python BM25
  hybrid_search.py   # RRF fusion (k=60)   reranker.py     # /v1/rerank (llama-server OR vllm)
  agent.py           # AgentWorkflow + system prompt + ComplianceAnswer schema + get_llm()
  response.py        # parse_agent_response() — JSON parser for agent output
  observability.py   # Phoenix init + tracer
  tools/             # search_policies (call FIRST), get_section, escalate_to_compliance
                     #   (clarify.py exists but is NOT imported/used)
api/                 # main.py (CORS + /health, inits Phoenix), routes/query.py, models.py
channels/teams/      # bot.py (poll+RAG+feedback), auth.py, renderer.py, feedback.py, utils.py
eval/                # evaluators.py, agent_wrapper.py, run_experiment.py
scripts/             # ingest_all, test_query, run_eval, make_dataset, start_*.sh
tests/               # unit/ (pure-logic) + docs/ (corpus parsing, auto-skip); see SETUP.md Testing
# stubs / not implemented: notification/ db/ frontend/ (empty React scaffold), notebooks/ (gitignored)
```

**Search flow:** `embed_query → vector_search (RERANKER_CANDIDATES) → [BM25 RRF] → [rerank → top RERANKER_TOP_N] → format_sources()` with `[Source N]` headers. The 3 agent tools: `search_policies` (search+rerank+format, always first), `get_section` (full section by doc_id+section_name), `escalate_to_compliance`.

**Eval tiers:** tier1 `retrieval-test-v1` (retrieval hit), tier2 `e2e-test-v1` (Q&A), chatbot `chatbot-test-v1` (realistic). Metadata captures infra (local/remote, urls, reranker backend).

## Config — `.env` (full list in `.env.example`)

**Current production profile (remote Spark):**
- **LLM:** `qwen3.6:35b` via **Ollama** (`LLM_BACKEND=ollama`, `USE_REMOTE_OLLAMA=true`)
- **Embedding:** `embeddinggemma` 768-dim via **Ollama** (`EMBEDDING_SOURCE=ollama`, `QDRANT_VECTOR_DIM=768`)
- **Reranker:** Qwen3-Reranker-4B via **vLLM**, enabled (`RERANKER_BACKEND=vllm`, `RERANKER_ENABLED=true`)
- **Qdrant:** remote (`USE_REMOTE_QDRANT=true`)  •  **BM25:** off
- `config.py` defaults are lighter dev fallbacks — the deployed `.env` is the source of truth.

```bash
LLM_BACKEND=ollama|openai-compatible      USE_REMOTE_OLLAMA / USE_REMOTE_QDRANT
LLM_MODEL=... (ollama) / OPENAI_MODEL=... (openai-compat)
EMBEDDING_SOURCE=huggingface|ollama       EMBEDDING_MODEL / QDRANT_VECTOR_DIM must match (768 gemma / 2048 nemotron / 4096 qwen3)
EMBEDDING_QUERY_PREFIX / EMBEDDING_PASSAGE_PREFIX   RERANKER_BACKEND=llama-server|vllm
BM25_ENABLED (off)                        PHOENIX_ENDPOINT (Docker: http://phoenix:6006/v1/traces)
TEAMS_TENANT_ID / CLIENT_ID / CLIENT_SECRET / REFRESH_TOKEN
```

## Critical Constraints — never violate

- `temperature=0.0` always (deterministic compliance answers).
- Agent must never answer without citing a retrieved chunk; citations come ONLY from retrieved chunk metadata (no hallucination).
- If `search_policies` returns `NO_RELEVANT_POLICY_FOUND` → escalate.
- `init_observability()` must run FIRST in every entry point (before any LlamaIndex/Ollama import).

## Code Style

- Follow best coding practices; match the conventions of the surrounding code.
- **Imports always at the top of the module** — never inside functions.

## Gotchas / Lessons Learned

| Pitfall | Fix |
|---|---|
| `OpenAILike` agent emits ReAct text instead of tool calls | Set `is_function_calling_model=True`. Do NOT set `response_format=json_object` (kills tool-call emission). |
| Word numbering jumps between sections | `NumberingResolver` continues the level-0 counter across multiple `numId`s. |
| `from rag.tools.search_policies import _last_search_results` returns stale data | Use `import rag.tools.search_policies as sp; sp._last_search_results`. |
| Eval matches the wrong field | Eval matches `section`/`clause` (names), NOT `section_display`. |
| `\n` in `.env` sent literally to reranker/embedding | Code converts `\\n`→`\n` in `RERANKER_QUERY_TEMPLATE` and `EMBEDDING_QUERY_PREFIX`. |
| Reranker scores compressed / model rejected | vLLM needs full Qwen3 chat template + model name == `--served-model-name`; llama-server uses simple template, any non-empty name. Falls back to original order on error — never blocks. |
| Embedding dim mismatch on re-ingest | Delete the Qdrant collection before re-ingesting with a different embedding model. |
| Teams rendering broken / duplicated | Limited HTML only: `<p> <b> <i> <ul>/<li> <hr>` — no `<div style>`. Don't repeat answer text in citations. |
| Teams rating mis-detected | Rating = `message.strip() in {"-1","0","1","2"}` exactly; anything else → new question, drop pending state. |
| Teams refresh token not loading | Bot prefers `channels/teams/data/refresh_token.json` (rotated) over `.env` (seed). |
| Multi-citation eval flags valid answers | Add alt `expected_citations` + `"match_mode":"any"` at the TOP level of the test-case JSON. |
| Eval task fails under Phoenix | Experiment task fns must be sync `def`, not `async` — Phoenix `run_experiment` is synchronous. |
| Bot can't reach Phoenix in Docker | Override `PHOENIX_ENDPOINT=http://phoenix:6006/v1/traces` in compose. |

**Not yet implemented:** email escalation. (Tier-A pytest suite exists under `tests/`; Tier-B/C and CI still pending.)
