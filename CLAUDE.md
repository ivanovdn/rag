# Compliance Q&A Bot — Claude Code Instructions

Internal Compliance Q&A bot using **Agentic RAG** (LlamaIndex `AgentWorkflow`, multi-tool). Answers employee questions **strictly from approved internal policy DOCX files**; if an answer can't be grounded in policy, it escalates to Compliance with full context.

**Channels:** Microsoft Teams only (polls Graph `/me/chats` every 5s, imports RAG directly — no HTTP). The bot is the sole entry point; there is no HTTP API.
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
  router.py          # pre-retrieval classify_message() + resolve() (greeting/in_scope/out_of_scope/unintelligible)
  resilience.py      # is_transient() + retry_transient() — classify/retry transient backend failures
  observability.py   # Phoenix init + tracer + record_infra_unavailable() + record_classification()
  tools/             # search_policies (call FIRST), get_section, escalate_to_compliance
                     #   (clarify.py exists but is NOT imported/used)
channels/teams/      # bot.py (poll+RAG+feedback), auth.py, renderer.py, feedback.py, utils.py
eval/                # evaluators.py, agent_wrapper.py, run_experiment.py
scripts/             # ingest_all, test_query, run_eval, make_dataset, start_*.sh
tests/               # unit/ (pure-logic) + docs/ (corpus parsing) + live/ (live-LLM accuracy); docs+live auto-skip; see SETUP.md Testing
# stubs / not implemented: notification/ db/ frontend/ (empty React scaffold), notebooks/ (gitignored)
```

**Search flow:** `embed_query → vector_search (RERANKER_CANDIDATES) → [BM25 RRF] → [rerank → top RERANKER_TOP_N] → format_sources()` with `[Source N]` headers. The 3 agent tools: `search_policies` (search+rerank+format, always first), `get_section` (full section by doc_id+section_name), `escalate_to_compliance`.

**Input classification (router):** before retrieval, `_send_reply` (when `ROUTER_ENABLED`) runs one temperature-0 LLM call (`rag/router.py` `classify_message`) tagging the message `greeting | in_scope | out_of_scope | unintelligible`. Only `in_scope` reaches the RAG pipeline; greeting→`WELCOME_HTML`, out_of_scope→`render_out_of_scope()`, unintelligible→`render_unintelligible()` (no search, no rating). **Safe-default invariant:** confidence `< ROUTER_CONFIDENCE_FLOOR`, or ANY classifier failure/unparseable output → `in_scope` (it can never refuse a real question). Logged to Phoenix via `record_classification` (`router_category/confidence/fallback/message` — full message recorded for audit). Editable tuning surface: `ROUTER_SYSTEM_PROMPT` (prompt+categories) in `rag/router.py`, the two messages in `renderer.py`.

**Infra resilience:** transient backend failures (conn errors, timeouts, 5xx from embeddings/Qdrant/LLM) are retried (`retry_transient`, backoffs `(0.5,1,2)s` → up to 4 attempts) and, if still failing, become a clean **"service temporarily unavailable"** reply — never a content escalation, never a leaked raw error. Two interception points: retrieval (inside `search_policies` → sets `sp._retrieval_unavailable`, returns sentinel `POLICY_SEARCH_UNAVAILABLE`) and the LLM/agent boundary (in `_run_rag`, returns `{"status":"unavailable"}`). Each records a distinct `infra_unavailable` Phoenix span (`failed_component`, `error_type`, `retries_attempted`). The unavailable reply gets no rating prompt and creates no feedback row. Non-transient errors still propagate to escalation as before.

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
ROUTER_ENABLED (kill switch) / ROUTER_LLM_MODEL (blank=main LLM) / ROUTER_CONFIDENCE_FLOOR (0.6)
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
| Tool raises an exception (e.g. Qdrant down) but `_run_rag`'s try/except never sees it | `AgentWorkflow` **swallows tool exceptions** and feeds them to the LLM as an observation. Detect infra failure *inside* the tool (`sp._retrieval_unavailable` flag), not by catching at `agent.run()`. |
| LLM-down not classified as transient | A dead Ollama LLM raises **builtins `ConnectionError`/`TimeoutError`** (the `ollama` client re-wraps httpx), NOT an httpx type. `is_transient` must catch these + `ollama.ResponseError`/`openai` 5xx — see `_TRANSIENT_TYPES`. Verify infra changes against the LLM path, not just retrieval. |
| Bot log empty / "not starting" when output redirected to a file | Python block-buffers stdout when not a TTY. Start with `python -u` (unbuffered) — the bot was alive and polling, just not flushing. |
| Stale `bot_state.json` floods the channel with backlog | Bot trusts `last_check` with no upper bound; a weeks-old file re-answers the whole backlog. When moving hosts, reset local state to now (back up + delete `bot_state.json`). Startup clamp is a pending fix. |

**Not yet implemented:** email escalation. (Tier-A pytest suite exists under `tests/`; Tier-B/C and CI still pending.)
