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

#### `TEAMS_CLIENT_SECRET` — it expires, and nothing warns you

It comes from the Azure app registration: **App registrations → the app → Certificates
& secrets → Client secrets → New client secret**. Copy the **Value** column the moment
it appears — it is rendered once, and the **Secret ID** beside it is a different thing.
`TEAMS_CLIENT_ID` and `TEAMS_TENANT_ID` are on the app's Overview page.

The dialog defaults to a **6-month** lifetime. Prefer 24 months: it is the same amount
of work and four times the runway.

A secret's expiry is knowable the day it is created, but it lives only in the portal —
`.env` holds the value and nothing about its lifetime, and the app cannot read its own
registration without `Application.Read.All`, which it should not have for this. So
**write the date in `.env.example` next to the key** when you rotate. That comment is
the only place in this repo the date can live.

When it does expire, the bot logs `AADSTS7000222 is terminal — retrying will not fix
it` once, then keeps polling and failing with a 401 per cycle. Create a new secret, put
it in `.env`, and recreate the container — `docker compose -f docker-compose-remote.yml
up -d`, **not** `restart`, which reuses the environment the container was created with
and would come back on the dead secret.

#### `TEAMS_REFRESH_TOKEN`

The refresh token is obtained via the device-code flow: `PYTHONPATH=. python scripts/get_refresh_token.py` (run from the repo root; sign in with the bot's Teams account when prompted). The bot then rotates and persists the token to `channels/teams/data/refresh_token.json` on every use.

This same command is also the **recovery** procedure if `channels/teams/data/refresh_token.json` is ever lost or corrupted — not just a one-time setup step. Azure invalidates a refresh token as soon as it is used, so the `TEAMS_REFRESH_TOKEN` seed in `.env` is superseded after the bot's very first refresh; restoring an old copy of the token file does not work either, since that copy has already been rotated past too. Re-running the script and signing in again is the only way back.

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

Local invocation — valid for `tier1` only; see the next section before running
`tier2` or `chatbot`.

```bash
# Tier 1 — retrieval only (fast, no LLM)
python eval/run_experiment.py --tier tier1 --name baseline-retrieval

# Tier 2 — full agent e2e
python eval/run_experiment.py --tier tier2 --name agentic-baseline

# Chatbot — realistic user questions
python eval/run_experiment.py --tier chatbot --name chatbot-baseline
```

Auto-generated experiment names include backend + reranker config. Metadata captures infra (`local`/`remote`) and URLs.

### Where to run it: the VM, not your laptop

**A `tier2` or `chatbot` run from the Mac is silently invalid.** The dev `.env`
points `RERANKER_URL` at `localhost:8081`; an unreachable reranker falls back to
the original ranking with only a log warning, so the run completes and reports
numbers that were never reranked. `tier1` is safe locally with
`USE_REMOTE_QDRANT=true` — anything touching the LLM or reranker is not.

On `srv-agent-01` (user `sa.ivanov`) the project is **one repo with two linked
git worktrees**, not two clones:

| path | role |
|---|---|
| `~/rag` | the deployed checkout, always on `main`. `rag-bot-1` and `rag-phoenix-1` run from its compose project. |
| `~/rag-eval` | the experiment workbench — check out the branch under test here. |

Two consequences of them being worktrees, both of which have cost time:

- `~/rag-eval` **cannot hold `main`** while `~/rag` does (`fatal: 'main' is
  already used by worktree at '/home/sa.ivanov/rag'`). Check out the branch
  under test, or a detached HEAD. Use `git worktree move`, never `mv`.
- `~/rag-eval` **has no `.env`** — worktrees do not share gitignored files.
  That is why eval runs from `~/rag`, not from the worktree it measures.

> ⚠️ `git clean -fdx` in `~/rag` deletes `channels/teams/data/refresh_token.json`,
> the live rotating Azure credential. Azure invalidates superseded tokens, so a
> backup is not a restore path. It is the only unrecoverable command here.

### Running an experiment on the VM

Eval runs as a throwaway container from `~/rag`, whose compose supplies the
production config, with the other worktree's source mounted over the baked image:

```bash
cd ~/rag
EVAL="-v /home/sa.ivanov/rag-eval/config.py:/app/config.py \
      -v /home/sa.ivanov/rag-eval/rag:/app/rag \
      -v /home/sa.ivanov/rag-eval/eval:/app/eval \
      -v /home/sa.ivanov/rag-eval/scripts:/app/scripts"

# 1. Assert you are running the code you think you are — BEFORE spending GPU.
docker compose -f docker-compose-remote.yml run --rm $EVAL --entrypoint python bot \
  -c "import rag.vector_store as vs; from rag.agent import ALL_TOOLS; print('tools:', ALL_TOOLS, '| RRF_K:', vs.RRF_K)"

# 2. Upload the dataset (once per dataset change).
docker compose -f docker-compose-remote.yml run --rm $EVAL --entrypoint python bot \
  scripts/make_dataset.py eval/datasets/chatbot_test_cases.json --phoenix-url http://phoenix:6006

# 3. Run.
docker compose -f docker-compose-remote.yml run --rm $EVAL \
  -e GIT_COMMIT=$(git -C /home/sa.ivanov/rag-eval describe --always --dirty) \
  -e QDRANT_COLLECTION=compliance_policies_v2 -e BM25_ENABLED=true \
  --entrypoint python bot eval/run_experiment.py \
  --tier chatbot --name <run-name> --phoenix-url http://phoenix:6006
```

**Pass `-e GIT_COMMIT` on every run.** The container has no `.git`, so the
experiment's `git_commit` comes from that variable — and without the override it
is the image's, i.e. the *deployed* commit, not the worktree being measured.
Nothing can detect that from inside. The prompt hashes beside it
(`system_prompt_sha12`, `agent_input_sha12`) are content hashes and are right
either way; every production `compliance_request` span carries the same values
as `identity.*`, so an experiment and a trace match by string comparison.

`--phoenix-url` takes the compose network name `phoenix` — the same instance you
reach at `172.20.1.10:6006` from outside. The `-e` overrides let one image
measure several configurations without an edit or a rebuild, which is how a
control run isolates one variable. `--no-trace` turns tracing off for a run.

To sweep the candidate count, either `-e RERANKER_CANDIDATES=40` or the
`--candidates 40` flag — they reach the same setting, so both apply to every
tier. It is the only candidate knob there is: it sizes the dense prefetch, the
sparse prefetch, and the fused limit together, so raising it actually widens the
pool rather than leaving it pinned by a per-branch cap.

**Step 1 is not optional.** The image bakes `config.py` and `rag/` at build time,
so mounting only `eval/` gives you the new harness running main's pipeline — a
green run that measured the wrong code. Mount all four, and assert something
version-specific before the GPU bill starts.

Traces land under **Datasets & Experiments → the dataset → the experiment → a
row's trace link**, never on the Projects page: Phoenix files task spans under
its own per-experiment project. See the gotcha table in `CLAUDE.md`.

---

## Remote Stack (Spark @ 172.20.0.22)

For the production deployment, models run on Spark:
- Ollama (LLM + embedding) on port 11434
- Qdrant on port 6333
- vLLM (reranker) on port 8267

Switch via `.env`:

```
USE_REMOTE_OLLAMA=true
USE_REMOTE_QDRANT=true
EMBEDDING_SOURCE=ollama
OLLAMA_EMBEDDING_URL=http://172.20.0.22:11434
RERANKER_BACKEND=vllm-score
RERANKER_URL=http://172.20.0.22:8267
RERANKER_MODEL=Qwen/Qwen3-Reranker-4B
```

Re-ingest if embedding dimensions change. Verify connectivity to Spark before starting bot:

```bash
curl http://172.20.0.22:6333/collections
curl http://172.20.0.22:11434/api/tags
curl http://172.20.0.22:8267/v1/models
```

---

## Docker Deployment (Remote Bot Host)

For hosting the Teams bot on a separate machine that connects to Spark:

```bash
# On the bot host (Linux):
git clone <repo> compliance-bot && cd compliance-bot

# Copy a .env configured for remote stack (TEAMS_* + USE_REMOTE_*=true)
scp local:/path/to/.env .env

# Build & run. The code is baked into the image, so `git pull` alone deploys
# nothing -- always --build. GIT_COMMIT puts the commit in the startup banner.
GIT_COMMIT=$(git describe --always --dirty) docker compose -f docker-compose-remote.yml up -d --build
```

The first two banner lines say what is actually running: `Build: <commit>` — the
image's commit, which is not the checkout's until you rebuild — and
`LLM: <model> @ <digest>`. Set `LLM_MODEL_DIGEST` in `.env` to the digest you
have tested against and the banner warns when the tag on the model host has
moved; production's is `07d35212591f` (`qwen3.6:latest`, 2026-10-08). The check
runs at startup only, so a re-pull while the bot is running shows up at the
next restart.

The compose file runs:
- **bot** container (Python 3.12-slim) — code only, polls Graph API outbound
- **phoenix** container — observability, healthchecked

Volumes:
- **`./channels/teams/data` → `/app/channels/teams/data`** (bind mount — feedback DB, refresh token, bot state appear directly in the project folder so you can read them in your IDE)
- `phoenix_data` (named) → `/data` (traces)

`.env` is mounted via `env_file:`. Phoenix endpoint is overridden to `http://phoenix:6006/v1/traces` inside the compose network.

Logs: `docker compose -f docker-compose-remote.yml logs -f bot`

> **Ingestion is a one-time admin task** done before deploying. The bot image does not include `policies/` or `ingest/`.

### Single-poller handoff (local ↔ remote)

**Only one bot instance may run per Teams account.** Both poll the same Graph mailbox and share one **single-use refresh-token chain** — running two at once causes duplicate replies and corrupts the token chain.

When moving the bot between hosts (local testing → remote, or back), do this in order:

```bash
# 1. Stop the currently-running bot (frees the single-poller slot + lets it flush
#    the latest rotated token to disk).
#    Local:  Ctrl-C  (or  kill -INT <pid>)
#    Docker: docker compose -f docker-compose-remote.yml stop bot

# 2. Copy BOTH state files from the OLD host to the NEW host's channels/teams/data/:
#    - refresh_token.json : the *current* rotated token. The old host advanced the
#      chain while running, so the new host's stored token is stale and will fail auth.
#    - bot_state.json     : last_check + processed_messages, so the new host resumes
#      exactly where the old one stopped and re-answers NOTHING.
scp <old-host>:<repo>/channels/teams/data/refresh_token.json ./channels/teams/data/
scp <old-host>:<repo>/channels/teams/data/bot_state.json     ./channels/teams/data/

# 3. Start the bot on the new host.
```

**Why copy `bot_state.json`?** Each host keeps its own state. Start the new host with a stale or missing `bot_state.json` and it treats recent channel messages as new and re-answers them. The old host's file has `last_check ≈ now` and a `processed_messages` set listing what was already answered, so the handoff is silent.

**Backstop if you don't copy state:** `_load_state` clamps a stale `last_check` on startup — if it is older than `TEAMS_MAX_STATE_AGE_MINUTES` (default 60) it resets to `now − TEAMS_INITIAL_LOOKBACK_MINUTES` (default 5 min). That caps the worst case at re-answering the **last 5 minutes** of messages instead of the whole backlog. To re-answer *nothing* without copying state: either wait >5 min (no new messages) before starting, or set `TEAMS_INITIAL_LOOKBACK_MINUTES=0` for the handoff (answers only messages that arrive after startup).

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
`LLM_BACKEND`, `LLM_MODEL`, `OPENAI_MODEL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `LLM_TEMPERATURE`, `USE_REMOTE_OLLAMA`, `OLLAMA_BASE_URL`, `OLLAMA_REMOTE_URL`, `LLM_REQUEST_TIMEOUT`, `LLM_REMOTE_REQUEST_TIMEOUT`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_CTX`

### Embeddings
`EMBEDDING_SOURCE`, `EMBEDDING_MODEL`, `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_PASSAGE_PREFIX`, `OLLAMA_EMBEDDING_URL`, `HF_TOKEN`

### Vector Store
`USE_REMOTE_QDRANT`, `QDRANT_URL`, `QDRANT_REMOTE_URL`, `QDRANT_COLLECTION`, `QDRANT_VECTOR_DIM`

### Search & Reranker
`MIN_CONFIDENCE_SCORE`, `BM25_ENABLED`, `RERANKER_ENABLED`, `RERANKER_BACKEND`, `RERANKER_URL`, `RERANKER_MODEL`, `RERANKER_TOP_N`, `RERANKER_CANDIDATES`, `RERANKER_INSTRUCTION`, `RERANKER_QUERY_TEMPLATE`

### Agent
`AGENT_TIMEOUT`

### Teams Bot
`TEAMS_TENANT_ID`, `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET`, `TEAMS_REFRESH_TOKEN`, `TEAMS_POLL_INTERVAL`

### Observability
`PHOENIX_ENABLED`, `PHOENIX_ENDPOINT`, `PHOENIX_PROJECT_NAME`

---

## Testing

Tier-A unit tests (pure logic, no services), an offline poll-loop soak under
`tests/load/`, plus an auto-skipping corpus layer.

`tests/load/` drives the real `process_new_messages` against a fake Graph (30
chats, paging, injected timeouts and cyclic `@odata.nextLink`) on a simulated
clock, so a 65-minute watermark hold runs in milliseconds. It never touches
Microsoft Graph, the Spark box, the real `bot_state.json` or the refresh token
— a fixture makes `requests`/`httpx` raise, so that isolation is proven rather
than assumed. It runs in the default suite (~0.3s) and asserts the invariants
that matter: no message answered twice, none lost, `last_check` monotonic,
exactly one worker thread.

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
