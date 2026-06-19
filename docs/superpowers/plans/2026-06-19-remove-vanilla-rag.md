# Remove Vanilla RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the unused vanilla (non-agentic) RAG pipeline and every branch, flag, setting, and doc that references it, leaving agentic as the sole path.

**Architecture:** The vanilla pipeline lives in 3 dedicated files plus a set of `if settings.pipeline_mode == "vanilla"` branches in the API, Teams bot, and eval runner. Removal proceeds consumers-first (strip the branches that read `pipeline_mode`), then deletes the now-orphaned files and the `pipeline_mode` config field, then cleans docs. Each step leaves the repo importable.

**Tech Stack:** Python 3.12, FastAPI, LlamaIndex AgentWorkflow, pydantic-settings, Phoenix tracing. No pytest suite exists — verification is grep + import smoke-checks + a manual API smoke test.

## Global Constraints

- This is a **dead-code removal** — no behavior change for users; do not refactor the agentic pipeline beyond collapsing the now-pointless mode branches.
- **`init_observability()` must run first in every entry point**, before any LlamaIndex/Ollama import. The API (`api/main.py:6`) and Teams entry point (`scripts/start_teams_bot.py:18`) call it before importing `rag.*`. Because of this, the rag imports inside `api/routes/query.py` and `channels/teams/bot.py` stay **deferred (inside the function)** — do NOT hoist them to module top, even though the project's general rule is imports-at-top.
- Commit message convention: lowercase type prefix (`refactor:`, `docs:`). End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Do not commit until the user confirms** (per CLAUDE.md). Commit steps below are the intended boundaries; at execution, get the user's OK or let them run the commit.
- Work happens on a feature branch, not `main` (see Task 0).

---

### Task 0: Create feature branch

**Files:** none (git only)

- [ ] **Step 1: Branch off main**

```bash
git checkout -b chore/remove-vanilla-rag
```

- [ ] **Step 2: Confirm clean starting point**

Run: `git status --short`
Expected: no output (clean tree).

---

### Task 1: Remove vanilla from API + Teams runtime

Strip the three `pipeline_mode` reads in the live request paths and the `/health` mode field. After this task `settings.pipeline_mode` is read nowhere.

**Files:**
- Modify: `api/routes/query.py` (whole file)
- Modify: `api/main.py:28-34` (`/health` handler)
- Modify: `channels/teams/bot.py:37-60` (`_run_rag`) and `:365` (startup print)

**Interfaces:**
- Consumes: `rag.agent.build_agent()`, `rag.response.parse_agent_response(str) -> dict` (unchanged).
- Produces: no new interfaces. `query()` and `_run_rag()` keep their existing signatures and return the same `ComplianceAnswer` dict.

- [ ] **Step 1: Rewrite `api/routes/query.py`**

Replace the entire file with (note: `from config import settings` is dropped because `settings` is no longer used here; rag imports stay deferred for the observability constraint):

```python
from fastapi import APIRouter, HTTPException

from api.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Receive a compliance question, return structured answer with citations."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Deferred imports: init_observability() (api/main.py) must run before LlamaIndex loads.
    from rag.agent import build_agent
    from rag.response import parse_agent_response

    agent = build_agent()
    response = await agent.run(user_msg=request.question)
    return parse_agent_response(str(response))
```

- [ ] **Step 2: Edit `/health` in `api/main.py`**

Change the handler (lines 28-34) from:

```python
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": settings.pipeline_mode,
        "llm": settings.llm_model,
    }
```

to:

```python
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm": settings.llm_model,
    }
```

(Leave `from config import settings` at the top of `api/main.py` — `settings.llm_model` still uses it.)

- [ ] **Step 3: Simplify `_run_rag` in `channels/teams/bot.py`**

Replace lines 37-60 (the `_run_rag` function) with:

```python
def _run_rag(question: str) -> dict:
    """Run the RAG pipeline directly (no HTTP). Returns ComplianceAnswer dict."""
    try:
        # Deferred imports: init_observability() (start_teams_bot.py) must run before LlamaIndex loads.
        import asyncio
        from rag.agent import build_agent
        from rag.response import parse_agent_response

        async def _run():
            agent = build_agent()
            return await agent.run(user_msg=question)

        response = asyncio.run(_run())
        return parse_agent_response(str(response))
    except Exception as e:
        print(f"RAG pipeline error: {e}")
        return {
            "answer": "",
            "citations": [],
            "escalation": {"needed": True, "reason": str(e)},
        }
```

- [ ] **Step 4: Remove the pipeline-mode startup print in `channels/teams/bot.py`**

Delete this line (~365, inside `run()`):

```python
            print(f"Pipeline: {settings.pipeline_mode}")
```

Leave the surrounding `print(f"LLM: ...")` / `print(f"Polling every ...")` lines untouched. (`settings` is still used elsewhere in `bot.py`, so keep its import.)

- [ ] **Step 5: Verify no residual reads + clean imports**

Run:
```bash
grep -rniI "pipeline_mode" api/ channels/
```
Expected: **no output**.

Run:
```bash
PYTHONPATH=. python -c "import api.main, channels.teams.bot; print('ok')"
```
Expected: prints `ok` (no ImportError, no NameError).

- [ ] **Step 6: Commit**

```bash
git add api/routes/query.py api/main.py channels/teams/bot.py
git commit -m "refactor: drop vanilla pipeline branch from API and Teams runtime

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Remove the `--mode` flag from the eval runner

**Files:**
- Modify: `eval/run_experiment.py:185-186` (argparse), `:203` (mode_prefix), `:225` (print), `:256-265` (vanilla branch)

**Interfaces:**
- Consumes: `make_agent_task(verbose)`, `TIER2_EVALUATORS`, `CHATBOT_EVALUATORS` (unchanged).
- Produces: no new interfaces. The agent task becomes the only tier2/chatbot path.

- [ ] **Step 1: Delete the `--mode` argparse argument**

Remove these two lines (185-186):

```python
    parser.add_argument("--mode", choices=["agentic", "vanilla"], default="agentic",
                        help="Pipeline mode: 'agentic' (ReAct agent) or 'vanilla' (single LLM call)")
```

- [ ] **Step 2: Simplify the auto-generated name prefix**

Replace the `mode_prefix` block (lines 202-210) — change:

```python
    if not args.name:
        mode_prefix = "vanilla" if args.mode == "vanilla" else "agentic"
        embed_short = settings.embedding_model.split("/")[-1]
```

to (drop the `mode_prefix` line and hardcode `agentic` in the two f-strings below it):

```python
    if not args.name:
        embed_short = settings.embedding_model.split("/")[-1]
```

Then in the same block, change the two name templates from `f"{mode_prefix}_{args.tier}_..."` to `f"agentic_{args.tier}_..."`:

```python
        search = "hybrid" if settings.bm25_enabled else "vector"
        if settings.reranker_enabled:
            reranker_short = settings.reranker_model.replace("/", "-")
            args.name = f"agentic_{args.tier}_{embed_short}_{search}_cand{settings.reranker_candidates}_{reranker_short}_top{settings.reranker_top_n}"
        else:
            args.name = f"agentic_{args.tier}_{embed_short}_{search}_top{top_k}"
```

- [ ] **Step 3: Remove the `Mode:` print line**

Delete line 225:

```python
    print(f"  Mode:        {args.mode}")
```

- [ ] **Step 4: Collapse the task-selection branch**

Replace the three-way branch (lines 249-274) so tier1 stays and the `elif args.mode == "vanilla":` block is deleted, leaving the agent task as the `else`. Change:

```python
    if args.tier == "tier1":
        task = make_tier1_task(top_k=top_k)
        evaluators = TIER1_EVALUATORS
        metadata = {**infra_meta, "search_type": search_type, "embedding_model": settings.embedding_model,
                     "reranker": reranker_info, "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "top_k": top_k, "tier": "tier1"}
    elif args.mode == "vanilla":
        from eval.pipeline_wrapper import run_pipeline_task
        task = run_pipeline_task
        evaluators = TIER2_EVALUATORS if args.tier == "tier2" else CHATBOT_EVALUATORS
        metadata = {**infra_meta, "llm": settings.llm_model, "search_type": search_type,
                     "reranker": reranker_info,
                     "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "agent_type": "vanilla", "top_k": top_k, "tier": args.tier,
                     "structured_output": True}
    else:
        task = make_agent_task(verbose=args.verbose)
        evaluators = TIER2_EVALUATORS if args.tier == "tier2" else CHATBOT_EVALUATORS
        metadata = {**infra_meta, "llm": settings.llm_model, "search_type": search_type,
                     "reranker": reranker_info,
                     "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "agent_type": "react", "top_k": top_k, "tier": args.tier,
                     "structured_output": True}
```

to:

```python
    if args.tier == "tier1":
        task = make_tier1_task(top_k=top_k)
        evaluators = TIER1_EVALUATORS
        metadata = {**infra_meta, "search_type": search_type, "embedding_model": settings.embedding_model,
                     "reranker": reranker_info, "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "top_k": top_k, "tier": "tier1"}
    else:
        task = make_agent_task(verbose=args.verbose)
        evaluators = TIER2_EVALUATORS if args.tier == "tier2" else CHATBOT_EVALUATORS
        metadata = {**infra_meta, "llm": settings.llm_model, "search_type": search_type,
                     "reranker": reranker_info,
                     "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "agent_type": "react", "top_k": top_k, "tier": args.tier,
                     "structured_output": True}
```

- [ ] **Step 5: Verify**

Run:
```bash
grep -niI "args.mode\|mode_prefix\|vanilla\|pipeline_wrapper" eval/run_experiment.py
```
Expected: **no output**.

Run:
```bash
PYTHONPATH=. python -c "import eval.run_experiment; print('ok')"
```
Expected: prints `ok`.

Run (argparse no longer accepts `--mode`):
```bash
PYTHONPATH=. python eval/run_experiment.py --tier tier1 --mode vanilla 2>&1 | grep -i "unrecognized\|error"
```
Expected: an argparse error mentioning `--mode` is unrecognized.

- [ ] **Step 6: Commit**

```bash
git add eval/run_experiment.py
git commit -m "refactor: remove --mode/vanilla path from eval runner

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Delete orphaned files + config field + env line

Nothing imports these files or reads `pipeline_mode` after Tasks 1-2.

**Files:**
- Delete: `rag/pipeline.py`
- Delete: `eval/pipeline_wrapper.py`
- Delete: `scripts/test_pipeline.py`
- Modify: `config.py:74-75` (remove `# Pipeline` block)
- Modify: `.env` (remove `PIPELINE_MODE=agentic`)

**Interfaces:** none consumed or produced — pure deletion.

- [ ] **Step 1: Delete the three vanilla files**

```bash
git rm rag/pipeline.py eval/pipeline_wrapper.py scripts/test_pipeline.py
```

- [ ] **Step 2: Remove the `pipeline_mode` field from `config.py`**

Delete these lines (74-75):

```python
    # Pipeline
    pipeline_mode: str = "agentic"  # "agentic" or "vanilla"
```

Leave the `# Agent` block (`agent_max_iterations`, `agent_timeout`) that follows it intact.

- [ ] **Step 3: Remove the env line from `.env`**

Delete the line `PIPELINE_MODE=agentic` (currently line 58 of `.env`). `.env.example` has no such line — leave it alone.

- [ ] **Step 4: Repo-wide verification — no vanilla references remain in code**

Run:
```bash
grep -rniI "pipeline_mode\|run_pipeline_task\|rag\.pipeline\|test_pipeline\|pipeline_wrapper" --include=*.py .
```
Expected: **no output**.

Run (settings load cleanly despite any stray deployed-env line):
```bash
PYTHONPATH=. python -c "from config import settings; print(settings.llm_model)"
```
Expected: prints the model name, no AttributeError / ValidationError.

Run:
```bash
PYTHONPATH=. python -c "import api.main, channels.teams.bot, eval.run_experiment; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete vanilla pipeline files and pipeline_mode setting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Update SETUP.md

CLAUDE.md and `.env.example` have **no** vanilla references (verified) — only `SETUP.md` needs edits.

**Files:**
- Modify: `SETUP.md` (lines 45, 168-184, 192-193, 318-325, 498-499, 516)

**Interfaces:** none.

- [ ] **Step 1: Remove the `PIPELINE_MODE` config-table row (line 45)**

Delete:
```markdown
| `PIPELINE_MODE` | `agentic` (default) or `vanilla` |
```

- [ ] **Step 2: Collapse the "Test a query" section (lines 168-184)**

Replace:
```markdown
### Agentic agent (default)

```bash
PYTHONPATH=. python scripts/test_query.py -q "What is the policy on software installation?"
```

Returns `ComplianceAnswer` JSON with `answer`, `citations[]` (with `source_number`, `doc_title`, `section`, `clause`, `clause_number`, `quote`), and `escalation`.

### Vanilla pipeline (no LlamaIndex, single LLM call)

```bash
PYTHONPATH=. python scripts/test_pipeline.py -q "What is the policy on software installation?"
```

Faster (~30-40s vs ~50-90s for agentic), but no multi-search reasoning.
```

with:
```markdown
### Test a query

```bash
PYTHONPATH=. python scripts/test_query.py -q "What is the policy on software installation?"
```

Returns `ComplianceAnswer` JSON with `answer`, `citations[]` (with `source_number`, `doc_title`, `section`, `clause`, `clause_number`, `quote`), and `escalation`.
```

- [ ] **Step 3: Remove the Vanilla trace bullet (lines 192-193)**

Delete:
```markdown
- **Vanilla**: manual spans (`vanilla_rag_pipeline` → `search_policies` → `llm_call`)
```
Keep the `- **Agentic**: full ReAct trace …` bullet above it.

- [ ] **Step 4: Fix the eval examples (lines 318-325)**

Replace:
```markdown
# Tier 2 — full agent e2e
python eval/run_experiment.py --tier tier2 --mode agentic --name agentic-baseline

# Chatbot — realistic user questions; vanilla mode
python eval/run_experiment.py --tier chatbot --mode vanilla --name vanilla-baseline
```

`--mode` defaults to `agentic`. Auto-generated experiment names include backend + reranker config. Metadata captures infra (`local`/`remote`) and URLs.
```

with:
```markdown
# Tier 2 — full agent e2e
python eval/run_experiment.py --tier tier2 --name agentic-baseline

# Chatbot — realistic user questions
python eval/run_experiment.py --tier chatbot --name chatbot-baseline
```

Auto-generated experiment names include backend + reranker config. Metadata captures infra (`local`/`remote`) and URLs.
```

- [ ] **Step 5: Fix the config-reference "Pipeline" section (lines 498-499)**

Replace:
```markdown
### Pipeline
`PIPELINE_MODE`, `AGENT_MAX_ITERATIONS`, `AGENT_TIMEOUT`
```
with:
```markdown
### Agent
`AGENT_MAX_ITERATIONS`, `AGENT_TIMEOUT`
```

- [ ] **Step 6: Remove the Vanilla RAG status bullet (line 516)**

Delete:
```markdown
- Vanilla RAG (no LlamaIndex, manually traced)
```

- [ ] **Step 7: Verify docs are clean**

Run:
```bash
grep -niI "vanilla\|pipeline_mode\|PIPELINE_MODE\|test_pipeline" SETUP.md
```
Expected: **no output**.

- [ ] **Step 8: Commit**

```bash
git add SETUP.md
git commit -m "docs: remove vanilla pipeline references from SETUP.md

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Full verification + deployment note

**Files:** none (verification only)

- [ ] **Step 1: Repo-wide grep is clean**

Run:
```bash
grep -rniI "pipeline_mode\|vanilla\|run_pipeline_task\|rag\.pipeline\|test_pipeline\|pipeline_wrapper" --include=*.py --include=*.md . | grep -v "docs/superpowers/"
```
Expected: **no output** (the spec/plan under `docs/superpowers/` legitimately contain the word "vanilla" and are excluded).

- [ ] **Step 2: All entry points import**

Run:
```bash
PYTHONPATH=. python -c "import api.main, channels.teams.bot, eval.run_experiment; from config import settings; print('ok', settings.llm_model)"
```
Expected: prints `ok` and the model name.

- [ ] **Step 3: API smoke test**

Start the API (`./scripts/start_api.sh`), then in another shell:
```bash
curl -s localhost:8000/health
```
Expected: `{"status":"ok","llm":"..."}` — note **no** `mode` field.

```bash
curl -s -X POST localhost:8000/api/query -H 'Content-Type: application/json' -d '{"question":"What is the policy on software installation?"}'
```
Expected: a `ComplianceAnswer` JSON with `answer`, `citations`, `escalation` (requires Qdrant/Ollama reachable; skip if infra is unavailable and note it).

- [ ] **Step 4: Flag the deployed `.env`**

The Spark box's deployed `.env` likely still contains `PIPELINE_MODE=agentic`. It is harmless (the field no longer exists, and pydantic-settings ignores unknown env keys), but tell the user to remove that one line so the deployed config matches the repo. **This is a manual step on the remote host — not part of this branch.**

---

## Notes for the executor

- If any verification step fails, stop and surface the output — do not paper over it.
- Commits are batched per task; honor the user's "commit only when asked" rule — confirm before running the `git commit` steps, or hand the staged changes to the user.
- After all tasks: the branch `chore/remove-vanilla-rag` is ready for the finishing-a-development-branch skill (PR or merge).
