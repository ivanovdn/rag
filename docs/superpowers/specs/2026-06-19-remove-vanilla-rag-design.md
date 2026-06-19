# Remove vanilla RAG entirely

**Date:** 2026-06-19
**Status:** Approved (design), pending implementation
**Author:** Dmytro Ivanov (with Claude Code)

## Problem

The project carries two answer pipelines: the **agentic** pipeline (LlamaIndex
`AgentWorkflow` + tools, used in production) and a **vanilla** non-agentic
pipeline (single search + single LLM call). Vanilla is no longer used:

- `pipeline_mode` defaults to `agentic`, and the deployed profile is agentic.
- The bot and API never run vanilla in production.
- No vanilla e2e/chatbot baseline was ever saved under `eval/results/`,
  suggesting it was never actually used as a benchmark.

Keeping it is a silent maintenance tax: every change to the answer schema,
tracing, or source formatting must be mirrored in two pipelines, and dead code
drifts out of sync.

## Goal

Delete the vanilla pipeline and **every** branch, flag, setting, and doc that
references it. Agentic becomes the *only* mode — not a default, the sole path.
End state: the `pipeline_mode` concept does not exist anywhere in the codebase.

Non-goals: no behavior change for end users; no refactor of the agentic
pipeline beyond collapsing the now-pointless mode branches.

## Changes

### 1. Delete files (3)
- `rag/pipeline.py` — the vanilla pipeline (`run_query`, `_call_ollama`, `_parse_response`, `_fallback`)
- `eval/pipeline_wrapper.py` — eval wrapper (`run_pipeline_task`)
- `scripts/test_pipeline.py` — interactive vanilla tester

### 2. Collapse runtime branches to agentic-only
- **`api/routes/query.py`** — remove the `if settings.pipeline_mode == "vanilla"`
  branch. Call the agentic path unconditionally; inline `_run_agentic` into the
  handler (the indirection no longer earns its keep).
- **`channels/teams/bot.py`** — in `_run_rag()`, drop the vanilla branch and keep
  the agentic body. Remove the startup line `print(f"Pipeline: {settings.pipeline_mode}")` (~line 365).
- **`api/main.py`** — drop `"mode": settings.pipeline_mode` from the `/health`
  response (keep `status` and `llm`).

### 3. Eval — remove the `--mode` flag
- **`eval/run_experiment.py`**:
  - Delete the `--mode` argparse argument.
  - Remove the `mode_prefix` logic — the experiment name is always `agentic_…`.
  - Delete the `elif args.mode == "vanilla":` block (the `run_pipeline_task` path).
  - Remove the `Mode:` print line.
  - The former `else` (agent task) branch becomes the only tier2/chatbot path.

### 4. Config
- **`config.py`** — delete the `pipeline_mode` field (the `# Pipeline` block, ~lines 74-75).
- **`.env`** — remove line `PIPELINE_MODE=agentic`. (`.env.example` has no such
  line — nothing to change there.)
- `model_config` does not set `extra="forbid"`, so a leftover env line is
  harmless — but we remove it anyway to keep config honest.

### 5. Docs
- **`SETUP.md`** — remove/adjust the vanilla references: the `PIPELINE_MODE` table
  row (~45), the "Vanilla pipeline" section and its tracing note (~176-193), the
  vanilla eval example (~321-322), and the status/config mentions (~499, ~516).
  Drop the `test_pipeline.py` mention.
- **`CLAUDE.md`** — scan for vanilla / `pipeline_mode` / "both pipeline modes"
  references and update so it reads agentic-only.

## Verification

Before claiming done:

1. `grep -rniI "pipeline_mode\|vanilla\|run_pipeline_task\|rag\.pipeline" --include=*.py .` → **zero hits**.
2. `python -c "from config import settings; print(settings.llm_model)"` → loads clean (no stray-env crash).
3. `python -c "import api.main, channels.teams.bot, eval.run_experiment"` → all import without error.
4. `./scripts/start_api.sh`: `GET /health` returns `{status, llm}` (no `mode`); a
   `POST /api/query` smoke test returns a grounded answer.

## Deployment note

The Spark box's deployed `.env` likely also contains `PIPELINE_MODE=agentic`.
It is harmless (ignored once the field is gone) but should be removed manually so
the deployed config matches the repo. Flagged here as a one-line manual cleanup.

## Risks

- **Low.** Vanilla is unused in production; all its imports are lazy and isolated
  to mode branches. The main risk is missing a reference — mitigated by the
  grep-clean verification step.
