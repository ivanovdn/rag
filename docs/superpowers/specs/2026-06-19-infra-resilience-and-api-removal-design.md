# Transient-infra resilience + dead-API removal

**Date:** 2026-06-19
**Status:** Approved (design), pending implementation
**Author:** Dmytro Ivanov (with Claude Code)

Two independent, no-file-overlap changes shipped on one branch (post-audit cleanup + hardening).

---

## Part 1 — Remove the dead FastAPI service

### Problem
The `api/` FastAPI layer (`/api/query`) is unused: nothing outside `api/` imports it (the Teams bot calls `rag.*` directly), no deployment runs it (the `Dockerfile` CMD and both compose files run `start_teams_bot.py`), and its only would-be consumer is the empty React scaffold. It also carries three audit findings (no API auth, API 500 on infra error, `build_agent()` per request) that vanish when it's removed.

### Changes
- Delete `api/` (`__init__.py`, `main.py`, `models.py`, `routes/__init__.py`, `routes/query.py`).
- Delete `scripts/start_api.sh`.
- Remove `fastapi` and `uvicorn` from `requirements.txt` (only `api/routes/query.py` imported `fastapi`; nothing else uses either — verified by grep).
- CLAUDE.md: drop the FastAPI `/api/query` references (the "Channels" line and the `start_api.sh` command line) so docs match reality (Teams-only).
- Leave `frontend/` untouched (separate empty scaffold; not part of this change).

### Verification
- `grep -rniI "fastapi\|uvicorn\|api\.main\|api\.routes\|/api/query\|start_api" --include=*.py --include=*.sh .` (excluding `docs/superpowers/`) → no hits.
- `PYTHONPATH=. python -c "import channels.teams.bot, eval.run_experiment; from config import settings"` → imports clean.
- Full test suite green (`PYTHONPATH=. pytest -m "not corpus" -q`).

---

## Part 2 — Transient-infra resilience (audit finding #4)

### Problem
When a backend (Qdrant, Ollama-embeddings, or Ollama-LLM) is down or slow, the bot today does the wrong thing, and it does it two different ways:

- **Retrieval failure (embeddings/Qdrant):** the exception is raised inside the `search_policies` tool. **Verified by fault injection (2026-06-19):** LlamaIndex `AgentWorkflow` **swallows the tool exception and feeds it to the LLM as an observation** — it does NOT propagate to `agent.run()`. The LLM then escalates with a reason that leaks the raw error (e.g. *"connection error (Errno 61: Connection refused)"*). So `_run_rag`'s `try/except` never sees retrieval failures.
- **LLM-call failure:** the agent's own LLM call raises and **does** propagate to `_run_rag`, which catches all exceptions and returns an escalation with `reason=str(e)` (raw error leaked).

Both are wrong: a transient infra blip is not a policy question for a human, and once real escalation delivery (#1) lands, every blip would generate a spurious Compliance ticket. There is also no retry (a one-off blip fails immediately) and Qdrant's `query_points` has no timeout (a stuck connection hangs the single-threaded poll loop).

Note: embeddings, the LLM, and Qdrant all live on the same Spark box, so the common "box unreachable" case fails at the agent's first LLM call → the propagating path; the swallowed-tool-error path covers the rarer "Qdrant down, Ollama up."

### Goal
A transient backend failure becomes a clean, deterministic **"policy service temporarily unavailable — try again"** outcome — never a content escalation, never a leaked raw error — recorded as a distinct, queryable Phoenix signal, with retries absorbing brief blips.

### Decisions (locked)
- **Retry policy:** initial attempt + a retry after each backoff `(0.5s, 1.0s, 2.0s)` → up to **4 attempts**, ~3.5s total worst case, retried only for transient errors. (The backoff tuple is the source of truth; shorten it to dial retries down.)
- **Qdrant `query_points` timeout:** 10s.
- **Unavailable message:** an editable module constant `UNAVAILABLE_HTML` in `channels/teams/renderer.py` (same pattern as `WELCOME_HTML`/`LOADING_HTML`), so wording is a one-line change.

### Components

**`rag/resilience.py` (new, pure, unit-tested):**
- `is_transient(exc: BaseException) -> bool` — `True` for `httpx.ConnectError`, `httpx.ConnectTimeout`, `httpx.ReadTimeout`, `httpx.PoolTimeout`, `httpx.HTTPStatusError` with a 5xx status, and Qdrant connection/`ResponseHandlingException` errors. `False` for everything else (4xx, validation, parse, logic) — those must not be masked.
- `retry_transient(fn, *, backoffs=(0.5, 1.0, 2.0))` — calls `fn()`; on a transient exception sleeps `backoffs[i]` and retries (initial attempt + one retry per backoff entry = up to 4 attempts); on a non-transient exception re-raises immediately; after the final backoff re-raises the transient exception. Returns `fn()`'s value on success.

**Interception point 1 — retrieval, inside `rag/tools/search_policies.py`:**
- Wrap the `embed_query(query)` + `search_vectors(...)` calls (the non-BM25 branch; BM25 is off) in `retry_transient`.
- If it still raises a transient error: set module flag `_retrieval_unavailable = True`, record the Phoenix attributes (below), and return the sentinel string `"POLICY_SEARCH_UNAVAILABLE"` to the LLM (so it can't fabricate; the actual user outcome is decided deterministically by the flag, not the LLM).
- Add a `_retrieval_unavailable` module global next to `_last_search_results`, reset to `False` at the top of `search_policies`.

**Interception point 2 — LLM/agent boundary, inside `channels/teams/bot.py` `_run_rag`:**
- Before running: `import rag.tools.search_policies as sp; sp._retrieval_unavailable = False` (module-attr access per the documented gotcha).
- Wrap the `asyncio.run(_run())` call in `retry_transient`.
- After it returns: if `sp._retrieval_unavailable` → return the **unavailable outcome**.
- In the `except` block: if `is_transient(e)` → record Phoenix attributes and return the **unavailable outcome**; else keep the current behavior (escalation dict with `reason=str(e)`).

**Unavailable outcome + rendering:**
- `_run_rag` returns a distinct dict `{"status": "unavailable"}` for this case.
- `channels/teams/renderer.py`: add `UNAVAILABLE_HTML` constant and a `render_unavailable() -> str` returning it. Suggested initial text: *"⚠️ I can't reach the policy database right now. Please try again in a moment."* (no raw error text).
- In the bot's reply-handling code, add a branch **before** the escalation/answer branches: `if result.get("status") == "unavailable": send render_unavailable(); do NOT set a pending rating; continue.` (Unavailable is neither an answer nor an escalation, so it gets no rating prompt and creates no feedback row.)

**Phoenix instrumentation:**
- At each detection point, set span attributes on the current span (via `rag.observability.get_tracer()` / the active span): `infra_unavailable=true`, `failed_component` (`"embeddings"|"qdrant"|"llm"`), `error_type` (the exception class name), `retries_attempted` (int). This makes infra-down events filterable in Phoenix and distinct from content escalations.
- `failed_component` for interception point 1 is derived from which call raised (embeddings vs qdrant); for point 2 it is `"llm"`.

**Qdrant timeout:**
- In `rag/vector_store.py`, give the client (or the `query_points` call) a 10s timeout so a stuck connection can't hang the poll loop indefinitely. Apply via the `QdrantClient(..., timeout=10)` constructor (covers all calls).

### What is explicitly unchanged
- Genuine `NO_RELEVANT_POLICY_FOUND` → escalation behavior (a real "no policy matched" result still escalates as today).
- The reranker's existing fallback-on-error (already safe).
- Escalation delivery (#1) — still deferred; this change just guarantees infra blips never reach it.
- The confidence-floor behavior (#3 — won't-do by design).

### Testing
- **Unit (`tests/unit/test_resilience.py`, no services):** `is_transient` returns True for each transient type and False for 4xx / `ValueError` / parse errors; `retry_transient` (a) returns on first success, (b) retries then succeeds (counts calls), (c) re-raises after exhausting attempts on persistent transient error, (d) re-raises immediately on a non-transient error without retrying. Use a fake callable with a call counter; assert backoff sleeps are invoked (monkeypatch `time.sleep`).
- **Manual/local fault-injection (documented, not in CI):** point a backend at a dead port and confirm the bot path yields the unavailable outcome (not an escalation, no raw error) and that Phoenix shows the `infra_unavailable` attributes. Same "needs live infra / local-only" caveat as the corpus tests.

### Risks
- **Retrying `agent.run()` wholesale** re-does any tool calls from a partial run. Acceptable: transient failures are rare, the common case (box down) fails at the first LLM call before any tool runs, and the backoff budget is ~3.5s total.
- **`is_transient` classification breadth.** Too narrow → some infra errors still leak as escalations; too broad → a real logic error gets masked as "try again." Mitigation: start conservative (the explicit type list above), and the unit tests pin the intended boundary (4xx and logic errors are NOT transient).
