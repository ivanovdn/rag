# Infra Resilience + API Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dead FastAPI service, and make transient backend failures (Qdrant / Ollama-embeddings / Ollama-LLM down or slow) produce a clean, retried, Phoenix-tracked "service temporarily unavailable" outcome instead of a leaked-error escalation.

**Architecture:** A new pure `rag/resilience.py` provides `is_transient()` + `retry_transient()`. Retrieval failures (swallowed by LlamaIndex and fed back to the LLM) are caught *inside* `search_policies` via a module flag; LLM/agent-boundary failures are caught and classified in the Teams bot's `_run_rag`. Both converge on one deterministic "unavailable" outcome rendered by a new editable `UNAVAILABLE_HTML`, and both record `infra_unavailable` Phoenix span attributes.

**Tech Stack:** Python 3.12, pytest 9 (in `.venv`), httpx, qdrant-client, LlamaIndex AgentWorkflow, Phoenix/OpenTelemetry.

## Global Constraints

- **Tests for pure new code follow TDD** (write failing test → see it fail → implement). Tests over *existing* behavior (none here beyond resilience) would use the non-vacuity convention from the prior suite. The resilience unit tests are genuine red→green.
- **Imports at top of every module** EXCEPT the already-deferred `rag.*` imports inside `_run_rag`/tools — those stay deferred because `init_observability()` must run before any LlamaIndex/Ollama import. `rag/resilience.py` imports only `httpx`, `time`, and `qdrant_client.http.exceptions` (none are LlamaIndex/Ollama) so its imports go at top.
- Run from repo root with `PYTHONPATH=.`. Unit suite: `PYTHONPATH=. pytest -m "not corpus" -q`.
- **Retry policy:** backoff tuple `RETRY_BACKOFFS = (0.5, 1.0, 2.0)` → initial attempt + one retry per entry = up to 4 attempts. Tuple is the single source of truth.
- **Qdrant timeout:** 10s via `QdrantClient(timeout=10)`.
- **Unavailable message** is the editable constant `UNAVAILABLE_HTML` in `channels/teams/renderer.py`.
- **`is_transient` must stay conservative:** only connection/timeout errors and 5xx are transient; 4xx, validation, parse, and logic errors are NOT (must not be masked as "try again").
- **The module-attribute gotcha:** access the retrieval flag as `import rag.tools.search_policies as sp; sp._retrieval_unavailable` — never `from ... import _retrieval_unavailable` (would read a stale copy).
- Commit convention: lowercase prefix (`feat:`/`refactor:`/`test:`/`docs:`), end every message with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Commit only at task boundaries with the user's nod.** Work on a feature branch, not `main` (Task 0).

---

### Task 0: Create feature branch

**Files:** none (git only)

- [ ] **Step 1: Branch off main**
```bash
git checkout -b chore/infra-resilience-and-api-removal
```
- [ ] **Step 2: Confirm clean start**
Run: `git status --short`
Expected: no output.

---

### Task 1: Remove the dead FastAPI service

Nothing outside `api/` imports it; no deployment runs it (Dockerfile CMD = `start_teams_bot.py`; neither compose file launches it). Pure deletion.

**Files:**
- Delete: `api/__init__.py`, `api/main.py`, `api/models.py`, `api/routes/__init__.py`, `api/routes/query.py`
- Delete: `scripts/start_api.sh`
- Modify: `requirements.txt` (remove the `# API server` block: `fastapi>=0.135`, `uvicorn>=0.41`)
- Modify: `CLAUDE.md` (lines 5 and 19 — drop FastAPI/`/api/query`/`start_api.sh` references)

**Interfaces:** none consumed/produced — removal only.

- [ ] **Step 1: Delete the API package and launcher**
```bash
git rm -r api scripts/start_api.sh
```

- [ ] **Step 2: Remove fastapi/uvicorn from `requirements.txt`**
Delete these three lines (the block at lines 27-29):
```
# API server
fastapi>=0.135
uvicorn>=0.41
```

- [ ] **Step 3: Fix `CLAUDE.md` line 5** — change:
```markdown
**Channels:** Microsoft Teams (polls Graph `/me/chats` every 5s, imports RAG directly — no HTTP) + FastAPI `/api/query`.
```
to:
```markdown
**Channels:** Microsoft Teams only (polls Graph `/me/chats` every 5s, imports RAG directly — no HTTP). The bot is the sole entry point; there is no HTTP API.
```

- [ ] **Step 4: Fix `CLAUDE.md` line 19** — remove the `start_api.sh` line from the Serve commands block:
```bash
./scripts/start_api.sh                            # FastAPI :8000  (GET /health, POST /api/query)
```
(Delete that single line; leave the `start_teams_bot.py` and `docker compose` lines.)

- [ ] **Step 5: Verify no references remain + imports clean**
Run:
```bash
grep -rniI -e "fastapi" -e "uvicorn" -e "api\.main" -e "api\.routes" -e "/api/query" -e "start_api" --include=*.py --include=*.sh --include=*.txt . --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=.git | grep -v "docs/superpowers/"
```
Expected: **no output**.
Run:
```bash
PYTHONPATH=. python -c "import channels.teams.bot, eval.run_experiment; from config import settings; print('ok')"
```
Expected: prints `ok`.
Run:
```bash
PYTHONPATH=. pytest -m "not corpus" -q
```
Expected: existing suite passes (`45 passed`).

- [ ] **Step 6: Commit**
```bash
git add -A
git commit -m "refactor: remove dead FastAPI service (Teams bot is the only entry point)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `rag/resilience.py` — transient classifier + retry (TDD)

**Files:**
- Create: `rag/resilience.py`
- Create: `tests/unit/test_resilience.py`

**Interfaces:**
- Produces: `rag.resilience.RETRY_BACKOFFS: tuple[float, ...]` = `(0.5, 1.0, 2.0)`; `is_transient(exc: BaseException) -> bool`; `retry_transient(fn: Callable[[], T], *, backoffs: tuple[float, ...] = RETRY_BACKOFFS) -> T`.
- Consumed by: Task 3 (`search_policies`) and Task 5 (`_run_rag`).

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_resilience.py`
```python
import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.resilience import is_transient, retry_transient, RETRY_BACKOFFS


# --- is_transient ---

def test_httpx_connect_and_timeout_are_transient():
    assert is_transient(httpx.ConnectError("x"))
    assert is_transient(httpx.ConnectTimeout("x"))
    assert is_transient(httpx.ReadTimeout("x"))
    assert is_transient(httpx.PoolTimeout("x"))


def test_httpx_5xx_is_transient_4xx_is_not():
    req = httpx.Request("POST", "http://x")
    resp5 = httpx.Response(503, request=req)
    resp4 = httpx.Response(400, request=req)
    assert is_transient(httpx.HTTPStatusError("x", request=req, response=resp5))
    assert not is_transient(httpx.HTTPStatusError("x", request=req, response=resp4))


def test_qdrant_errors_classified():
    assert is_transient(ResponseHandlingException("conn refused"))
    assert is_transient(UnexpectedResponse(status_code=502, reason_phrase="", content=b"", headers=httpx.Headers()))
    assert not is_transient(UnexpectedResponse(status_code=404, reason_phrase="", content=b"", headers=httpx.Headers()))


def test_logic_errors_are_not_transient():
    assert not is_transient(ValueError("bad"))
    assert not is_transient(KeyError("missing"))
    assert not is_transient(RuntimeError("boom"))


# --- retry_transient ---

def test_returns_first_success_no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: slept.append(s))
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        return "ok"
    assert retry_transient(fn) == "ok"
    assert calls["n"] == 1
    assert slept == []


def test_retries_transient_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: slept.append(s))
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("down")
        return "recovered"
    assert retry_transient(fn) == "recovered"
    assert calls["n"] == 3
    assert slept == [0.5, 1.0]  # slept before the 2nd and 3rd attempts


def test_exhausts_and_reraises_on_persistent_transient(monkeypatch):
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: None)
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise httpx.ConnectError("still down")
    with pytest.raises(httpx.ConnectError):
        retry_transient(fn)
    assert calls["n"] == len(RETRY_BACKOFFS) + 1  # initial + one retry per backoff


def test_non_transient_reraises_immediately_without_retry(monkeypatch):
    slept = []
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: slept.append(s))
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise ValueError("logic bug")
    with pytest.raises(ValueError):
        retry_transient(fn)
    assert calls["n"] == 1
    assert slept == []
```

- [ ] **Step 2: Run, verify they fail**
Run: `PYTHONPATH=. pytest tests/unit/test_resilience.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.resilience'`).

- [ ] **Step 3: Implement `rag/resilience.py`**
```python
"""Transient-failure classification and retry for backend (httpx/Qdrant) calls.

A "transient" failure is a backend being briefly unreachable or slow (connection
errors, timeouts, HTTP 5xx). Logic errors (4xx, validation, parse, bugs) are NOT
transient and must surface immediately rather than being retried or masked.
"""

import time
from typing import Callable, TypeVar

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

T = TypeVar("T")

RETRY_BACKOFFS: tuple[float, ...] = (0.5, 1.0, 2.0)

_TRANSIENT_HTTPX = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
    httpx.WriteTimeout,
)


def is_transient(exc: BaseException) -> bool:
    """True if exc represents a transient backend failure (retry-worthy)."""
    if isinstance(exc, _TRANSIENT_HTTPX):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, ResponseHandlingException):
        # qdrant wraps connection/timeout failures in this
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and 500 <= exc.status_code < 600
    return False


def retry_transient(fn: Callable[[], T], *, backoffs: tuple[float, ...] = RETRY_BACKOFFS) -> T:
    """Call fn(); retry on transient errors with the given backoff schedule.

    Up to len(backoffs)+1 attempts. Non-transient exceptions re-raise immediately.
    The last transient exception re-raises after the schedule is exhausted.
    """
    for i, delay in enumerate(backoffs):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified below
            if not is_transient(exc):
                raise
            time.sleep(delay)
    # final attempt (no sleep after); let any exception propagate
    return fn()
```

- [ ] **Step 4: Run, verify pass**
Run: `PYTHONPATH=. pytest tests/unit/test_resilience.py -q`
Expected: PASS (`8 passed`).

- [ ] **Step 5: Commit**
```bash
git add rag/resilience.py tests/unit/test_resilience.py
git commit -m "feat: add transient-failure classifier and retry helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Phoenix helper + Qdrant timeout

**Files:**
- Modify: `rag/observability.py` (add `record_infra_unavailable`)
- Modify: `rag/vector_store.py:25` (add `timeout=10`)

**Interfaces:**
- Produces: `rag.observability.record_infra_unavailable(failed_component: str, error_type: str, retries_attempted: int) -> None`.
- Consumed by: Tasks 4 and 5.

- [ ] **Step 1: Add `record_infra_unavailable` to `rag/observability.py`**
Append this function at the end of the file (after `get_tracer`):
```python
def record_infra_unavailable(failed_component: str, error_type: str, retries_attempted: int) -> None:
    """Emit a Phoenix span marking a transient backend-unavailable event.

    failed_component: "embeddings" | "qdrant" | "llm"
    Makes infra-down events filterable in Phoenix, distinct from content escalations.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("infra_unavailable") as span:
        span.set_attribute("infra_unavailable", True)
        span.set_attribute("failed_component", failed_component)
        span.set_attribute("error_type", error_type)
        span.set_attribute("retries_attempted", retries_attempted)
```

- [ ] **Step 2: Add the Qdrant client timeout** in `rag/vector_store.py`
Change line 25 from:
```python
        _client = QdrantClient(url=settings.active_qdrant_url)
```
to:
```python
        _client = QdrantClient(url=settings.active_qdrant_url, timeout=10)
```

- [ ] **Step 3: Verify import + smoke**
Run:
```bash
PYTHONPATH=. python -c "from rag.observability import record_infra_unavailable; record_infra_unavailable('qdrant','ConnectError',3); print('ok')"
```
Expected: prints `ok` (no exception; with Phoenix off the span is a no-op).
Run:
```bash
PYTHONPATH=. python -c "import rag.vector_store; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Commit**
```bash
git add rag/observability.py rag/vector_store.py
git commit -m "feat: add infra-unavailable Phoenix span + 10s Qdrant timeout

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Make `search_policies` resilient (retrieval interception point)

LlamaIndex swallows tool exceptions and feeds them to the LLM (verified). So retrieval failures must be caught here, flagged, and signalled with a sentinel — the deterministic user outcome is decided later by the flag, not the LLM.

**Files:**
- Modify: `rag/tools/search_policies.py`

**Interfaces:**
- Consumes: `rag.resilience.retry_transient`, `rag.resilience.RETRY_BACKOFFS`, `rag.resilience.is_transient`, `rag.observability.record_infra_unavailable`.
- Produces: module global `_retrieval_unavailable: bool` (read by Task 5 via `import rag.tools.search_policies as sp; sp._retrieval_unavailable`); new sentinel return string `"POLICY_SEARCH_UNAVAILABLE"`.

- [ ] **Step 1: Add the module flag** near the top of `rag/tools/search_policies.py`, right after `_last_search_results`:
```python
_last_search_results: list[dict] = []
_retrieval_unavailable: bool = False
```

- [ ] **Step 2: Reset the flag and wrap the retrieval calls** in the non-BM25 branch.
Inside `search_policies`, at the very top of the function body (after `global _last_search_results`), add the flag to the global declaration and reset it:
```python
    global _last_search_results, _retrieval_unavailable
    _retrieval_unavailable = False
```
Then replace the non-BM25 retrieval block (currently):
```python
    else:
        from rag.embeddings import embed_query
        from rag.vector_store import search_vectors

        query_vector = embed_query(query)
        raw = search_vectors(query_vector, top_k=retrieve_k)

        if not raw:
            _last_search_results = []
            return "NO_RELEVANT_POLICY_FOUND"
```
with:
```python
    else:
        from rag.embeddings import embed_query
        from rag.vector_store import search_vectors
        from rag.resilience import retry_transient, is_transient, RETRY_BACKOFFS
        from rag.observability import record_infra_unavailable

        try:
            query_vector = retry_transient(lambda: embed_query(query))
        except Exception as exc:
            if is_transient(exc):
                _last_search_results = []
                _retrieval_unavailable = True
                record_infra_unavailable("embeddings", type(exc).__name__, len(RETRY_BACKOFFS))
                return "POLICY_SEARCH_UNAVAILABLE"
            raise

        try:
            raw = retry_transient(lambda: search_vectors(query_vector, top_k=retrieve_k))
        except Exception as exc:
            if is_transient(exc):
                _last_search_results = []
                _retrieval_unavailable = True
                record_infra_unavailable("qdrant", type(exc).__name__, len(RETRY_BACKOFFS))
                return "POLICY_SEARCH_UNAVAILABLE"
            raise

        if not raw:
            _last_search_results = []
            return "NO_RELEVANT_POLICY_FOUND"
```
(These `rag.*` imports stay inside the function, matching the existing deferred-import pattern in this file.)

- [ ] **Step 3: Smoke — flag trips on a dead embedding backend, no exception escapes**
Run:
```bash
PYTHONPATH=. python - <<'PY'
from config import settings
settings.phoenix_enabled = False
settings.bm25_enabled = False
settings.ollama_embedding_url = "http://127.0.0.1:1"  # dead
import rag.tools.search_policies as sp
out = sp.search_policies("remote access policy")
print("returned:", out)
print("flag:", sp._retrieval_unavailable)
assert out == "POLICY_SEARCH_UNAVAILABLE"
assert sp._retrieval_unavailable is True
print("OK")
PY
```
Expected: `returned: POLICY_SEARCH_UNAVAILABLE`, `flag: True`, `OK`. (Takes ~3.5s due to the retry backoff against the dead port.)

- [ ] **Step 4: Commit**
```bash
git add rag/tools/search_policies.py
git commit -m "feat: search_policies retries transient retrieval errors, signals unavailable

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Bot `_run_rag` classification + unavailable rendering (LLM interception point + UX)

**Files:**
- Modify: `channels/teams/renderer.py` (add `UNAVAILABLE_HTML` + `render_unavailable`)
- Modify: `channels/teams/bot.py` (import `render_unavailable`; rewrite `_run_rag`; add the unavailable reply branch)

**Interfaces:**
- Consumes: `rag.resilience.retry_transient/is_transient/RETRY_BACKOFFS`, `rag.observability.record_infra_unavailable`, `rag.tools.search_policies._retrieval_unavailable` (via module attr).
- Produces: `render_unavailable() -> str`; `_run_rag` may now return `{"status": "unavailable"}`.

- [ ] **Step 1: Add the editable constant + renderer** to `channels/teams/renderer.py`
After the `LOADING_HTML` definition (near line 27), add:
```python
# Shown when a backend (policy DB / models) is transiently unreachable.
# Editable: reword freely; it must stay valid Teams-limited HTML.
UNAVAILABLE_HTML = (
    "<p><b>⚠️ Policy service temporarily unavailable</b></p>"
    "<p>I can't reach the policy database right now. "
    "Please try again in a moment.</p>"
)


def render_unavailable() -> str:
    """Render the transient-infra-unavailable message."""
    return UNAVAILABLE_HTML
```

- [ ] **Step 2: Import `render_unavailable` in `channels/teams/bot.py`**
In the `from channels.teams.renderer import (...)` block (lines 15-23), add `render_unavailable,` to the imported names.

- [ ] **Step 3: Rewrite `_run_rag`** (lines 37-60) to reset the retrieval flag, retry the agent call, classify failures, and detect the retrieval-unavailable flag:
```python
def _run_rag(question: str) -> dict:
    """Run the RAG pipeline directly (no HTTP). Returns a result dict.

    Outcomes:
      - {"status": "unavailable"}            transient backend failure (retried)
      - {"answer", "citations", "escalation"} normal ComplianceAnswer
    """
    # Deferred imports: init_observability() (start_teams_bot.py) must run before LlamaIndex loads.
    import asyncio
    import rag.tools.search_policies as sp
    from rag.agent import build_agent
    from rag.response import parse_agent_response
    from rag.resilience import retry_transient, is_transient, RETRY_BACKOFFS
    from rag.observability import record_infra_unavailable

    sp._retrieval_unavailable = False

    async def _run():
        agent = build_agent()
        return await agent.run(user_msg=question)

    try:
        response = retry_transient(lambda: asyncio.run(_run()))
    except Exception as e:
        if is_transient(e):
            record_infra_unavailable("llm", type(e).__name__, len(RETRY_BACKOFFS))
            return {"status": "unavailable"}
        print(f"RAG pipeline error: {e}")
        return {
            "answer": "",
            "citations": [],
            "escalation": {"needed": True, "reason": str(e)},
        }

    # Retrieval failed inside the tool (LlamaIndex swallows tool exceptions) →
    # the flag was set in search_policies; surface the unavailable outcome.
    if sp._retrieval_unavailable:
        return {"status": "unavailable"}

    return parse_agent_response(str(response))
```

- [ ] **Step 4: Add the unavailable reply branch** in the message handler (the block at lines 218-238). Replace:
```python
        # Render response
        escalation = result.get("escalation", {})
        if escalation.get("needed"):
            html = render_escalation(text, result)
        elif result.get("answer"):
            html = render_answer(result)
        else:
            html = render_error(text, "No answer returned from the pipeline.")

        sent = self._send_message(chat_id, html)
        if sent:
            print("Reply sent")
            # Send rating prompt and store pending context
            self._send_message(chat_id, RATING_PROMPT_HTML)
            _pending_ratings[chat_id] = {
                "question": text,
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "user": sender_name,
            }
        return bool(sent)
```
with:
```python
        # Transient backend failure — not an answer, not an escalation; no rating prompt.
        if result.get("status") == "unavailable":
            sent = self._send_message(chat_id, render_unavailable())
            if sent:
                print("Unavailable notice sent")
            return bool(sent)

        # Render response
        escalation = result.get("escalation", {})
        if escalation.get("needed"):
            html = render_escalation(text, result)
        elif result.get("answer"):
            html = render_answer(result)
        else:
            html = render_error(text, "No answer returned from the pipeline.")

        sent = self._send_message(chat_id, html)
        if sent:
            print("Reply sent")
            # Send rating prompt and store pending context
            self._send_message(chat_id, RATING_PROMPT_HTML)
            _pending_ratings[chat_id] = {
                "question": text,
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "user": sender_name,
            }
        return bool(sent)
```

- [ ] **Step 5: Verify imports + renderer unit coverage**
Run:
```bash
PYTHONPATH=. python -c "import channels.teams.bot; from channels.teams.renderer import render_unavailable; print(render_unavailable()[:30])"
```
Expected: prints the opening of the unavailable HTML, no error.
Add a renderer test to `tests/unit/test_renderer.py`:
```python
def test_render_unavailable_is_safe_html():
    from channels.teams.renderer import render_unavailable
    html = render_unavailable()
    assert "temporarily unavailable" in html.lower()
    assert "<div" not in html  # Teams-safe tags only
    assert "Errno" not in html and "Exception" not in html  # no raw error leakage
```
Run: `PYTHONPATH=. pytest tests/unit/test_renderer.py -q`
Expected: PASS (prior renderer tests + the new one).

- [ ] **Step 6: Commit**
```bash
git add channels/teams/renderer.py channels/teams/bot.py tests/unit/test_renderer.py
git commit -m "feat: render transient-infra failures as 'unavailable' instead of escalation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: End-to-end fault-injection verification + docs

**Files:**
- Modify: `SETUP.md` (extend the Testing section with the manual fault-injection check)

**Interfaces:** none.

- [ ] **Step 1: Manual fault-injection — dead embedding backend yields the unavailable outcome (not an escalation)**
Run:
```bash
PYTHONPATH=. python - <<'PY'
from config import settings
settings.phoenix_enabled = False
settings.bm25_enabled = False
settings.ollama_embedding_url = "http://127.0.0.1:1"  # dead
import channels.teams.bot as bot
result = bot._run_rag("What is the policy on remote access?")
print("result:", result)
assert result == {"status": "unavailable"}, result
print("OK — unavailable, not escalation, no raw error")
PY
```
Expected: `result: {'status': 'unavailable'}` then `OK`. (Note: this runs the real agent's first LLM call against the live Spark box, then hits the dead embed backend inside the tool — needs the Spark box reachable; if it is not, the agent's own LLM call fails transient → still returns `{"status": "unavailable"}`, which also satisfies the assert. Record which path executed.)

- [ ] **Step 2: Full unit suite green**
Run: `PYTHONPATH=. pytest -m "not corpus" -q`
Expected: all pass (prior 45 + 8 resilience + 1 renderer = `54 passed`).

- [ ] **Step 3: Document the manual check in `SETUP.md`**
In the `## Testing` section, after the existing commands block, add:
```markdown
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

With Phoenix running, the event appears as an `infra_unavailable` span
(attributes: `failed_component`, `error_type`, `retries_attempted`).
```

- [ ] **Step 4: Commit**
```bash
git add SETUP.md
git commit -m "docs: document transient-infra resilience manual check

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the executor

- If a step's verification fails, stop and surface the output — don't force a pass.
- The retrieval smoke test (Task 4 Step 3) and the e2e check (Task 6 Step 1) take ~3.5s+ each due to retry backoff against the dead port — that's expected, not a hang.
- Honor "commit only at task boundaries with the user's nod."
- After all tasks: branch `chore/infra-resilience-and-api-removal` is ready for the finishing-a-development-branch skill.

## Risks

- **`is_transient` breadth.** Conservative by design (Task 2 tests pin the boundary: 4xx/logic = not transient). If a real transient type is later observed leaking as an escalation, add it to the classifier with a test.
- **Whole-`agent.run()` retry** re-does tool calls on a partial run. Acceptable: rare, and the common "box down" case fails at the first LLM call before tools run.
