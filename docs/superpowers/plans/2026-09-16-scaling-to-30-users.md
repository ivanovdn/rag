# Scaling to 20–30 Users Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Teams bot serve 20–30 employees without the burst-time silence, the linear Graph polling growth, or the cold-model-load penalty that show up at that headcount.

**Architecture:** The poll loop stops running the RAG pipeline inline. It detects a message, acknowledges it immediately, and hands the work to **exactly one** worker thread; polling therefore never stalls. Throughput is unchanged (and does not need to change — the inference host serialises anyway), but every user gets a reply within ~2s instead of waiting in silence. Alongside that: client-side `keep_alive` removes the model-reload penalty, a missing `thinking` kill-switch on the `openai-compatible` backend is closed, the Graph poll stops issuing one request per chat per cycle, and it reads every page of the chat list instead of only the first 20 chats.

**Tech Stack:** Python 3.12, `queue.Queue` + `threading` (stdlib — no new dependencies), llama-index (`Ollama` / `OpenAILike` via existing `get_llm()`), pydantic-settings, `requests` (Microsoft Graph), pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-scaling-audit.md` — markdown copy of the audit report (original artifact: <https://claude.ai/code/artifact/97477412-f0f1-4fe8-986a-21b2564eac20>, private to one account). Its load-bearing findings are reproduced under "Findings this plan rests on" below so this plan is self-contained. The spec's addendum records two corrections found when this plan was audited on 2026-09-16 — Graph pagination and the LLM-client caching hazard — both folded into Tasks 4 and 6 below.

## Findings this plan rests on

Measured 2026-09-15/16 against the live stack. Do not re-derive these — they cost GPU time and land in the same place.

| Finding | Value |
|---|---|
| End-to-end per question | ~16 s co-located (18.0 s mean across a 285 ms link) |
| Decode share of that | ~78%, flat **60 tok/s** |
| Retrieval share | ~5% (embed ~85 ms, Qdrant ~160 ms, rerank ~245 ms) |
| Router LLM call | **650 ms** warm, raw HTTP |
| Ollama concurrency | `NUM_PARALLEL=1` — 1/3/6 concurrent took 4.80 / 12.53 / 25.48 s; aggregate throughput flat ~33 tok/s |
| vLLM reranker concurrency | 6 concurrent in 1.71× the time of one (it *does* batch) |
| Cold model load | observed at both **3.77 s** and **51.7 s** on a shared host |
| Capacity | ~210 questions/hour; ~5% utilisation at 30 users × 3 questions/day |

**Already tested and rejected — do not re-propose:** Qwen3-14B on `172.20.0.23` as router (3.4× slower than the current MoE); a BERT classifier container (optimises 4% of latency); Celery (adds a broker to feed a `NUM_PARALLEL=1` bottleneck); "turn thinking off in the router" (already done at `rag/agent.py:176`).

## Global Constraints

- `temperature=0.0` everywhere — deterministic compliance answers. Nothing in this plan changes sampling.
- **Exactly one worker thread. Never more.** `rag/tools/search_policies.py` keeps `_retrieval_unavailable` and `_last_search_results` as module globals that are reset-then-read across a ~16 s agent run (`bot.py:55` resets, `bot.py:76` reads). A second worker interleaves those resets and silently converts a transient infra failure into a false content escalation — the exact invariant the resilience layer exists to uphold. `asyncio` alone is enough to trigger it; threads are not required. Fixing those globals is a **prerequisite** for any worker count above one, and is explicitly out of scope here.
- `init_observability()` must run before any LlamaIndex/Ollama import — `channels/teams/bot.py` keeps its deferred imports inside functions. This is the documented exception to imports-at-top; do not "fix" it.
- Teams HTML supports only `<p> <b> <i> <ul>/<li> <hr> <code>` — never `<div>` or inline styles.
- Rating detection is exact: `message.strip() in {"-1","0","1","2"}`. Anything else is a new question and drops pending state.
- No new runtime dependencies. `queue` and `threading` are stdlib.
- Do not change server-side configuration on `172.20.0.22` — it is a shared host this project does not own. Every change in this plan is client-side.
- After Task 3, **two threads call Microsoft Graph** (poll thread: chat list, acks, ratings; worker: answers). Anything they share must be lock-protected: `TeamsBot._inflight` (its own lock) and `TokenRefresher.get_access_token` (Task 3 Step 4b). `_pending_ratings` needs no lock — every access is a single dict operation, atomic under the GIL.
- **Never cache the LLM client across requests** — no `lru_cache` on `get_llm`, no module-level `Ollama`. `_run_rag` starts a fresh event loop per request with `asyncio.run()`; llama-index's `Ollama` creates its `httpx.AsyncClient` once and reuses it, and a pooled connection from a closed loop fails on the next loop with `RuntimeError: Event loop is closed` (reproduced 2026-09-16, httpx 0.28.1). `RuntimeError` is not transient, so it would surface as a false content escalation. See Task 6.
- Microsoft Graph **pages every collection**. `GET /me/chats` returns 20 chats per page by default (`$top` max 50) plus an `@odata.nextLink` when there are more. Any code that lists chats must follow it (Task 4 Step 4b).

## Before you start

- [ ] **Work on three branches, split by subsystem — never on `main`.** Each branch is reviewed, verified and merged on its own (finish each with superpowers:finishing-a-development-branch). Tasks 2 and 6 extend the test file Task 1 creates, and Task 5 builds on Task 4's helpers, so those pairs stay together. Tasks 3, 4 and 5 all edit `process_new_messages` and `run()` in `bot.py`, so the Graph branch is cut from the queue branch, not from `main`.

| Branch | Tasks | Cut from | Why it stands alone |
|---|---|---|---|
| `feat/llm-client-tuning` | 1, 2, 6 | `main` | Touches only `rag/agent.py`, the Ollama block of `config.py`, one test file and CLAUDE.md. No Teams testing, no production window. Merge first. |
| `feat/teams-worker-queue` | 3 | `main` | The user-visible fix and the largest change. Needs the manual burst and restart checks below. Its own revert path. |
| `feat/graph-polling` | 4, 5 | `feat/teams-worker-queue` | Same two `bot.py` functions as Task 3. Task 5 can be abandoned at its probe gate without touching the other branches. |

Create each branch when you reach it:

```bash
git switch -c feat/llm-client-tuning main                    # Tasks 1, 2, 6
git switch -c feat/teams-worker-queue main                   # Task 3
git switch -c feat/graph-polling feat/teams-worker-queue     # Tasks 4, 5 — once Task 3 is complete
```

- [ ] **Confirm the baseline is green**, so any later failure is yours:

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit -q`
Expected: all passed (101 passed on 2026-09-16). A failure here is pre-existing — stop and report it rather than starting Task 1 on top of it.

- [ ] **Book the Task 5 probe window.** The probe must run on the bot host with the bot container stopped (it rotates the live refresh token), so Task 5 costs about a minute of production downtime. Agree the time before you reach it.

---

### Task 1: Client-side `keep_alive` for the Ollama model

The measured cold reload cost ranged from 3.77 s to 51.7 s on a shared host. `keep_alive` is a field on LlamaIndex's `Ollama` class (default `'5m'`, type `float | str | None`), so model residency is controllable from this repo without touching the host. Use `30m` rather than an unbounded value: parking 23.9 GB on a box other teams share is a social cost, and 30 m covers working-day gaps.

**Files:**
- Modify: `config.py` (add one setting to the Ollama block, after `llm_remote_request_timeout` at `config.py:20`)
- Modify: `rag/agent.py` (the `Ollama(...)` construction in `get_llm`, `rag/agent.py:171-178`)
- Modify: `.env.example` (document the knob in the Ollama block)
- Test: `tests/unit/test_llm_config.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `settings.ollama_keep_alive: str`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_llm_config.py`:

```python
import pytest

from config import settings
from rag.agent import get_llm


def test_ollama_llm_gets_keep_alive_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    monkeypatch.setattr(settings, "ollama_keep_alive", "30m")
    llm = get_llm()
    assert llm.keep_alive == "30m"


def test_ollama_keep_alive_default_is_longer_than_ollama_default():
    # Ollama's own default is '5m'; ours must outlast a working-day gap.
    assert settings.ollama_keep_alive != "5m"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_llm_config.py -v`
Expected: FAIL — both tests, with `AttributeError: ... has no attribute 'ollama_keep_alive'`, raised by `monkeypatch.setattr` because the setting does not exist yet. That message embeds the full `Settings` repr — including secrets from your `.env` — so never paste it into a ticket or commit message.

- [ ] **Step 3: Add the setting**

In `config.py`, immediately after the `llm_remote_request_timeout` line in the Ollama block:

```python
    # How long Ollama keeps the model resident after a request. Ollama's own
    # default is "5m"; sporadic use then pays a reload (measured 3.8s-51.7s on
    # the shared host). Not unbounded: the box is shared with other projects.
    ollama_keep_alive: str = "30m"
```

- [ ] **Step 4: Wire it into the LLM**

In `rag/agent.py`, in the `else:` (Ollama) branch of `get_llm`, add the `keep_alive` argument:

```python
        return Ollama(
            model=model or settings.llm_model,
            base_url=settings.active_ollama_url,
            request_timeout=float(settings.active_request_timeout),
            temperature=settings.llm_temperature,
            thinking=False,
            keep_alive=settings.ollama_keep_alive,
            additional_kwargs={"num_predict": 4096, "num_ctx": 8192},
        )
```

- [ ] **Step 5: Document the knob**

In `.env.example`, in the `# Ollama` block, after `LLM_REQUEST_TIMEOUT=120`:

```bash
# How long Ollama keeps the model loaded after a request (Ollama default: 5m).
OLLAMA_KEEP_ALIVE=30m
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_llm_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add config.py rag/agent.py .env.example tests/unit/test_llm_config.py
git commit -m "perf(llm): keep the Ollama model resident for 30m

Cold reloads measured between 3.8s and 51.7s on the shared host.
keep_alive is a client-side field, so this needs no host config change."
```

---

### Task 2: Close the `thinking` gap on the `openai-compatible` backend

`get_llm`'s Ollama branch sets `thinking=False`; the `OpenAILike` branch has no equivalent. Flipping `LLM_BACKEND=openai-compatible` (a documented, supported path — SETUP.md uses it for vLLM/llama-server) with any Qwen3 model silently restores reasoning on the router *and* every agent turn. Measured on `172.20.0.23`: **294 reasoning tokens and 34,575 ms** to emit a 16-token classification, versus 2,184 ms with thinking off. It still parses — `_extract_json` is tolerant — so it degrades quietly rather than failing.

vLLM takes the switch as a request-body field, which the OpenAI SDK passes through via `extra_body` (verified present in `Completions.create`), and `OpenAILike` forwards `additional_kwargs` into the request.

**Files:**
- Modify: `rag/agent.py` (the `OpenAILike(...)` construction in `get_llm`, `rag/agent.py:157-167`)
- Test: `tests/unit/test_llm_config.py` (extend the file created in Task 1)

**Interfaces:**
- Consumes: `get_llm()` from `rag/agent.py` (unchanged signature).
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_llm_config.py`:

```python
def test_openai_like_disables_thinking(monkeypatch):
    # Qwen3 models think by default on vLLM. Measured cost: 294 reasoning
    # tokens and 34.5s for one 16-token router classification.
    monkeypatch.setattr(settings, "llm_backend", "openai-compatible")
    llm = get_llm()
    body = llm.additional_kwargs["extra_body"]
    assert body["chat_template_kwargs"]["enable_thinking"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_llm_config.py::test_openai_like_disables_thinking -v`
Expected: FAIL — `KeyError: 'extra_body'`

- [ ] **Step 3: Add the kill-switch**

In `rag/agent.py`, above `def get_llm(...)`, add the constant:

```python
# Qwen3-family models emit reasoning traces by default on vLLM/llama-server.
# Measured: 294 reasoning tokens and 34.5s to produce a 16-token router
# classification, versus 2.2s with thinking off. The tolerant JSON extractor
# still parses it, so this regresses silently — hence the explicit switch.
# The Ollama branch has its own native `thinking=False` argument.
_NO_THINKING_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
```

Then in the `openai-compatible` branch of `get_llm`:

```python
        return OpenAILike(
            model=model or settings.openai_model,
            api_base=settings.openai_api_base,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            request_timeout=float(settings.active_request_timeout),
            is_chat_model=True,
            is_function_calling_model=True,
            additional_kwargs={"extra_body": _NO_THINKING_BODY},
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_llm_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Record the gotcha**

Add one row to the "Gotchas / Lessons Learned" table in `CLAUDE.md`:

```markdown
| Qwen3 on `openai-compatible` silently thinks | `get_llm()`'s Ollama branch sets `thinking=False`; the `OpenAILike` branch needs `additional_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}`. Measured 34.5s vs 2.2s for one router call — and it still parses, so it degrades quietly. |
```

- [ ] **Step 6: Commit**

```bash
git add rag/agent.py tests/unit/test_llm_config.py CLAUDE.md
git commit -m "fix(llm): disable thinking on the openai-compatible backend

Only the Ollama branch had a kill-switch. Switching LLM_BACKEND with any
Qwen3 model restored reasoning on the router and every agent turn: 34.5s
vs 2.2s for one classification, and it parses, so it failed silently."
```

---

### Task 3: Queue + a single worker thread

This is the task that fixes the user-visible problem. Today `process_new_messages()` → `_send_reply()` → `_run_rag()` blocks the poll loop for ~16 s, so a second user is not even *detected* until the first is answered, and their "searching…" notice arrives after the queue wait rather than before it. At a burst of 10 the last person waits ~2.7 minutes in complete silence.

Splitting detection from processing fixes the acknowledgement without changing throughput (the host serialises anyway, so a single worker loses nothing).

Four consequences to get right:

1. **Acknowledgement copy.** The ack is now sent *before* routing, so it must suit all four outcomes — a greeting must not be told "Searching compliance policies…". `LOADING_HTML` is replaced by an outcome-neutral `ACK_HTML`.
2. **Crash safety.** `last_check` currently advances for every message *seen*. With a queue, a restart with queued work would skip those messages forever — they were marked processed but never answered. The persisted watermark is therefore held back to just before the oldest in-flight message, and in-flight IDs are excluded from the persisted `processed_messages`. In-memory state still advances, so the running process never re-enqueues. Semantics become at-least-once: a crash between sending an answer and clearing in-flight re-delivers that answer. That is the right trade for a compliance bot — a duplicate answer beats a silently dropped question.
3. **Two threads now call Graph.** `_send_message` runs on both the poll thread (ack, ratings, welcome) and the worker (answers), and both go through `TokenRefresher.get_access_token`, which refreshes and rewrites `refresh_token.json` with no lock. Two threads seeing an expired token at once would refresh twice and could interleave the file write. A lock in `TokenRefresher` closes this (Step 4b). `_pending_ratings` needs none: each access is one dict operation. Note what actually happens in the awkward window, which is *not* "credited to the last answered question": once the user sends a new question, `_handle_inbound` clears the pending rating immediately, before that question is even queued. So if they then type `2` while it is still waiting for the worker, the `2` is no longer recognised as a rating — it is acked and enqueued as a new question and goes through the RAG pipeline. A rating only lands when it arrives after an answer and before the next message. That is reasonable behaviour, not a bug; it is documented here because it is easy to assume otherwise.
4. **A dead worker must not be silent.** If the worker thread ever exits, the poll thread would keep sending "Got your message" forever and nobody would get an answer. The poll loop therefore checks the worker every cycle and restarts it (`_ensure_worker`). The worker also never sends raw exception text to a user — today those exceptions never reach the chat, and the renderer does not HTML-escape.

**Files:**
- Modify: `channels/teams/renderer.py:24-27` (replace `LOADING_HTML` with `ACK_HTML`)
- Modify: `channels/teams/bot.py` (imports; `TeamsBot.__init__`; `_save_state`; split `_send_reply` into `_handle_inbound` + `_answer`; `_worker_loop` + `_ensure_worker`; `process_new_messages`; `run`)
- Modify: `channels/teams/auth.py` (`TokenRefresher.__init__` + `get_access_token` — one lock)
- Modify: `tests/unit/test_bot_routing.py` (4 assertions change — the ack is now sent for every routed message)
- Test: `tests/unit/test_bot_queue.py` (create)
- Test: `tests/unit/test_teams_auth.py` (create)

**Interfaces:**
- Consumes: `settings.teams_poll_interval`, `settings.teams_max_processed_messages` (existing); `ACK_HTML` from `renderer.py`.
- Produces:
  - `TeamsBot._handle_inbound(chat_id: str, message_text: str, sender_name: str = "Unknown", message_id: str | None = None, created_time: datetime | None = None) -> bool` — poll-thread entry point. Handles ratings/commands inline, otherwise acks and enqueues.
  - `TeamsBot._answer(chat_id: str, text: str, sender_name: str = "Unknown") -> bool` — worker-thread entry point. Router + RAG + reply. This is the old `_send_reply` minus the rating/command/ack handling.
  - `TeamsBot._worker_loop() -> None` — the single consumer.
  - `TeamsBot._ensure_worker() -> None` — starts the worker, or restarts it if it died. Called once at startup and once per poll cycle.
  - `TeamsBot._worker: threading.Thread | None`, `TeamsBot._inflight: dict[str, datetime]`, `TeamsBot._inflight_lock: threading.Lock`.
  - `TokenRefresher.get_access_token()` — unchanged signature, now thread-safe.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_bot_queue.py`:

```python
import threading
from datetime import datetime, timedelta, timezone

import pytest

import channels.teams.bot as bot


@pytest.fixture
def qbot(monkeypatch):
    """A TeamsBot with network mocked; records every HTML it 'sends'."""
    b = bot.TeamsBot(token_refresher=object())
    sent = []
    monkeypatch.setattr(b, "_send_message",
                        lambda chat_id, text, content_type="html": sent.append(text) or True)
    bot._pending_ratings.clear()
    while not b._work_q.empty():
        b._work_q.get_nowait()
    b._inflight.clear()
    b._sent = sent
    return b


def test_inbound_acks_immediately_and_does_not_run_rag(monkeypatch, qbot):
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("poll thread must not run RAG"))
    now = datetime.now(timezone.utc)
    qbot._handle_inbound("chat1", "Can I install software?", "Ann", "m1", now)
    assert any("Got your message" in h for h in qbot._sent)
    assert qbot._work_q.qsize() == 1


def test_inbound_marks_message_inflight(qbot):
    now = datetime.now(timezone.utc)
    qbot._handle_inbound("chat1", "Can I install software?", "Ann", "m1", now)
    assert qbot._inflight == {"m1": now}


def test_rating_is_handled_inline_and_not_enqueued(monkeypatch, qbot):
    monkeypatch.setattr(bot, "save_feedback", lambda **kw: None)
    bot._pending_ratings["chat1"] = {"question": "q", "answer": "a", "citations": [], "user": "Ann"}
    qbot._handle_inbound("chat1", "2", "Ann", "m2", datetime.now(timezone.utc))
    assert qbot._work_q.qsize() == 0
    assert "chat1" not in bot._pending_ratings


def test_worker_drains_job_and_clears_inflight(monkeypatch, qbot):
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: {"answer": "See AUP.", "citations": [], "escalation": {"needed": False}})
    now = datetime.now(timezone.utc)
    qbot._handle_inbound("chat1", "Can I install software?", "Ann", "m1", now)

    t = threading.Thread(target=qbot._worker_loop, daemon=True)
    t.start()
    qbot._work_q.join()

    assert qbot._inflight == {}
    assert any("See AUP." in h for h in qbot._sent)


def test_saved_watermark_is_held_before_the_oldest_inflight_message(qbot, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    old = datetime.now(timezone.utc) - timedelta(minutes=2)
    new = datetime.now(timezone.utc)
    qbot.last_check = new
    qbot.processed_messages = {"m_old", "m_done"}
    qbot._inflight = {"m_old": old}

    qbot._save_state()
    saved = json.loads((tmp_path / "bot_state.json").read_text())

    # Watermark rewound behind the queued message, so a restart re-delivers it.
    assert datetime.fromisoformat(saved["last_check"]) < old
    # ...and it is not in the processed set, which would otherwise skip it.
    assert "m_old" not in saved["processed_messages"]
    assert "m_done" in saved["processed_messages"]


def test_saved_watermark_is_last_check_when_nothing_inflight(qbot, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    now = datetime.now(timezone.utc)
    qbot.last_check = now
    qbot.processed_messages = {"m1"}
    qbot._inflight = {}

    qbot._save_state()
    saved = json.loads((tmp_path / "bot_state.json").read_text())
    assert datetime.fromisoformat(saved["last_check"]) == now


def test_ensure_worker_restarts_a_dead_worker(qbot):
    qbot._ensure_worker()
    first = qbot._worker
    assert first is not None and first.is_alive()

    # Simulate the worker dying: swap in a thread that has already finished.
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    qbot._worker = dead

    qbot._ensure_worker()
    assert qbot._worker is not dead
    assert qbot._worker.is_alive()
```

Create `tests/unit/test_teams_auth.py`:

```python
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import channels.teams.auth as auth


def test_concurrent_callers_refresh_the_token_once(tmp_path, monkeypatch):
    """The poll thread and the RAG worker both call get_access_token(); an expired
    token must be refreshed once, not once per thread."""
    token_file = tmp_path / "refresh_token.json"
    token_file.write_text(json.dumps({"refresh_token": "seed"}))
    monkeypatch.setattr(auth, "_TOKEN_FILE", token_file)
    refresher = auth.TokenRefresher()

    calls = []

    def slow_refresh():
        calls.append(1)
        time.sleep(0.05)  # long enough for the second thread to arrive mid-refresh
        refresher.access_token = "tok"
        refresher.token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        return refresher.access_token

    monkeypatch.setattr(refresher, "_refresh_access_token", slow_refresh)

    threads = [threading.Thread(target=refresher.get_access_token) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == [1], f"token refreshed {len(calls)} times"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_bot_queue.py tests/unit/test_teams_auth.py -v`
Expected: `test_bot_queue.py` — all 7 tests ERROR in the `qbot` fixture with `AttributeError: 'TeamsBot' object has no attribute '_work_q'`; `test_teams_auth.py` — FAIL with `AssertionError: token refreshed 2 times`. (Both verified against the pre-change code on 2026-09-16.)

- [ ] **Step 3: Replace `LOADING_HTML` with an outcome-neutral `ACK_HTML`**

In `channels/teams/renderer.py`, replace lines 24–27:

```python
# Sent by the poll thread the moment a message is seen — before routing, so it
# must read correctly whether the reply turns out to be an answer, a greeting,
# or an out-of-scope redirect.
ACK_HTML = (
    "<p><b>Got your message.</b><br>"
    "<i>Looking into it — this can take up to a minute.</i></p>"
)
```

- [ ] **Step 4: Add the queue, the worker and in-flight tracking**

In `channels/teams/bot.py`, add to the stdlib imports at the top:

```python
import queue
import threading
```

Change the renderer import block to take `ACK_HTML` instead of `LOADING_HTML`:

```python
from channels.teams.renderer import (
    ACK_HTML,
    RATING_PROMPT_HTML,
    RATING_THANKS_HTML,
    WELCOME_HTML,
    render_answer,
    render_escalation,
    render_error,
    render_out_of_scope,
    render_unavailable,
    render_unintelligible,
)
```

In `TeamsBot.__init__`, after `self.processed_messages = state["processed_messages"]`:

```python
        # EXACTLY ONE worker consumes this queue. Do not raise the worker count.
        # rag/tools/search_policies.py keeps _retrieval_unavailable and
        # _last_search_results as module globals, reset before an agent run and
        # read ~16s later. A second worker interleaves those resets and turns a
        # transient infra failure into a false content escalation, silently.
        # Fix those globals (ToolCallResult.tool_output is per-request) before
        # ever running more than one.
        self._work_q: "queue.Queue[tuple[str, str, str, str]]" = queue.Queue()
        self._worker: threading.Thread | None = None  # started by _ensure_worker() in run()
        # message_id -> createdDateTime, for messages accepted but not yet answered.
        self._inflight: dict[str, datetime] = {}
        self._inflight_lock = threading.Lock()
```

- [ ] **Step 4b: Make token refresh thread-safe**

In `channels/teams/auth.py`, add `import threading` to the stdlib imports at the top (after `import json`). In `TokenRefresher.__init__`, after `self.token_expires_at = None`:

```python
        # Both the poll thread and the RAG worker call get_access_token().
        self._lock = threading.Lock()
```

Replace `get_access_token`:

```python
    def get_access_token(self):
        """Get access token, refreshing only if expired.

        Thread-safe: the poll thread (chat list, acks, ratings) and the RAG worker
        (answers) both call this. Without the lock, two threads that see an expired
        token refresh twice and can interleave the refresh_token.json rewrite.
        """
        with self._lock:
            if self._is_token_expired():
                self._refresh_access_token()
            return self.access_token
```

- [ ] **Step 5: Hold the persisted watermark behind in-flight work**

Replace the body of `_save_state` in `channels/teams/bot.py`:

```python
    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with self._inflight_lock:
            pending_times = list(self._inflight.values())
            pending_ids = set(self._inflight)

        # Persist a watermark BEHIND anything still queued, and keep queued ids
        # out of the processed set, so a restart re-delivers unanswered work
        # instead of skipping it. In-memory last_check still advances, so the
        # running process never re-enqueues what it already holds.
        watermark = min(pending_times) - timedelta(milliseconds=1) if pending_times else self.last_check

        ids = [
            mid for mid in list(self.processed_messages)[-settings.teams_max_processed_messages:]
            if mid not in pending_ids
        ]
        with open(STATE_FILE, "w") as f:
            json.dump(
                {
                    "last_check": watermark.isoformat(),
                    "processed_messages": ids,
                },
                f,
                indent=2,
            )
```

- [ ] **Step 6: Split `_send_reply` into `_handle_inbound` and `_answer`**

In `channels/teams/bot.py`, rename `_send_reply` to `_answer` and delete from it the command handling, the `[media/emoji]` guard, the rating branch, and the `self._send_message(chat_id, LOADING_HTML)` line — those move to the poll thread. `_answer` begins directly at the router block:

```python
    def _answer(self, chat_id, text, sender_name="Unknown"):
        """Worker thread: route, run RAG, reply. Never called from the poll loop."""
        # Pre-retrieval classification: only in-scope questions reach policy search.
        if settings.router_enabled:
            from rag.router import classify_message, resolve, Category  # deferred: observability-first
            from rag.observability import record_classification

            decision = classify_message(text)
            category = resolve(decision, settings.router_confidence_floor)
            record_classification(
                category.value,
                decision.confidence,
                fallback=(category != decision.category or decision.fallback),
                message=text,
            )

            if category == Category.GREETING:
                self._send_message(chat_id, WELCOME_HTML)
                return True
            if category == Category.OUT_OF_SCOPE:
                self._send_message(chat_id, render_out_of_scope())
                return True
            if category == Category.UNINTELLIGIBLE:
                self._send_message(chat_id, render_unintelligible())
                return True
            # Category.IN_SCOPE falls through to the RAG pipeline below.

        result = _run_rag(text)
```

The remainder of the old method (the `unavailable` branch, rendering, sending, and storing the pending rating) is unchanged.

Then add the poll-thread entry point immediately above `_answer`:

```python
    def _handle_inbound(self, chat_id, message_text, sender_name="Unknown",
                        message_id=None, created_time=None):
        """Poll thread: answer cheap cases inline, acknowledge and enqueue the rest.

        Nothing here may make an LLM call — the poll loop must keep polling.
        """
        if not message_text or not message_text.strip():
            return False

        text = message_text.strip()

        if text.lower() in ("start", "/start", "help", "/help"):
            _pending_ratings.pop(chat_id, None)
            self._send_message(chat_id, WELCOME_HTML)
            return True

        if text == "[media/emoji]":
            return True

        # Check if this is a rating for a pending answer
        if chat_id in _pending_ratings and text in _VALID_RATINGS:
            ctx = _pending_ratings.pop(chat_id)
            save_feedback(
                question=ctx["question"],
                answer=ctx["answer"],
                citations=ctx["citations"],
                rating=int(text),
                user=ctx["user"],
                chat_id=chat_id,
            )
            self._send_message(chat_id, RATING_THANKS_HTML)
            print(f"Feedback saved: rating={text} for '{ctx['question'][:50]}...'")
            return True

        # Not a rating — clear any pending state and hand off as a question.
        _pending_ratings.pop(chat_id, None)

        if message_id and created_time:
            with self._inflight_lock:
                self._inflight[message_id] = created_time

        self._send_message(chat_id, ACK_HTML)
        self._work_q.put((chat_id, text, sender_name, message_id or ""))
        return True

    def _worker_loop(self):
        """The single consumer. See the __init__ comment before adding a second."""
        while True:
            chat_id, text, sender_name, message_id = self._work_q.get()
            try:
                self._answer(chat_id, text, sender_name=sender_name)
            except Exception as e:
                # Log the detail; never send raw exception text to a user — the
                # renderer does not HTML-escape, and today these never reach the chat.
                print(f"Worker error answering in {chat_id}: {e!r}")
                self._send_message(
                    chat_id,
                    render_error(text, "Something went wrong while looking this up."),
                )
            finally:
                if message_id:
                    with self._inflight_lock:
                        self._inflight.pop(message_id, None)
                self._work_q.task_done()

    def _ensure_worker(self):
        """Start the single worker, or restart it if it has died.

        Called once at startup and once per poll cycle. A dead worker would
        otherwise be silent: the poll thread keeps acknowledging and nobody is
        ever answered.
        """
        if self._worker is not None and self._worker.is_alive():
            return
        if self._worker is not None:
            print("ERROR: rag-worker thread died; restarting it")
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="rag-worker")
        self._worker.start()
```

- [ ] **Step 7: Route detection through the new entry point**

In `process_new_messages`, replace the final call in the message loop:

```python
                self.processed_messages.add(message_id)
                self._handle_inbound(
                    chat_id, clean_message,
                    sender_name=sender_name,
                    message_id=message_id,
                    created_time=created_time if created_datetime else None,
                )
```

Note `created_time` is already parsed a few lines above for the `newest_message_time` comparison; reuse that variable rather than parsing twice.

- [ ] **Step 8: Start the worker, and keep it alive**

In `run()`, immediately after `self._acquire_pid_lock()`:

```python
        self._ensure_worker()
```

In the `while True:` loop, add a liveness check as the first line of the `try:` body, before `self.process_new_messages()`:

```python
                    self._ensure_worker()  # restarts the worker if it ever died
                    self.process_new_messages()
```

And extend the startup banner, after the `Polling every ...` line:

```python
            print("Workers: 1 (single-threaded by design — see _work_q comment)")
```

- [ ] **Step 9: Update the four affected assertions in the existing routing tests**

In `tests/unit/test_bot_routing.py`, the ack is now sent by the poll thread, so `_answer` no longer sends a loading notice. Change the fixture's method under test and the four assertions:

- Every `teams_bot._send_reply("chat1", ...)` call becomes `teams_bot._answer("chat1", ...)`.
- Lines 32, 42, 52 — `assert not any("Searching compliance policies" in h ...)` — become:
  ```python
      assert not any("Got your message" in h for h in teams_bot._sent)  # ack is the poll thread's job
  ```
- Line 62 — `assert any("Searching compliance policies" in h ...)  # LOADING_HTML` — delete it. The next line already asserts `"chat1" in bot._pending_ratings`, which is the surviving evidence that the in-scope path ran and answered.

- [ ] **Step 10: Run the full unit suite**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit -v`
Expected: PASS — `test_bot_queue.py` 7 passed, `test_teams_auth.py` 1 passed, `test_bot_routing.py` still passing, no regressions elsewhere.

- [ ] **Step 11: Commit**

```bash
git add channels/teams/bot.py channels/teams/renderer.py channels/teams/auth.py \
        tests/unit/test_bot_queue.py tests/unit/test_bot_routing.py tests/unit/test_teams_auth.py
git commit -m "feat(teams): acknowledge inbound messages off the poll loop

The poll loop ran the ~16s RAG pipeline inline, so a queued user was not
detected until the previous answer finished and their loading notice
arrived after the wait rather than before it. Detection now acks
immediately and hands off to a single worker thread.

Exactly one worker: search_policies' module globals are reset-then-read
across an agent run and would race with a second.

The persisted watermark is held behind in-flight work so a restart
re-delivers unanswered questions instead of skipping them.

TokenRefresher is now lock-protected (two threads call Graph), and the
poll loop restarts the worker if it ever dies instead of acking into a
void."
```

---

### Task 4: Stop pulling 20 full messages per chat, and back off when idle

Each cycle issues `GET /me/chats` plus one `GET /me/chats/{id}/messages` per chat, with no `$top` — so every cycle downloads the default 20-message page, with full HTML bodies, for every chat. At 30 chats that is ~31 calls every cycle, around the clock, against a delegated user token.

`$top` is the safe half of the fix and needs no verification. The adaptive interval is pure arithmetic. (`$expand=lastMessagePreview`, which would collapse the N+1 entirely, is Task 5 — it needs live verification first.)

This task also fixes a latent bug that becomes real at 30 users: Graph pages `GET /me/chats` (20 per page by default, `$top` max 50), and the current loop reads only the first page and never follows `@odata.nextLink`. Past ~20 chats, some users are simply never polled. The docs are explicit — "If the result set for all chats spans multiple pages, the response object includes an @odata.nextLink property … continue making additional requests with the @odata.nextLink URL" — so this needs no live verification either.

**Files:**
- Modify: `config.py` (four settings after `teams_poll_interval` at `config.py:107`)
- Modify: `channels/teams/bot.py` (a `_CHATS_PAGE_SIZE` constant; a new `_get_all_pages`; the chat-list `url` and the `messages_url` in `process_new_messages`; a new `_current_poll_interval`; the sleep in `run`)
- Modify: `.env.example` (add a Teams Bot block — there is none today — with the auth placeholders and the polling knobs)
- Test: `tests/unit/test_bot_polling.py` (create)

**Interfaces:**
- Consumes: `settings.teams_poll_interval` (existing).
- Produces: `TeamsBot._current_poll_interval(now: datetime) -> int`; `TeamsBot._get_all_pages(url: str) -> list[dict]`; `_CHATS_PAGE_SIZE: int` (module constant); `settings.teams_messages_page_size: int`; `settings.teams_idle_poll_interval: int`; `settings.teams_business_hours_start_utc: int`; `settings.teams_business_hours_end_utc: int`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_bot_polling.py`:

```python
from datetime import datetime, timezone

import pytest

import channels.teams.bot as bot


@pytest.fixture
def pbot(tmp_path, monkeypatch):
    # process_new_messages() calls _save_state(); keep it off the real state file.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    return bot.TeamsBot(token_refresher=object())


def test_business_hours_use_the_fast_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 5)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 30)
    # Tuesday 11:00 UTC
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 11, 0, tzinfo=timezone.utc)) == 5


def test_nights_use_the_idle_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 5)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 30)
    # Tuesday 03:00 UTC
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)) == 30


def test_weekends_use_the_idle_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 5)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 30)
    # Saturday 11:00 UTC
    assert pbot._current_poll_interval(datetime(2026, 9, 19, 11, 0, tzinfo=timezone.utc)) == 30


def test_messages_are_requested_with_a_page_cap(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 5)
    urls = []
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")


    def fake_api(url, method="GET", json_data=None):
        urls.append(url)
        # Match on the chat-list call without assuming its query string: Task 5
        # appends $expand=lastMessagePreview to this same URL.
        if "/messages" not in url:
            return {"value": [{"id": "c1"}]}
        return {"value": []}

    monkeypatch.setattr(pbot, "_api_request", fake_api)
    pbot.process_new_messages()
    message_urls = [u for u in urls if "/messages" in u]
    # Check the per-chat message URLs only: the chat-list URL carries its own
    # $top=50, and "$top=5" is a substring of "$top=50".
    assert message_urls and all("$top=5" in u for u in message_urls), urls


def test_chat_list_follows_next_link(monkeypatch, pbot):
    """Graph pages /me/chats (20 per page by default). A chat on page 2 must still be polled."""
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 5)
    urls = []
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")

    def fake_api(url, method="GET", json_data=None):
        urls.append(url)
        if "/messages" in url:
            return {"value": []}
        if "skiptoken" in url:                      # page 2
            return {"value": [{"id": "chatB"}]}
        return {                                    # page 1
            "value": [{"id": "chatA"}],
            "@odata.nextLink": f"{bot.GRAPH_API}/me/chats?$top=50&$skiptoken=abc",
        }

    monkeypatch.setattr(pbot, "_api_request", fake_api)
    pbot.process_new_messages()
    assert any("chatB/messages" in u for u in urls), urls
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_bot_polling.py -v`
Expected: FAIL — all five tests, each at its first `monkeypatch.setattr(bot.settings, ...)` line with `AttributeError: ... has no attribute 'teams_idle_poll_interval'` (or `'teams_messages_page_size'`), because the settings do not exist yet (verified 2026-09-16; the message embeds the `Settings` repr with your `.env` secrets — don't paste it anywhere). Once Step 3 adds the settings, the failures move to the real gaps: the interval tests with `AttributeError: 'TeamsBot' object has no attribute '_current_poll_interval'`, the page-cap test with `AssertionError` (no `$top` on the message URL), the nextLink test with `AssertionError` (`chatB` never fetched).

- [ ] **Step 3: Add the settings**

In `config.py`, in the Teams Bot block after `teams_poll_interval: int = 5`:

```python
    teams_idle_poll_interval: int = 30        # outside business hours / weekends
    teams_business_hours_start_utc: int = 7   # fast polling from this UTC hour (inclusive), Mon-Fri...
    teams_business_hours_end_utc: int = 19    # ...until this UTC hour (exclusive)
    teams_messages_page_size: int = 5         # $top on the per-chat message fetch (Graph default: 20)
```

- [ ] **Step 4: Cap the page size**

In `process_new_messages`, change the messages URL:

```python
            messages_url = (
                f"{GRAPH_API}/me/chats/{chat_id}/messages"
                f"?$top={settings.teams_messages_page_size}"
            )
```

- [ ] **Step 4b: Follow `@odata.nextLink` on the chat list**

In `channels/teams/bot.py`, add a module constant after `PID_FILE`:

```python
# Graph pages /me/chats at 20 per page by default; 50 is the documented maximum.
# Fewer pages per cycle — and _get_all_pages follows @odata.nextLink for the rest.
_CHATS_PAGE_SIZE = 50
```

Add to `TeamsBot`, in the "Graph API helpers" section after `_send_message`:

```python
    def _get_all_pages(self, url):
        """GET a Graph collection, following @odata.nextLink until exhausted.

        Graph pages every collection. Reading only the first page of /me/chats
        silently stops polling chats past the first 20 — invisible at 5 users,
        a dropped-user bug at 30.
        """
        items = []
        while url:
            data = self._api_request(url)
            if not data:
                break
            items.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        return items
```

In `process_new_messages`, replace the two chat-list lines (`chats_data = self._api_request(url)` and `chats = chats_data.get("value", []) if chats_data else []`) with:

```python
        url = f"{GRAPH_API}/me/chats?$top={_CHATS_PAGE_SIZE}"
        chats = self._get_all_pages(url)
```

- [ ] **Step 5: Add the adaptive interval**

Add to `TeamsBot`, immediately above `process_new_messages`:

```python
    @staticmethod
    def _current_poll_interval(now):
        """Poll fast during the working week, slowly otherwise.

        The bot is a business-hours tool; polling every 5s around the clock
        spends roughly 70% of its Graph budget on hours nobody is asking.
        `now` must be timezone-aware UTC. The window is configured in UTC
        (default 07-19, i.e. 09/10-21/22 Kyiv) so it needs no tz database.
        """
        is_weekday = now.weekday() < 5
        is_business_hours = (
            settings.teams_business_hours_start_utc <= now.hour < settings.teams_business_hours_end_utc
        )
        if is_weekday and is_business_hours:
            return settings.teams_poll_interval
        return settings.teams_idle_poll_interval
```

- [ ] **Step 6: Use it in the main loop**

In `run()`, replace `time.sleep(settings.teams_poll_interval)`:

```python
                    time.sleep(self._current_poll_interval(datetime.now(timezone.utc)))
```

- [ ] **Step 7: Document the knobs**

`.env.example` has no Teams block at all today (CLAUDE.md claims it holds the full list). Append one at the end of the file:

```bash
# Teams Bot (delegated user account; the bot rotates the refresh token into
# channels/teams/data/refresh_token.json — .env is only the seed)
TEAMS_TENANT_ID=
TEAMS_CLIENT_ID=
TEAMS_CLIENT_SECRET=
TEAMS_REFRESH_TOKEN=

# Teams polling (hours are UTC; the fast window applies Mon-Fri)
TEAMS_POLL_INTERVAL=5                 # inside business hours
TEAMS_IDLE_POLL_INTERVAL=30           # nights and weekends
TEAMS_BUSINESS_HOURS_START_UTC=7
TEAMS_BUSINESS_HOURS_END_UTC=19
TEAMS_MESSAGES_PAGE_SIZE=5            # $top per chat (Graph default: 20)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_bot_polling.py tests/unit/test_bot_queue.py -v`
Expected: PASS (12 passed — 5 polling, 7 queue; the queue tests are present because `feat/graph-polling` is cut from the Task 3 branch)

- [ ] **Step 9: Commit**

```bash
git add config.py channels/teams/bot.py .env.example tests/unit/test_bot_polling.py
git commit -m "perf(teams): cap the message page, page the chat list, back off when idle

Every cycle pulled the default 20-message page with full HTML bodies for
every chat, around the clock. $top=5 plus a night/weekend interval cuts
the request budget substantially without touching detection latency
during working hours.

The chat list is now read to the end via @odata.nextLink; previously only
the first page (20 chats) was ever polled, so users past it were never
answered."
```

---

### Task 5: Collapse the per-chat N+1 with `lastMessagePreview`

The single biggest Graph win: `GET /me/chats?$expand=lastMessagePreview` carries each chat's most recent message inline (documented: `$expand` "currently supports members and lastMessagePreview"; the `chatMessageInfo` preview has `id`, `createdDateTime`, `from`, `body`, `isDeleted`, `messageType`), so one call replaces the current 31, with per-chat fetches only for chats that actually changed. `$orderby=lastMessagePreview/createdDateTime desc` is also documented, and puts the most recently active chats on page 1.

**What the docs do not settle** is whether the preview is populated and current for this tenant's chats, and whether `$expand`, `$orderby` and `$top=50` combine on one request. Step 1 settles that against live Graph before any code is written, and the task stops there if the answer is no. Body completeness does not matter: the skip is keyed on the preview's `id`, never on its text.

Two behaviours to understand before touching the loop:

- **After every bot reply, that chat is fetched once more.** The preview is then the bot's own message, which is not yet in `processed_messages`; the fetch runs, `_should_process_message` files it as `self_message`, and from the next cycle the chat is skipped. One extra call per reply is the correct price — see the next point.
- **Do not skip chats whose preview sender is the bot.** With the worker answering asynchronously, a user's second question can land *before* the bot's answer to their first, so the preview is the bot's message while an unprocessed question sits underneath it. A sender-based skip would drop that question until the user typed again. Skip on processed `id` only — and `continue`, never `break`: a chat whose fetch failed transiently on an earlier cycle can still hold an older unprocessed message.

> **Credential safety — read before Step 1.** Azure AD refresh tokens rotate on use. The bot prefers `channels/teams/data/refresh_token.json` (rotated) over `.env` (seed). Using the `.env` seed from a second machine can invalidate the token the running bot holds and take it offline. Run the probe **on the bot host, with the bot stopped**, so it uses and rotates the same token file the bot will resume with. Do not run it from a laptop against a live bot.

**Files:**
- Create: `scripts/probe_graph_preview.py` (verification tool; kept — it is how this gets re-checked if Graph changes)
- Modify: `channels/teams/bot.py` (`process_new_messages`)
- Test: `tests/unit/test_bot_polling.py` (extend)

**Interfaces:**
- Consumes: `TokenRefresher` from `channels/teams/auth.py`; `settings.teams_messages_page_size`, `_get_all_pages` and `_CHATS_PAGE_SIZE` from Task 4.
- Produces: no new public functions; `process_new_messages` gains an early-skip path.

- [ ] **Step 1: Write the verification probe**

Create `scripts/probe_graph_preview.py`:

```python
"""Does /me/chats?$expand=lastMessagePreview carry a usable message body?

Run ON THE BOT HOST WITH THE BOT STOPPED — refresh tokens rotate on use.

    PYTHONPATH=. python scripts/probe_graph_preview.py
"""

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from channels.teams.auth import TokenRefresher

GRAPH_API = "https://graph.microsoft.com/v1.0"
# The exact URL Task 5 Step 5 will use in production — probe the combination, not the parts.
CHATS_URL = (
    f"{GRAPH_API}/me/chats"
    "?$expand=lastMessagePreview"
    "&$orderby=lastMessagePreview/createdDateTime desc"
    "&$top=50"
)


def main():
    token = TokenRefresher().get_access_token()
    if not token:
        print("No access token — aborting.")
        return 1

    t0 = time.perf_counter()
    resp = requests.get(CHATS_URL, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"HTTP {resp.status_code} in {elapsed_ms:.0f} ms")
    if resp.status_code != 200:
        print(resp.text[:600])
        return 1

    payload = resp.json()
    chats = payload.get("value", [])
    print(f"chats on page 1: {len(chats)} | nextLink present: {'@odata.nextLink' in payload}")

    go = True
    for chat in chats[:10]:
        preview = chat.get("lastMessagePreview") or {}
        body = (preview.get("body") or {}).get("content", "")
        sender = preview.get("from") or {}
        from_user_id = (sender.get("user") or {}).get("id")
        from_app_id = (sender.get("application") or {}).get("id")
        print(json.dumps({
            "chat_id": (chat.get("id") or "")[:24],
            "preview_id": preview.get("id"),
            "createdDateTime": preview.get("createdDateTime"),
            "messageType": preview.get("messageType"),
            "from_user_id": from_user_id,
            "from_app_id": from_app_id,
            "body_len": len(body),
            "body_head": body[:120],
        }, indent=2))
        if not preview.get("id") or not preview.get("createdDateTime"):
            go = False
        # systemEventMessage previews have from=null by design; only real messages need a sender.
        if preview.get("messageType") == "message" and not (from_user_id or from_app_id):
            go = False

    print("\nGO if: every preview has id + createdDateTime, and every messageType=='message'")
    print("preview has a sender (from.user.id, or from.application.id for bot-sent messages).")
    print("Body completeness is irrelevant: the skip is keyed on id, not text.")
    print("Record the elapsed ms above — Graph latency was assumed (~250 ms) in the audit, never measured.")
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'}")
    return 0 if go else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the probe on the bot host and record the verdict**

```bash
# on the bot host, with the bot container stopped
docker compose -f docker-compose-remote.yml stop bot
PYTHONPATH=. python scripts/probe_graph_preview.py | tee /tmp/graph-preview-probe.txt
```

Expected: `HTTP 200 in <n> ms`, one row per chat, and `VERDICT: GO`.

**Decision gate.**
- `HTTP 400` → Graph rejected the parameter combination. Remove the `$orderby` line from `CHATS_URL` and rerun. If that passes, drop `$orderby` from Step 5's URL as well — pagination still covers every chat; ordering is only an optimisation. Record which variant passed in the commit message.
- `VERDICT: NO-GO` → **stop here**: mark this task abandoned in the plan, commit only `scripts/probe_graph_preview.py` with the probe output recorded in the commit message, and keep the Task 4 optimisations as the Graph outcome. Do not proceed to Step 3.
- Either way, note the measured Graph latency (`in <n> ms`) in the commit message — it replaces the audit's ~250 ms assumption.

- [ ] **Step 3: Write the failing test**

Append to `tests/unit/test_bot_polling.py`:

```python
def test_unchanged_chats_are_not_fetched(monkeypatch, pbot):
    """A chat whose preview is already in processed_messages costs no second call."""
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 5)
    pbot.processed_messages = {"seen-msg"}
    urls = []

    def fake_api(url, method="GET", json_data=None):
        urls.append(url)
        # Match the chat list on "not a messages URL", not on "$expand=..." —
        # otherwise the pre-implementation run returns no chats and this test
        # passes before Step 5 exists.
        if "/messages" in url:
            return {"value": []}
        return {"value": [{"id": "chatA",
                           "lastMessagePreview": {"id": "seen-msg"}}]}

    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_api_request", fake_api)
    pbot.process_new_messages()

    assert len(urls) == 1, urls          # the chat list only
    assert not any("/messages" in u for u in urls)
    assert "$expand=lastMessagePreview" in urls[0], urls[0]


def test_changed_chats_are_still_fetched(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 5)
    pbot.processed_messages = set()
    urls = []

    def fake_api(url, method="GET", json_data=None):
        urls.append(url)
        if "/messages" in url:
            return {"value": []}
        return {"value": [{"id": "chatA",
                           "lastMessagePreview": {"id": "brand-new"}}]}

    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_api_request", fake_api)
    pbot.process_new_messages()

    assert any("/messages" in u for u in urls), urls
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_bot_polling.py -k preview -v`
Expected: FAIL — `test_unchanged_chats_are_not_fetched` sees 2 URLs, because every chat is still fetched (and its chat-list URL carries no `$expand`). `test_changed_chats_are_still_fetched` already passes — it is the non-regression half.

- [ ] **Step 5: Skip unchanged chats**

In `process_new_messages`, request the preview on the chat list. This replaces the Task 4 URL; `_get_all_pages` still follows `@odata.nextLink`, and `requests` percent-encodes the space in `desc`:

```python
        url = (
            f"{GRAPH_API}/me/chats"
            f"?$expand=lastMessagePreview"
            f"&$orderby=lastMessagePreview/createdDateTime desc"
            f"&$top={_CHATS_PAGE_SIZE}"
        )
        chats = self._get_all_pages(url)
```

And inside the `for chat in chats:` loop, immediately after the `chat_id` guard:

```python
            # One call now tells us each chat's newest message. If we have already
            # handled it, the per-chat fetch is pure waste — skip it. This is what
            # keeps Graph volume flat as the user count grows.
            #
            # Skip on processed id ONLY — not on "sender is the bot": the worker
            # answers asynchronously, so a user's next question can sit under the
            # bot's newer answer and would be dropped. And `continue`, not `break`:
            # a chat whose fetch failed on an earlier cycle can hold an older
            # unprocessed message even though other chats' newer ones are processed.
            preview_id = safe_get_nested(chat, "lastMessagePreview", "id")
            if preview_id and preview_id in self.processed_messages:
                continue
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_bot_polling.py tests/unit/test_bot_queue.py tests/unit/test_bot_routing.py -v`
Expected: PASS (all green)

- [ ] **Step 7: Commit**

```bash
git add scripts/probe_graph_preview.py channels/teams/bot.py tests/unit/test_bot_polling.py
git commit -m "perf(teams): skip unchanged chats using lastMessagePreview

One expanded chat-list call (ordered by last activity, 50 per page) now
reports each chat's newest message, so the per-chat fetch only runs where
something changed. Graph volume stops scaling with headcount. Verified
against live Graph with scripts/probe_graph_preview.py before implementing
(Graph latency measured: <n> ms)."
```

---

### Task 6: Measure the client-construction gap — and do NOT cache the LLM client

The router measured **650 ms** over raw HTTP but **1.8 s** in-pipeline. Both `classify_message` (`rag/router.py:82`) and `build_agent` (`rag/agent.py:183`) call `get_llm()`, which constructs a fresh LlamaIndex client every time. ~1.2 s is unaccounted for.

The obvious fix — `lru_cache` on `get_llm` — is **wrong here, and was reproduced failing on 2026-09-16.** llama-index's `Ollama` creates its `httpx.AsyncClient` once (the `async_client` property) and reuses it for the object's lifetime, while `_run_rag` runs every request in a fresh loop via `asyncio.run()`. A pooled connection created under a closed loop fails on the next loop:

```text
httpx 0.28.1 | httpcore 1.0.9 — one AsyncClient, three successive asyncio.run() loops
run 0: HTTP 200
run 1: ERROR RuntimeError: Event loop is closed
run 2: HTTP 200
```

`RuntimeError` is not in `_TRANSIENT_TYPES`, so in production this becomes `{"escalation": {"needed": true, "reason": "Event loop is closed"}}` — a false content escalation that leaks a raw error, on roughly every other question. Constructing a fresh client per request is what keeps the per-request loop safe. The only correct way to share a client is to give the worker thread one persistent event loop; that is a design change with its own plan, and there is no evidence yet that it would buy anything:

Construction is almost certainly not the gap. `Ollama(...)` is a pydantic object; nothing connects until the first call. A likelier explanation is server-side — Ollama keeps one prompt cache per slot, and with `NUM_PARALLEL=1` the router's system prompt is evicted by every agent turn in between, so in-pipeline the router re-prefills its whole prompt while the raw probe (same prompt, back to back) hit the cache. Testing that costs GPU time and is out of scope here.

This task therefore **measures** construction so the number is on record, adds a guard test that fails if anyone reintroduces a cache, and writes the gotcha down.

**Files:**
- Create: `scripts/bench_llm_construction.py`
- Modify: `rag/agent.py` (`get_llm` docstring only)
- Modify: `CLAUDE.md` (one gotcha row)
- Test: `tests/unit/test_llm_config.py` (extend)

**Interfaces:**
- Consumes: `get_llm()` from `rag/agent.py` (unchanged signature).
- Produces: nothing new. `get_llm()` keeps returning a fresh client on every call — that is now an asserted property.

- [ ] **Step 1: Write and run the benchmark, and record the number**

Create `scripts/bench_llm_construction.py`:

```python
"""How much of the router's in-pipeline latency is client construction?

Offline: constructs clients, makes no LLM calls.

    PYTHONPATH=. python scripts/bench_llm_construction.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()

from rag.agent import get_llm

build = []
for _ in range(10):
    t0 = time.perf_counter()
    get_llm()
    build.append(time.perf_counter() - t0)

print(f"get_llm() construction: median {statistics.median(build) * 1000:7.1f} ms "
      f"| min {min(build) * 1000:.1f} | max {max(build) * 1000:.1f}")
print("Router call measured at 650 ms raw HTTP vs 1.8 s in-pipeline;")
print("this figure is how much of that ~1.2 s gap construction explains.")
print("Do NOT fix it with lru_cache(get_llm) — see the gotcha in CLAUDE.md.")
```

Run: `PYTHONPATH=. .venv/bin/python scripts/bench_llm_construction.py`
Expected order of magnitude: single-digit milliseconds. Record the median in the commit message at Step 5.

- [ ] **Step 2: Write the guard test**

Append to `tests/unit/test_llm_config.py`:

```python
def test_get_llm_builds_a_fresh_client_per_call(monkeypatch):
    # _run_rag runs each request in a new asyncio.run() loop. llama-index's Ollama
    # caches its httpx.AsyncClient on first use, and a pooled connection from a
    # closed loop raises "RuntimeError: Event loop is closed" on the next one
    # (reproduced 2026-09-16). A fresh client per call is what keeps that safe.
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    assert get_llm() is not get_llm()
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_llm_config.py -v`
Expected: PASS (4 passed). This test passes on the current code by design — it exists to fail the day someone adds a cache.

- [ ] **Step 4: Write the reason into the code and the gotchas**

In `rag/agent.py`, give `get_llm` a docstring (it has none). Only the docstring changes; the body stays exactly as Tasks 1 and 2 left it:

```python
def get_llm(model: str | None = None):
    """Build a fresh LLM client. Deliberately NOT cached.

    _run_rag runs each request under its own asyncio.run() loop. llama-index's
    Ollama creates its httpx.AsyncClient once and reuses it, and a pooled
    connection from a closed loop fails on the next loop with
    "RuntimeError: Event loop is closed" (reproduced 2026-09-16) — which is not
    a transient error, so it would surface as a false content escalation.
    Construction is cheap (see scripts/bench_llm_construction.py). If a shared
    client is ever wanted, the worker thread must own one persistent event loop
    first.
    """
```

Add one row to the "Gotchas / Lessons Learned" table in `CLAUDE.md`:

```markdown
| Caching the LLM client (`lru_cache(get_llm)`, module-level `Ollama`) breaks every other question with `RuntimeError: Event loop is closed` | `_run_rag` uses `asyncio.run()` per request; llama-index's `Ollama` reuses one `httpx.AsyncClient` across loops. Not transient → false content escalation. Build a fresh client per call (guarded by `test_get_llm_builds_a_fresh_client_per_call`), or give the worker a persistent loop first. |
```

- [ ] **Step 5: Commit**

```bash
git add rag/agent.py scripts/bench_llm_construction.py tests/unit/test_llm_config.py CLAUDE.md
git commit -m "docs(llm): measure client construction; guard against caching get_llm

The router measured 650ms over raw HTTP but 1.8s in-pipeline. Measured
construction cost of get_llm(): <median from Step 1> ms - construction is
not the gap. Caching the client is also unsafe: _run_rag uses asyncio.run()
per request and llama-index's Ollama reuses one httpx.AsyncClient across
loops, which fails with 'Event loop is closed' (reproduced). A guard test
now fails if a cache is reintroduced."
```

---

## Verification after all tasks

Run the whole list once after the last branch merges. The burst and restart checks also close out the Task 3 branch on their own; the CLAUDE.md edits belong in whichever branch changes the behaviour they describe, so `main` never documents code it does not have.

- [ ] **Full suite:** `PYTHONPATH=. .venv/bin/pytest -v` — unit tests pass; `docs/` and `live/` auto-skip without a corpus or an LLM.
- [ ] **Live smoke:** with the remote profile in `.env`, run `PYTHONPATH=. .venv/bin/python scripts/test_query.py -q "Can I install a free screen recording tool?"` and confirm a cited answer comes back.
- [ ] **Burst behaviour:** send 3 questions from 2 different Teams accounts within ~5 seconds. Every sender must receive "Got your message." within ~2 seconds, and answers must arrive one at a time in order. This is the whole point of Task 3 — check it by hand.
- [ ] **Restart safety:** send a question, stop the bot container within ~2 seconds (before the answer sends), restart it. The question must be answered after restart rather than silently dropped.
- [ ] **Worker liveness:** after the burst test, the container log shows the `Workers: 1 ...` banner line and no `rag-worker thread died` line.
- [ ] **CLAUDE.md:** the architecture section still describes `_send_reply`. Update it to the `_handle_inbound` / `_answer` / single-worker split; add the worker-count constraint and "never cache the LLM client across requests" to "Critical Constraints — never violate"; note on the Channels line that the Graph poll follows `@odata.nextLink`; and fix the Config line's claim that `.env.example` holds the full list (it now does, once Task 4 lands).

## Explicitly out of scope

- **Fixing the `search_policies` module globals.** Required only for a second worker, which this plan does not introduce. When it is needed: `ToolCallResult` exposes `tool_output`, and `search_policies` already returns its `POLICY_SEARCH_UNAVAILABLE` sentinel in-band, so consuming `handler.stream_events()` removes the side channel entirely.
- **`OLLAMA_NUM_PARALLEL`.** Server-side on a host this project does not own, and not needed at this scale.
- **Graph change notifications / a Bot Framework app.** The right answer at 100+ users; needs an endpoint Microsoft can reach, which the current private-network design deliberately avoids.
- **Shortening answers.** Decode is 78% of latency, but the system prompt's "cite ALL relevant sources with verbatim quotes" is a compliance requirement. Changing it is a product decision, and it should be measured with the existing eval harness, not folded into infrastructure work.
- **Sharing one LLM client across requests.** Requires the worker thread to own a single persistent event loop (`asyncio.new_event_loop()` once; `loop.run_until_complete(...)` per request) so llama-index's cached `httpx.AsyncClient` never outlives its loop. Only worth designing if Task 6's measurement shows construction is a material cost — it is expected not to be.
- **Testing the prompt-cache-eviction hypothesis** for the router's extra ~1.2 s (router call → agent call → router call, timed against the live stack). Needs GPU time; a separate, small investigation.
