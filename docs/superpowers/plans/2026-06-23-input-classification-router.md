# Pre-retrieval Input Classification Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify every inbound Teams message before retrieval (greeting / in_scope / out_of_scope / unintelligible) so only in-scope policy questions reach the RAG pipeline.

**Architecture:** A single temperature-0 LLM classification call (`rag/router.py`) sits in front of `_run_rag` in `channels/teams/bot.py`. Greetings, out-of-scope, and unintelligible messages get fixed deterministic replies and never search. The classifier can only *add* short-circuits for high-confidence non-questions; uncertainty (confidence below a floor) or any failure resolves to IN_SCOPE, so a real question is never refused.

**Tech Stack:** Python, llama-index (`Ollama`/`OpenAILike` via existing `get_llm()`, `ChatMessage`/`MessageRole`), pydantic, pydantic-settings, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-23-input-classification-router-design.md`

## Global Constraints

- `temperature=0.0` for the classifier (project-wide determinism — uses the existing `settings.llm_temperature=0.0` via `get_llm()`).
- The router can NEVER refuse a real question: `confidence < ROUTER_CONFIDENCE_FLOOR` → IN_SCOPE; any classifier exception or unparseable output → IN_SCOPE (`fallback=True`).
- Four categories only: `in_scope`, `greeting`, `out_of_scope`, `unintelligible`. No `escalation_request` (dropped for v1).
- Everything tuneable: scalars/flags in `.env`/`config.py`; multi-line text (classifier prompt, user-facing messages) as editable module constants — a multi-line value in `.env` would hit the documented `\n`-sent-literally gotcha.
- Imports at top of module — EXCEPT `channels/teams/bot.py` defers `from rag.router import ...` and `from rag.observability import ...` inside `_send_reply` (the `init_observability()`-first rule; matches the existing `_run_rag` deferred imports).
- Teams HTML uses only `<p> <b> <i> <ul>/<li> <hr> <code>` — never `<div>`/inline styles.
- Non-`in_scope` replies set NO pending rating and create NO feedback row (same treatment as the existing "unavailable" outcome).
- The classifier uses a plain `llm.chat()` completion with NO tools; a JSON-only instruction + tolerant parse is correct here (unlike the agent, where `response_format=json_object` kills tool-call emission).

---

### Task 1: Config knobs

**Files:**
- Modify: `config.py` (add three settings to `class Settings`, after the Agent block at `config.py:75-76`)
- Modify: `.env.example` (document the three knobs)
- Test: `tests/unit/test_router_config.py` (create)

**Interfaces:**
- Produces: `settings.router_enabled: bool` (default `True`), `settings.router_llm_model: str` (default `""`), `settings.router_confidence_floor: float` (default `0.6`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_router_config.py`:

```python
from config import Settings


def test_router_defaults():
    s = Settings(_env_file=None)  # ignore local .env; assert the code defaults
    assert s.router_enabled is True
    assert s.router_llm_model == ""
    assert s.router_confidence_floor == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_router_config.py -v`
Expected: FAIL — `AttributeError`/`ValidationError` (fields don't exist yet).

- [ ] **Step 3: Implement — add the settings**

In `config.py`, immediately after the Agent settings (`agent_timeout: int = 120`), add:

```python
    # Router (pre-retrieval classification)
    router_enabled: bool = True
    router_llm_model: str = ""            # classifier model override; empty -> main LLM
    router_confidence_floor: float = 0.6  # below this -> safe default IN_SCOPE
```

In `.env.example`, after the `# Agent` block (`AGENT_TIMEOUT=120`), add:

```bash
# Router (pre-retrieval classification)
ROUTER_ENABLED=true
ROUTER_LLM_MODEL=
ROUTER_CONFIDENCE_FLOOR=0.6
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/unit/test_router_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config.py .env.example tests/unit/test_router_config.py
git commit -m "feat(router): add ROUTER_ENABLED / ROUTER_LLM_MODEL / ROUTER_CONFIDENCE_FLOOR config"
```

---

### Task 2: Router core (`rag/router.py`) + `get_llm` model override

**Files:**
- Create: `rag/router.py`
- Modify: `rag/agent.py` (`get_llm` gains an optional `model` parameter — `rag/agent.py:155-178`)
- Test: `tests/unit/test_router.py` (create)

**Interfaces:**
- Consumes: `settings.router_llm_model` (Task 1); `rag.resilience.retry_transient` (existing); `rag.response._extract_json` (existing tolerant JSON extractor).
- Produces:
  - `Category(str, Enum)` with members `IN_SCOPE="in_scope"`, `GREETING="greeting"`, `OUT_OF_SCOPE="out_of_scope"`, `UNINTELLIGIBLE="unintelligible"`.
  - `RouterDecision(BaseModel)` with `category: Category`, `confidence: float`, `fallback: bool = False`.
  - `classify_message(text: str) -> RouterDecision` — never raises; failure/unparseable → `RouterDecision(IN_SCOPE, 0.0, fallback=True)`.
  - `resolve(decision: RouterDecision, floor: float) -> Category` — pure.
  - `ROUTER_SYSTEM_PROMPT: str` — editable constant.
  - `rag.agent.get_llm(model: str | None = None)` — `None` keeps current behavior.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_router.py`:

```python
import httpx
import pytest

from rag.router import (
    Category,
    RouterDecision,
    classify_message,
    resolve,
    _parse_decision,
)


# --- resolve (pure) ---

@pytest.mark.parametrize("cat", list(Category))
def test_resolve_below_floor_is_in_scope(cat):
    d = RouterDecision(category=cat, confidence=0.4)
    assert resolve(d, 0.6) == Category.IN_SCOPE


def test_resolve_fallback_is_in_scope_regardless_of_confidence():
    d = RouterDecision(category=Category.OUT_OF_SCOPE, confidence=0.99, fallback=True)
    assert resolve(d, 0.6) == Category.IN_SCOPE


def test_resolve_at_or_above_floor_passes_category_through():
    d = RouterDecision(category=Category.GREETING, confidence=0.6)
    assert resolve(d, 0.6) == Category.GREETING
    d2 = RouterDecision(category=Category.OUT_OF_SCOPE, confidence=0.9)
    assert resolve(d2, 0.6) == Category.OUT_OF_SCOPE


# --- _parse_decision ---

def test_parse_plain_json():
    d = _parse_decision('{"category": "greeting", "confidence": 0.9}')
    assert d.category == Category.GREETING and d.confidence == 0.9


def test_parse_fenced_json():
    d = _parse_decision('```json\n{"category": "out_of_scope", "confidence": 0.8}\n```')
    assert d.category == Category.OUT_OF_SCOPE


def test_parse_unknown_category_is_none():
    assert _parse_decision('{"category": "banana", "confidence": 0.9}') is None


def test_parse_missing_confidence_is_none():
    assert _parse_decision('{"category": "greeting"}') is None


def test_parse_non_json_is_none():
    assert _parse_decision("I think this is a greeting") is None


# --- classify_message (LLM mocked) ---

class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeResp:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeLLM:
    def __init__(self, content=None, exc=None):
        self._content, self._exc = content, exc

    def chat(self, messages):
        if self._exc is not None:
            raise self._exc
        return _FakeResp(self._content)


def test_classify_success_returns_parsed_decision(monkeypatch):
    monkeypatch.setattr(
        "rag.router.get_llm",
        lambda model=None: _FakeLLM(content='{"category": "greeting", "confidence": 0.95}'),
    )
    d = classify_message("hi there")
    assert d.category == Category.GREETING and d.confidence == 0.95 and d.fallback is False


def test_classify_unparseable_falls_back_to_in_scope(monkeypatch):
    monkeypatch.setattr(
        "rag.router.get_llm", lambda model=None: _FakeLLM(content="not json at all")
    )
    d = classify_message("hi there")
    assert d.category == Category.IN_SCOPE and d.fallback is True


def test_classify_llm_failure_falls_back_to_in_scope(monkeypatch):
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "rag.router.get_llm",
        lambda model=None: _FakeLLM(exc=httpx.ConnectError("llm down")),
    )
    d = classify_message("Can I install software?")
    assert d.category == Category.IN_SCOPE and d.fallback is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.router'`.

- [ ] **Step 3: Add the `get_llm` model override in `rag/agent.py`**

Replace the `get_llm` signature and the two `model=` arguments (`rag/agent.py:155-178`):

```python
def get_llm(model: str | None = None):
    if settings.llm_backend == "openai-compatible":
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=model or settings.openai_model,
            api_base=settings.openai_api_base,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            request_timeout=float(settings.active_request_timeout),
            is_chat_model=True,
            is_function_calling_model=True,
        )
    else:
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=model or settings.llm_model,
            base_url=settings.active_ollama_url,
            request_timeout=float(settings.active_request_timeout),
            temperature=settings.llm_temperature,
            thinking=False,
            additional_kwargs={"num_predict": 4096, "num_ctx": 8192},
        )
```

- [ ] **Step 4: Implement `rag/router.py`**

Create `rag/router.py`:

```python
"""Pre-retrieval input classification: greeting / in_scope / out_of_scope / unintelligible.

A single temperature-0 LLM call decides whether a message reaches policy search. The
classifier can only ADD short-circuits for high-confidence non-questions; it can never
refuse a real question — uncertainty or failure resolves to IN_SCOPE (see resolve()).
"""

import json
from enum import Enum

from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, ValidationError

from config import settings
from rag.agent import get_llm
from rag.resilience import retry_transient
from rag.response import _extract_json  # reuse the tolerant JSON extractor (DRY)


class Category(str, Enum):
    IN_SCOPE = "in_scope"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"
    UNINTELLIGIBLE = "unintelligible"


class RouterDecision(BaseModel):
    category: Category
    confidence: float
    fallback: bool = False  # True when set by the failure path, not the model


# Editable tuning surface for classifier behavior (see spec "Tuning surface").
ROUTER_SYSTEM_PROMPT = """\
You are an input classifier for an internal Compliance Policy assistant.
Classify the user's message into EXACTLY ONE category and output ONLY a JSON object.

Categories:
- "in_scope": a question an internal company compliance or policy document could plausibly
  answer (security, HR, data handling, device/access, conduct, travel, expenses, etc.).
  When unsure whether a message is a real policy question, choose "in_scope".
- "greeting": a greeting, pleasantry, or small talk with no question
  (e.g. "hi", "hello", "good morning", "thanks", "how are you").
- "out_of_scope": an intelligible request or question that company policy would NOT cover
  (e.g. "order me a pizza", "what's the weather", "who is Sarah Connor", general knowledge).
- "unintelligible": text that cannot be read as a meaningful message — random characters,
  gibberish, or text typed with the wrong keyboard layout (e.g. Cyrillic characters that are
  English words typed on a Ukrainian/Russian layout like "црфе ші").

Output EXACTLY one JSON object and nothing else:
{"category": "in_scope|greeting|out_of_scope|unintelligible", "confidence": <number 0..1>}

confidence is your certainty in the chosen category; use a low value when the message is ambiguous."""

_FALLBACK = RouterDecision(category=Category.IN_SCOPE, confidence=0.0, fallback=True)


def _parse_decision(raw: str) -> RouterDecision | None:
    """Parse the classifier's JSON output; return None if unparseable or invalid."""
    json_str = _extract_json(raw)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
        return RouterDecision(category=data["category"], confidence=float(data["confidence"]))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, ValidationError):
        return None


def classify_message(text: str) -> RouterDecision:
    """Classify a message. Never raises; any failure/unparseable output -> IN_SCOPE fallback."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=ROUTER_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=text),
    ]
    try:
        llm = get_llm(settings.router_llm_model or None)
        response = retry_transient(lambda: llm.chat(messages))
        decision = _parse_decision(str(response.message.content))
        return decision if decision is not None else _FALLBACK
    except Exception:
        # Fail safe to search: a classifier problem must never block a real question.
        return _FALLBACK


def resolve(decision: RouterDecision, floor: float) -> Category:
    """Apply the safe-default bias: failure or low confidence -> IN_SCOPE."""
    if decision.fallback:
        return Category.IN_SCOPE
    if decision.confidence < floor:
        return Category.IN_SCOPE
    return decision.category
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_router.py -v`
Expected: PASS (all).

- [ ] **Step 6: Verify `get_llm` change didn't break the agent**

Run: `PYTHONPATH=. python -c "from rag.agent import build_agent, get_llm; get_llm(); build_agent(); print('agent OK')"`
Expected: prints `agent OK` (no TypeError from the new signature).

- [ ] **Step 7: Commit**

```bash
git add rag/router.py rag/agent.py tests/unit/test_router.py
git commit -m "feat(router): classifier core (Category, RouterDecision, classify_message, resolve)"
```

---

### Task 3: Phoenix classification signal

**Files:**
- Modify: `rag/observability.py` (add `record_classification`, after `record_infra_unavailable` at `rag/observability.py:93-104`)
- Test: `tests/unit/test_observability.py` (create)

**Interfaces:**
- Produces: `record_classification(category: str, confidence: float, fallback: bool) -> None` — opens a `classification` span with attributes `router_category`, `router_confidence`, `router_fallback`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_observability.py`:

```python
import rag.observability as obs


def test_record_classification_no_raise_when_phoenix_disabled(monkeypatch):
    # phoenix disabled -> get_tracer() returns a no-op tracer; the call must not raise.
    monkeypatch.setattr(obs.settings, "phoenix_enabled", False)
    assert obs.record_classification("greeting", 0.95, False) is None
    assert obs.record_classification("in_scope", 0.0, True) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_observability.py -v`
Expected: FAIL — `AttributeError: module 'rag.observability' has no attribute 'record_classification'`.

- [ ] **Step 3: Implement**

In `rag/observability.py`, after `record_infra_unavailable`, add:

```python
def record_classification(category: str, confidence: float, fallback: bool) -> None:
    """Emit a Phoenix span for a pre-retrieval classification decision.

    category: the resolved Category value acted on
              ("in_scope" | "greeting" | "out_of_scope" | "unintelligible").
    fallback: True when the safe default (IN_SCOPE) overrode the model or the classifier failed.
    Makes the classification distribution and safe-default fallback rate queryable in Phoenix.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("classification") as span:
        span.set_attribute("router_category", category)
        span.set_attribute("router_confidence", confidence)
        span.set_attribute("router_fallback", fallback)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/unit/test_observability.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/observability.py tests/unit/test_observability.py
git commit -m "feat(router): record_classification Phoenix span"
```

---

### Task 4: Renderer messages (out-of-scope + unintelligible)

**Files:**
- Modify: `channels/teams/renderer.py` (add constants + render fns next to `UNAVAILABLE_HTML`/`render_unavailable` at `channels/teams/renderer.py:29-40`)
- Test: `tests/unit/test_renderer.py` (append tests)

**Interfaces:**
- Produces: `OUT_OF_SCOPE_HTML`, `UNINTELLIGIBLE_HTML` constants; `render_out_of_scope() -> str`, `render_unintelligible() -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_renderer.py`:

```python
def test_render_out_of_scope_is_safe_html():
    from channels.teams.renderer import render_out_of_scope
    html = render_out_of_scope()
    assert "compan" in html.lower()  # mentions company policies
    assert "<div" not in html  # Teams-safe tags only


def test_render_unintelligible_is_safe_html():
    from channels.teams.renderer import render_unintelligible
    html = render_unintelligible()
    assert "retype" in html.lower() or "keyboard" in html.lower()
    assert "<div" not in html  # Teams-safe tags only
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_renderer.py -v`
Expected: FAIL — `ImportError: cannot import name 'render_out_of_scope'`.

- [ ] **Step 3: Implement**

In `channels/teams/renderer.py`, after `render_unavailable()` (line 40), add:

```python
# Shown when the router classifies a message as off-topic (ROUTER-3). Editable: reword
# freely; must stay valid Teams-limited HTML.
OUT_OF_SCOPE_HTML = (
    "<p><b>I can only answer questions about company policies.</b></p>"
    "<p>Ask me about a policy and I'll find the relevant section and clause.</p>"
)

# Shown when the router can't read the message (gibberish / wrong keyboard layout). Editable.
UNINTELLIGIBLE_HTML = (
    "<p><b>I couldn't read that.</b></p>"
    "<p>It may have been typed with a different keyboard layout. "
    "Please retype your question.</p>"
)


def render_out_of_scope() -> str:
    """Render the out-of-scope redirect message."""
    return OUT_OF_SCOPE_HTML


def render_unintelligible() -> str:
    """Render the unintelligible-input retype prompt."""
    return UNINTELLIGIBLE_HTML
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_renderer.py -v`
Expected: PASS (all, including the pre-existing renderer tests).

- [ ] **Step 5: Commit**

```bash
git add channels/teams/renderer.py tests/unit/test_renderer.py
git commit -m "feat(router): out-of-scope + unintelligible Teams messages"
```

---

### Task 5: Wire the router into the bot

**Files:**
- Modify: `channels/teams/bot.py` (renderer import block at `channels/teams/bot.py:15-24`; `_send_reply` at `channels/teams/bot.py:200-265`)
- Test: `tests/unit/test_bot_routing.py` (create)

**Interfaces:**
- Consumes: `settings.router_enabled`, `settings.router_confidence_floor` (Task 1); `rag.router.classify_message`, `rag.router.resolve`, `rag.router.Category` (Task 2); `rag.observability.record_classification` (Task 3); `channels.teams.renderer.render_out_of_scope`, `render_unintelligible`, `WELCOME_HTML` (Task 4 / existing).
- Produces: classification gate in `_send_reply` between pending-state clearing and `LOADING_HTML`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_bot_routing.py`:

```python
import pytest

import channels.teams.bot as bot
import rag.router as router
from rag.router import RouterDecision, Category


@pytest.fixture
def teams_bot(monkeypatch):
    """A TeamsBot with network + RAG mocked; records every HTML it 'sends'."""
    b = bot.TeamsBot(token_refresher=object())
    sent = []
    monkeypatch.setattr(b, "_send_message", lambda chat_id, text, content_type="html": sent.append(text) or True)
    bot._pending_ratings.clear()
    b._sent = sent
    return b


def _force(monkeypatch, category, confidence=0.95):
    monkeypatch.setattr(router, "classify_message",
                        lambda text: RouterDecision(category=category, confidence=confidence))


def test_greeting_replies_welcome_no_search(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.GREETING)
    called = {"rag": False}
    monkeypatch.setattr(bot, "_run_rag", lambda q: called.__setitem__("rag", True) or {})
    teams_bot._send_reply("chat1", "hello")
    assert called["rag"] is False
    assert any("Compliance Policy Assistant" in h for h in teams_bot._sent)  # WELCOME_HTML
    assert "chat1" not in bot._pending_ratings


def test_out_of_scope_replies_redirect_no_search(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.OUT_OF_SCOPE)
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("must not search"))
    teams_bot._send_reply("chat1", "order me a pizza")
    assert any("only answer questions about company policies" in h for h in teams_bot._sent)
    assert "chat1" not in bot._pending_ratings


def test_unintelligible_replies_retype_no_search(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.UNINTELLIGIBLE)
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("must not search"))
    teams_bot._send_reply("chat1", "црфе ші")
    assert any("retype" in h.lower() for h in teams_bot._sent)
    assert "chat1" not in bot._pending_ratings


def test_in_scope_runs_rag_and_prompts_rating(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.IN_SCOPE)
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: {"answer": "See AUP.", "citations": [], "escalation": {"needed": False}})
    teams_bot._send_reply("chat1", "Can I install software?")
    assert any("Searching compliance policies" in h for h in teams_bot._sent)  # LOADING_HTML
    assert "chat1" in bot._pending_ratings  # rating prompt stored


def test_low_confidence_safe_default_searches(monkeypatch, teams_bot):
    # OUT_OF_SCOPE but below floor -> resolve() forces IN_SCOPE -> search runs.
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    monkeypatch.setattr(bot.settings, "router_confidence_floor", 0.6)
    _force(monkeypatch, Category.OUT_OF_SCOPE, confidence=0.3)
    called = {"rag": False}
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: called.__setitem__("rag", True) or {"answer": "x", "citations": [], "escalation": {"needed": False}})
    teams_bot._send_reply("chat1", "ambiguous thing")
    assert called["rag"] is True
    assert not any("only answer questions about company policies" in h for h in teams_bot._sent)


def test_router_disabled_bypasses_classifier(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(router, "classify_message", lambda text: pytest.fail("classifier must not run"))
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: {"answer": "x", "citations": [], "escalation": {"needed": False}})
    teams_bot._send_reply("chat1", "hello")  # would be a greeting, but router off -> search
    assert "chat1" in bot._pending_ratings
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_bot_routing.py -v`
Expected: FAIL — greeting/out-of-scope/unintelligible tests fail because the gate doesn't exist yet (every message currently searches).

- [ ] **Step 3: Implement — add the renderer imports**

In `channels/teams/bot.py`, extend the renderer import block (`channels/teams/bot.py:15-24`) to add the two new render functions:

```python
from channels.teams.renderer import (
    LOADING_HTML,
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

- [ ] **Step 4: Implement — insert the classification gate**

In `_send_reply`, between `_pending_ratings.pop(chat_id, None)` (the "Not a rating" line, `channels/teams/bot.py:230`) and `# Show loading indicator`, insert:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_bot_routing.py -v`
Expected: PASS (all six).

- [ ] **Step 6: Regression — full unit suite**

Run: `PYTHONPATH=. pytest -m "not corpus and not live_llm" -q`
Expected: all pass (the pre-existing suite plus the new tests).

- [ ] **Step 7: Commit**

```bash
git add channels/teams/bot.py tests/unit/test_bot_routing.py
git commit -m "feat(router): gate _send_reply on classification before retrieval"
```

---

### Task 6: Live classifier-accuracy test (auto-skips without an LLM)

**Files:**
- Create: `tests/_llm.py` (LLM-reachability probe)
- Create: `tests/live/__init__.py`
- Create: `tests/live/test_router_classification.py`
- Modify: `pytest.ini` (register the `live_llm` marker)

**Interfaces:**
- Consumes: `rag.router.classify_message` (Task 2); `settings.active_ollama_url` / `settings.openai_api_base` (existing).
- Produces: `tests._llm.llm_reachable() -> bool`; the `live_llm` pytest marker.

- [ ] **Step 1: Register the marker in `pytest.ini`**

Add under `markers =`:

```ini
    live_llm: requires a reachable LLM (auto-skipped when absent)
```

- [ ] **Step 2: Create the reachability probe**

Create `tests/_llm.py`:

```python
"""Probe whether a classification LLM is reachable, so live tests can auto-skip
(like the docx-corpus tests) on machines/CI without the local model stack."""
import httpx

from config import settings


def llm_reachable() -> bool:
    url = settings.active_ollama_url if settings.llm_backend == "ollama" else settings.openai_api_base
    try:
        httpx.get(url, timeout=2.0)  # any HTTP response (even 404) means it's up
        return True
    except Exception:
        return False
```

- [ ] **Step 3: Write the live accuracy test**

Create `tests/live/__init__.py` (empty), then `tests/live/test_router_classification.py`:

```python
import pytest

from tests._llm import llm_reachable

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(not llm_reachable(), reason="no reachable LLM (local-only)"),
]

CASES = [
    ("hi", "greeting"),
    ("hello there", "greeting"),
    ("thanks!", "greeting"),
    ("Can I install software on my work laptop?", "in_scope"),
    ("What is our policy on remote work?", "in_scope"),
    ("How many vacation days do I get?", "in_scope"),
    ("order me a pizza", "out_of_scope"),
    ("who is Sarah Connor", "out_of_scope"),
    ("what is the weather today", "out_of_scope"),
    ("црфе ші ърщдшсн", "unintelligible"),
    ("asdkj qweoiu zxcmnv", "unintelligible"),
]


def test_classifier_accuracy_on_labeled_set():
    """Tuning signal (not a hard gate): a live model is non-deterministic, so we assert
    aggregate accuracy and report misses rather than failing per case."""
    from rag.router import classify_message

    misses = []
    for text, expected in CASES:
        got = classify_message(text).category.value
        if got != expected:
            misses.append((text, expected, got))
    accuracy = (len(CASES) - len(misses)) / len(CASES)
    assert accuracy >= 0.8, f"classifier accuracy {accuracy:.0%} < 80%; misses={misses}"
```

- [ ] **Step 4: Run the live test (local stack up) and confirm skip behavior**

Run (local LLM reachable): `PYTHONPATH=. pytest tests/live/test_router_classification.py -v`
Expected: PASS (accuracy ≥ 80%). If it fails, the misses list shows which cases to tune in `ROUTER_SYSTEM_PROMPT` — adjust the prompt, do not lower the threshold without cause.

Confirm auto-skip: `PYTHONPATH=. pytest -m "not live_llm" -q` runs the rest without this test; on a machine with no LLM, the test reports as skipped rather than failing.

- [ ] **Step 5: Commit**

```bash
git add tests/_llm.py tests/live/ pytest.ini
git commit -m "test(router): live classifier-accuracy test (auto-skips without an LLM)"
```

---

## Final verification (after all tasks)

- [ ] Full offline suite: `PYTHONPATH=. pytest -m "not corpus and not live_llm" -q` → all pass.
- [ ] Live suite (local stack up): `PYTHONPATH=. pytest -m "live_llm" -q` → passes.
- [ ] Manual smoke in Teams (local bot): send "hi" (→ welcome, no search), "order me a pizza" (→ redirect), "црфе ші" (→ retype), and a real policy question (→ normal answer + rating). Confirm Phoenix shows `classification` spans with `router_category`/`router_confidence`/`router_fallback`.
- [ ] Update `CLAUDE.md`: add the router to the architecture map (`rag/router.py`), a one-line "Input classification" note in the search-flow section, and the three `ROUTER_*` knobs in the Config section.
