# Search-first retrieval, prompt/tool audit, relevance floor, grounding backstop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move retrieval in front of the agent so it can never be skipped, strip the agent to zero tools, cut the system prompt from 1,066 to 540 tokens, wire a relevance floor to the reranker score, and make it structurally impossible for an answer without a citation to reach a user.

**Architecture:** `search_policies` runs in code before the agent is built; its formatted sources are appended to the user message. The agent keeps no tools at all. Two sentinels (`POLICY_SEARCH_UNAVAILABLE`, `NO_RELEVANT_POLICY_FOUND`) short-circuit before any LLM call. After the run, a backstop in `bot.py` refuses to render an answer whose parse failed or whose citations are empty, escalating instead.

**Tech Stack:** Python 3, llama-index `AgentWorkflow` (FunctionAgent), Ollama (`qwen3.6:35b`), Qdrant, vLLM reranker, OpenTelemetry/Phoenix, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-search-first-design.md` — read it before Task 1. Every decision here argues from a decision there (D1-D8).

**Branch:** `feat/search-first` (already checked out; do not create a worktree).

## Global Constraints

Copied verbatim from the spec and `CLAUDE.md`. Every task's requirements implicitly include this section.

- `temperature=0.0` always (deterministic compliance answers).
- Agent must never answer without citing a retrieved chunk; citations come ONLY from retrieved chunk metadata.
- If `search_policies` returns `NO_RELEVANT_POLICY_FOUND` → escalate.
- `init_observability()` must run FIRST in every entry point (before any LlamaIndex/Ollama import).
- Never cache the LLM client across requests (no `lru_cache` on `get_llm`, no module-level client).
- **Exactly one worker thread** consumes `_work_q`. Never raise it.
- Do not change server-side configuration on `172.20.0.22` — it is a shared host this project does not own. Every change in this plan is client-side.
- `num_ctx` stays **4096** and `num_predict` stays **1024** (spec D6, D7). Do not raise either. `num_ctx >= 8192` is one of five conditions for a reproducible CUDA crash; a previous restoration attempt passed local probing and crashed in production.
- **Imports always at the top of the module** — never inside functions. The one deliberate exception already in the codebase is `channels/teams/bot.py:_run_rag`, whose imports are deferred so `init_observability()` runs before LlamaIndex loads. Keep that exception; do not add new ones.
- pytest `AttributeError`/assertion output touching `settings` embeds the full `Settings` repr containing real `.env` secrets (`hf_token`, `teams_client_secret`, `teams_refresh_token`, `smtp_password`). **Never** write `assert settings.foo < X` directly — bind to a local variable first, then assert on the local.
- The Teams renderer does **not** HTML-escape. Raw exception text or raw model output must never reach a rendered message.
- Never execute `scripts/probe_graph_preview.py` — it rotates a live credential.
- Tests must not touch the network or `172.20.0.22`. Mock every HTTP/Qdrant call.

---

## File Structure

**Created:**
- `rag/search_first.py` — the retrieval front-end. `prefetch()` runs the search and classifies the outcome; `compose_agent_input()` builds the agent's user message. Shared by `channels/teams/bot.py` and `eval/agent_wrapper.py` so eval cannot drift from production.
- `tests/unit/test_search_floor.py` — relevance floor behaviour.
- `tests/unit/test_search_first.py` — prefetch classification and input composition.
- `tests/unit/test_bot_search_first.py` — `_run_rag` short-circuits and the grounding backstop.

**Modified:**
- `config.py` — add `reranker_min_score`; delete `agent_max_iterations` and `escalation_ticket_prefix`; wire `agent_timeout`.
- `rag/observability.py` — add `record_floor_rejection()`.
- `rag/tools/search_policies.py` — sentinel constants; the relevance floor.
- `rag/agent.py` — new `SYSTEM_PROMPT`; `ALL_TOOLS = []`; corrected docstring; `agent_timeout` wired.
- `rag/tools/__init__.py` — drop the deleted tools and the dead duplicate `ALL_TOOLS`.
- `channels/teams/bot.py` — `_run_rag` uses `prefetch`; grounding backstop before rendering.
- `channels/teams/renderer.py` — delete the citation-less fallback.
- `eval/agent_wrapper.py`, `eval/evaluators.py`, `eval/run_experiment.py` — match production.
- `tests/unit/test_llm_config.py` — token budget guard.
- `.env.example`, `CLAUDE.md`.

**Deleted:**
- `rag/tools/get_section.py`, `rag/tools/escalate.py`.

---

### Task 1: Relevance floor wired to the reranker score

Implements spec **D8**. Today `min_confidence_score` is applied only when the reranker is OFF (`rag/tools/search_policies.py:81`) and production runs it ON, so the floor never fires.

**Files:**
- Modify: `config.py` (Reranker block, near `reranker_top_n`)
- Modify: `rag/observability.py` (after `record_classification`)
- Modify: `rag/tools/search_policies.py` (after Step 3, before Step 4)
- Modify: `.env.example`
- Test: `tests/unit/test_search_floor.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `settings.reranker_min_score: float`; `rag.observability.record_floor_rejection(top_score: float, threshold: float) -> None`. `search_policies` may now return `"NO_RELEVANT_POLICY_FOUND"` with `_last_search_results` **non-empty**.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_search_floor.py`:

```python
"""The relevance floor (spec D8).

min_confidence_score has never run in production: search_policies applies it only
when the reranker is OFF, and production runs it ON. These tests pin the replacement,
including the two ways it must NOT fire — disabled by default, and never on the
reranker's degraded fallback path, where a missing rerank_score means "the reranker
never ran", not "everything scored zero".
"""

import pytest

import rag.tools.search_policies as sp


@pytest.fixture
def reranked(monkeypatch):
    """search_policies with retrieval mocked, reranker ON, returning one scored hit."""
    monkeypatch.setattr(sp.settings, "bm25_enabled", False)
    monkeypatch.setattr(sp.settings, "reranker_enabled", True)

    class _Hit:
        score = 0.9
        payload = {
            "doc_title": "Acceptable Use Policy [Internal]",
            "doc_id": "acceptable-use-policy-internal",
            "section": "Corporate Workstation and Software Use",
            "clause": "Software Installation",
            "clause_number": "4.7",
            "text": "Team Members are forbidden to install any unlicensed software.",
        }

    monkeypatch.setattr("rag.embeddings.embed_query", lambda q: [0.0] * 768)
    monkeypatch.setattr("rag.vector_store.search_vectors", lambda v, top_k: [_Hit()])
    return sp


def _rerank_returning(score):
    def _fake(query, results, top_n):
        out = dict(results[0])
        if score is not None:
            out["rerank_score"] = score
        return [out]
    return _fake


def test_floor_rejects_a_low_scoring_top_result(reranked, monkeypatch):
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.5)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(0.10))

    assert reranked.search_policies("anything") == "NO_RELEVANT_POLICY_FOUND"


def test_a_rejected_search_still_reports_what_it_found(reranked, monkeypatch):
    """Deliberately unlike the other sentinel paths, which clear _last_search_results.

    Retrieval DID return candidates. The tier-1 retrieval evaluators and the
    threshold tuning both need to see what they were and how they scored, so
    clearing this list here would destroy the only evidence for choosing the
    threshold. Do not "fix" this into matching the other paths.
    """
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.5)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(0.10))

    reranked.search_policies("anything")

    assert len(reranked._last_search_results) == 1
    assert reranked._last_search_results[0]["rerank_score"] == 0.1


def test_floor_passes_a_high_scoring_top_result(reranked, monkeypatch):
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.5)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(0.80))

    assert "[Source 1]" in reranked.search_policies("anything")


def test_a_zero_threshold_disables_the_floor(reranked, monkeypatch):
    """Ships at 0.0; the live value is measured on the VM, not guessed here."""
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.0)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(0.0))

    assert "[Source 1]" in reranked.search_policies("anything")


def test_the_reranker_fallback_path_is_never_floored(reranked, monkeypatch):
    """A reranker outage must not become "no policy exists" for every question.

    rag/reranker.py falls back to the original ordering on error, and those results
    carry NO rerank_score (see its comment at the top_score span attribute). Reading
    a missing score as 0.0 would reject every question the moment the reranker went
    down — turning a degraded-but-working pipeline into a total outage.
    """
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.9)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(None))

    assert "[Source 1]" in reranked.search_policies("anything")


def test_the_default_threshold_ships_disabled():
    # Bound to a local first: a failing `assert settings.x == y` prints the whole
    # Settings repr, which carries live .env secrets.
    value = sp.settings.reranker_min_score
    assert value == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_search_floor.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'reranker_min_score'`

- [ ] **Step 3: Add the setting**

In `config.py`, in the Reranker block immediately after `reranker_candidates`:

```python
    # Relevance floor on the reranker's 0.0-1.0 score. 0.0 means OFF.
    # This is NOT a reuse of min_confidence_score: that one is cosine similarity
    # on the reranker-off path, this one is a reranker relevance probability, and
    # one knob for two scales would be a latent bug.
    # The live value is measured from Phoenix `reranker.top_score` on the VM and
    # set before merge — never guessed. CLAUDE.md records a "reranker scores
    # compressed" failure mode, so the distribution has to be looked at.
    reranker_min_score: float = 0.0
```

- [ ] **Step 4: Add the observability helper**

In `rag/observability.py`, after `record_classification`:

```python
def record_floor_rejection(top_score: float, threshold: float) -> None:
    """Emit a Phoenix span for a search rejected by the relevance floor.

    Makes the rejection rate countable in production, which is how the threshold
    gets tuned and how a too-high floor is caught before users feel it.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("retrieval_floor_rejected") as span:
        span.set_attribute("retrieval.floor_rejected", True)
        span.set_attribute("retrieval.top_score", top_score)
        span.set_attribute("retrieval.floor_threshold", threshold)
```

- [ ] **Step 5: Apply the floor**

In `rag/tools/search_policies.py`, add to the imports at the top of the module:

```python
from rag.observability import record_floor_rejection
```

Then, **after** Step 3 (`_last_search_results = [...]`) and **before** Step 4 (`return format_sources(results)`):

```python
    # Step 3b: Relevance floor (spec D8).
    # Guarded on the key being PRESENT, not on its value: the reranker's fallback
    # path returns results with no rerank_score at all, and defaulting that to 0.0
    # would reject every question the moment the reranker degraded.
    if settings.reranker_min_score > 0 and results and "rerank_score" in results[0]:
        top_score = results[0]["rerank_score"]
        if top_score < settings.reranker_min_score:
            record_floor_rejection(top_score, settings.reranker_min_score)
            # _last_search_results deliberately left populated — see
            # test_a_rejected_search_still_reports_what_it_found.
            return "NO_RELEVANT_POLICY_FOUND"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_search_floor.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Add the key to `.env.example`**

Next to the other `RERANKER_*` keys:

```bash
RERANKER_MIN_SCORE=0.0   # relevance floor on the reranker score; 0.0 = off. Measure before setting.
```

- [ ] **Step 8: Run the full unit suite**

Run: `PYTHONPATH=. pytest tests/unit -q`
Expected: PASS, no regressions.

- [ ] **Step 9: Commit**

```bash
git add config.py rag/observability.py rag/tools/search_policies.py tests/unit/test_search_floor.py .env.example
git commit -m "feat(retrieval): wire a relevance floor to the reranker score

min_confidence_score has never run in production — search_policies applies it
only when the reranker is OFF, and production runs it ON, so the sentinel needed
Qdrant to return zero rows. Retrieval always handed back reranker_top_n sources
however irrelevant.

Floor is guarded on rerank_score being present, not on its value: the reranker's
fallback path carries no score, and reading that as 0.0 would turn a reranker
outage into 'no policy exists' for every question.

Ships at 0.0 (off); the live value is measured from Phoenix on the VM."
```

---

### Task 2: `rag/search_first.py` — the retrieval front-end

Implements spec **D1**. Retrieval stops being the agent's decision.

**Files:**
- Modify: `rag/tools/search_policies.py` (extract the two sentinel strings into module constants)
- Create: `rag/search_first.py`
- Test: `tests/unit/test_search_first.py` (create)

**Interfaces:**
- Consumes: `search_policies(query, top_k)` and its sentinels from Task 1.
- Produces:
  - `rag.tools.search_policies.NO_MATCH: str` and `.UNAVAILABLE: str`
  - `rag.search_first.PrefetchResult` — a frozen dataclass with `status: str` (`"ok" | "unavailable" | "no_match"`) and `sources: str` (empty unless `ok`)
  - `rag.search_first.prefetch(question: str) -> PrefetchResult`
  - `rag.search_first.compose_agent_input(question: str, sources: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_search_first.py`:

```python
"""Retrieval runs before the agent (spec D1).

The agent used to decide whether to search. Measured in production: 21 searches
across 25 runs. These tests pin the front-end that removes the decision, and the
two outcomes that must never reach an LLM at all.
"""

import pytest

import rag.search_first as sf
import rag.tools.search_policies as sp


def test_the_question_reaches_search_verbatim(monkeypatch):
    """No rewriting, no keyword extraction — the retrieval stack is tuned for
    natural-language questions, and the old prompt spent tokens saying so."""
    seen = {}
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: seen.setdefault("q", q) or "[Source 1] x")

    sf.prefetch("If it's just for internal tools, can I skip approvals?")

    assert seen["q"] == "If it's just for internal tools, can I skip approvals?"


def test_sources_are_returned_on_the_ok_path(monkeypatch):
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: "=== RETRIEVED POLICY SOURCES ===\n\n[Source 1] AUP")

    result = sf.prefetch("anything")

    assert result.status == "ok"
    assert "[Source 1] AUP" in result.sources


def test_the_bare_no_match_sentinel_is_classified(monkeypatch):
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.NO_MATCH)

    result = sf.prefetch("anything")

    assert result.status == "no_match"
    assert result.sources == ""


def test_the_formatted_no_match_sentinel_is_classified(monkeypatch):
    """format_sources([]) wraps the sentinel in a header, so it arrives in a
    second shape. Both must classify the same way."""
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.format_sources([]))

    assert sf.prefetch("anything").status == "no_match"


def test_the_unavailable_sentinel_is_classified(monkeypatch):
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.UNAVAILABLE)

    assert sf.prefetch("anything").status == "unavailable"


def test_the_unavailable_flag_is_honoured_even_without_the_sentinel(monkeypatch):
    """Belt and braces: the flag is the authoritative signal, the string is a
    convenience. A future return-shape change must not silently downgrade an
    infra outage into a content escalation."""
    def _flagged(q, *a, **k):
        sp._retrieval_unavailable = True
        return "whatever"

    monkeypatch.setattr(sp, "search_policies", _flagged)

    assert sf.prefetch("anything").status == "unavailable"


def test_prefetch_resets_the_unavailable_flag_before_searching(monkeypatch):
    """The flag is a module global read after the call; a stale True from a
    previous request would report a false outage."""
    sp._retrieval_unavailable = True
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: "[Source 1] x")

    assert sf.prefetch("anything").status == "ok"


def test_an_unavailable_prefetch_logs_which_component_failed(monkeypatch, capsys):
    """search_policies itself never logs — it only emits the Phoenix span. This
    print is the only container-log record that retrieval, not the LLM, failed.
    It used to live in _run_rag's post-run check, which Task 5 deletes."""
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.UNAVAILABLE)

    sf.prefetch("anything")

    assert "Unavailable (retrieval)" in capsys.readouterr().out


def test_compose_puts_the_question_first_then_the_sources():
    composed = sf.compose_agent_input("Can I install software?", "=== RETRIEVED POLICY SOURCES ===\n\n[Source 1] AUP")

    assert composed.startswith("Can I install software?")
    assert "[Source 1] AUP" in composed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_search_first.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.search_first'`

- [ ] **Step 3: Extract the sentinel constants**

In `rag/tools/search_policies.py`, add below the existing module globals:

```python
NO_MATCH = "NO_RELEVANT_POLICY_FOUND"
UNAVAILABLE = "POLICY_SEARCH_UNAVAILABLE"
```

Replace every literal occurrence of those two strings in this file with the constants — the five `return "NO_RELEVANT_POLICY_FOUND"` sites, the two `return "POLICY_SEARCH_UNAVAILABLE"` sites, and the one inside `format_sources`. Leave the docstring prose alone.

- [ ] **Step 4: Create the module**

Create `rag/search_first.py`:

```python
"""Retrieval that happens before the agent, not because the agent asked.

The agent used to own `search_policies` as a tool and chose whether to call it:
21 calls across 25 production runs. A compliance answer produced without retrieval
violates the project's first constraint, so the choice is removed — retrieval runs
here, and its formatted sources are handed to the agent as part of the question.

Shared by channels/teams/bot.py and eval/agent_wrapper.py so eval measures what
production does.
"""

from dataclasses import dataclass

import rag.tools.search_policies as sp


@dataclass(frozen=True)
class PrefetchResult:
    """status is one of "ok" | "unavailable" | "no_match". sources is empty unless ok."""

    status: str
    sources: str = ""


def prefetch(question: str) -> PrefetchResult:
    """Run retrieval for `question` and classify the outcome.

    The question is passed verbatim: the retrieval stack is tuned for natural
    language, and rewriting it into keywords measurably hurts.
    """
    sp._retrieval_unavailable = False
    text = sp.search_policies(question)

    if sp._retrieval_unavailable or text == sp.UNAVAILABLE:
        # search_policies emits the Phoenix span but never logs, so this is the
        # only container-log record that retrieval — not the LLM — was the
        # failing component.
        print(
            "[worker] Unavailable (retrieval): search_policies flagged the "
            "backend unavailable (embeddings/qdrant)"
        )
        return PrefetchResult("unavailable")

    if text == sp.NO_MATCH or text.endswith(f"\n\n{sp.NO_MATCH}"):
        return PrefetchResult("no_match")

    return PrefetchResult("ok", text)


def compose_agent_input(question: str, sources: str) -> str:
    """The agent's user message: the question, then the sources.

    `sources` already carries its own "=== RETRIEVED POLICY SOURCES ===" header
    from format_sources, so nothing is added around it.
    """
    return f"{question}\n\n{sources}"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_search_first.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Run the full unit suite**

Run: `PYTHONPATH=. pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add rag/search_first.py rag/tools/search_policies.py tests/unit/test_search_first.py
git commit -m "feat(rag): retrieval runs before the agent, not by its choice

prefetch() runs search_policies and classifies the outcome; the two sentinels
short-circuit so unavailable and no-match never reach an LLM. Sentinel strings
become module constants because prefetch has to match both shapes the no-match
case arrives in (bare, and wrapped by format_sources).

Shared with eval so it measures what production does."
```

---

### Task 3: Rewrite the system prompt

Implements spec **D3**. 1,066 → 540 tokens, measured against a live `qwen3.6` 36B.

**Files:**
- Modify: `rag/agent.py:58-141` (`SYSTEM_PROMPT`)
- Modify: `tests/unit/test_llm_config.py` (add the budget guard)

**Interfaces:**
- Consumes: nothing.
- Produces: `rag.agent.SYSTEM_PROMPT` (2,169 characters), `rag.agent.FIXED_OVERHEAD_TOKENS: int = 540`, `rag.agent.MAX_SOURCE_TOKENS: int = 1300`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_llm_config.py`:

```python
def test_the_request_budget_fits_in_the_context_window():
    """num_ctx is pinned at 4096 by the MoE+CUDA crash matrix, so the budget is
    fixed and has to be checked, not hoped for.

    Before this work the fixed overhead was 1883 tokens (1066 prompt + 817 of tool
    schemas) — 46% of the window — and num_predict 1024 exceeded the 956 tokens
    actually left at the largest observed prompt. The cap was nominal.

    FIXED_OVERHEAD_TOKENS is measured, not derived, so the char assertion below
    pins it to the prompt it was measured against: editing the prompt without
    re-measuring fails here instead of silently rotting the constant.
    """
    from rag.agent import FIXED_OVERHEAD_TOKENS, MAX_SOURCE_TOKENS, SYSTEM_PROMPT

    num_ctx = settings.ollama_num_ctx
    num_predict = 1024

    assert len(SYSTEM_PROMPT) == 2169, (
        "SYSTEM_PROMPT changed; re-measure FIXED_OVERHEAD_TOKENS with "
        "/api/chat num_predict=1 and update both numbers together"
    )
    assert FIXED_OVERHEAD_TOKENS + MAX_SOURCE_TOKENS + num_predict <= num_ctx


def test_the_system_prompt_names_no_tools():
    """The agent is tool-free (spec D2). A prompt that still orders a tool call
    would make the model attempt one that does not exist."""
    from rag.agent import SYSTEM_PROMPT

    for tool in ("search_policies", "get_section", "escalate_to_compliance"):
        assert tool not in SYSTEM_PROMPT
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_llm_config.py -v -k "budget or names_no_tools"`
Expected: FAIL — `ImportError: cannot import name 'FIXED_OVERHEAD_TOKENS'`

- [ ] **Step 3: Replace `SYSTEM_PROMPT`**

In `rag/agent.py`, replace the whole `SYSTEM_PROMPT = """\ ... """` assignment (lines 58-141) with exactly this:

```python
# Measured against a live qwen3.6 36B at num_ctx 4096 via /api/chat with
# num_predict=1, reading prompt_eval_count:
#   this prompt            540 tokens   (the previous one: 1066)
#   tool schemas             0 tokens   (the previous three: 817)
#   fixed overhead         540 tokens   (previously 1883 — 46% of the window)
# Re-measure both constants below if you edit the prompt; the guard in
# tests/unit/test_llm_config.py pins them to its character count.
FIXED_OVERHEAD_TOKENS = 540
# Largest source payload observed in Phoenix (~1257 tokens), rounded up.
MAX_SOURCE_TOKENS = 1300

SYSTEM_PROMPT = """\
You are an internal Compliance Policy Locator. You find the company policy that answers the user's question and show exactly where it is.

You are a POINTER, not an ADVISOR. The policy text IS the answer. Never interpret, explain, summarize, advise, or add reasoning of your own.

== SOURCES ==

The relevant policy sources are given in the user message under [Source N] headers. Read ALL of them before answering.
If none of them answers the question, set escalation.needed = true with a short reason, and leave citations empty. Never guess and never answer from your own knowledge.

== RULES ==

- Quote policy text VERBATIM from every source you cite. Never paraphrase.
- Cite EVERY source that addresses the question, not just the first. Each gets its own entry in citations with its own quote.
- Never cite a source you did not use in the answer.
- Never invent or assume a rule that is not written in the sources.
- Copy source_number, doc_title, section, clause and clause_number exactly as they appear in the source header. Use an empty string when there is no clause.
- If you are uncertain whether a source applies, escalate instead of guessing.

== ANSWER ==

Name the document and location, then quote it:
"According to [Policy Name], Section: [Section Name], Clause [Number] ([Clause Name]): '[verbatim quote].'"

WRONG — advice, and not grounded in a source:
"You should not install software because it could pose a security risk. The IT team needs to approve all installations first."

== OUTPUT ==

Reply with valid JSON only. No text before or after it.

{
  "answer": "[Document A], Section: [X], Clause [N]: '[quote]'. [Document B], Section: [Y], Clause [M]: '[quote]'.",
  "citations": [
    {"source_number": 1, "doc_title": "exact title from the source header", "section": "exact section name", "clause": "exact clause name", "clause_number": "4.7", "quote": "verbatim text from the source"},
    {"source_number": 2, "doc_title": "second document title", "section": "exact section name", "clause": "exact clause name", "clause_number": "8.7", "quote": "verbatim text from the second source"}
  ],
  "escalation": {"needed": false, "reason": ""}
}"""
```

- [ ] **Step 4: Document the response-schema drift**

Directly above `class Citation(BaseModel):` in `rag/agent.py`, add:

```python
# AUTHORITATIVE CONTRACT: the JSON block inside SYSTEM_PROMPT.
# These models are never instantiated and never sent to the LLM — no
# response_format is set (it kills tool-call emission), and rag/response.py parses
# with plain dict.get. They are documentation of the same contract, kept here for
# readers. Change the prompt and these together, or they drift apart silently.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_llm_config.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full unit suite**

Run: `PYTHONPATH=. pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add rag/agent.py tests/unit/test_llm_config.py
git commit -m "perf(agent): cut the system prompt from 1066 to 540 tokens

Every behavioural rule is preserved; only duplication goes. 'Cite all relevant
sources' appeared 3x, the citation-copying rule 2x, pointer-not-advisor 4x. Two
CORRECT examples collapse to one (the JSON block already shows multi-citation)
and two WRONG examples to one. The search-ordering rules become a SOURCES
section now that retrieval runs before the agent.

Measured via /api/chat num_predict=1. The budget guard pins the constants to the
prompt's character count so an edit without a re-measure fails the suite."
```

---

### Task 4: Tool-free agent, delete the dead tools and dead config

Implements spec **D2**, **D5**, and the config cleanup. Depends on Task 3 — the new prompt must already be in place, or the agent would be told to call tools that no longer exist.

**Files:**
- Modify: `rag/agent.py` (imports, `ALL_TOOLS`, `build_agent`)
- Modify: `rag/tools/__init__.py`
- Delete: `rag/tools/get_section.py`, `rag/tools/escalate.py`
- Modify: `config.py`, `.env.example`
- Test: `tests/unit/test_llm_config.py` (extend)

**Interfaces:**
- Consumes: `rag.agent.SYSTEM_PROMPT` from Task 3.
- Produces: `rag.agent.ALL_TOOLS == []`; `build_agent()` unchanged in signature and still returns `AgentWorkflow`; `settings.agent_timeout` now applied.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_llm_config.py`:

```python
def test_the_agent_is_tool_free():
    """Spec D2. Measured: get_section and escalate_to_compliance were never
    invoked in production, and three controlled probes could not force a second
    retrieval step — once search returns usable sources the model stops.
    search_policies goes too because retrieval now runs before the agent.

    A re-added tool costs ~270-410 tokens of schema on EVERY request, so this
    guard is about the budget as much as the design.
    """
    from rag.agent import ALL_TOOLS

    assert ALL_TOOLS == []


def test_the_agent_run_is_bounded_by_agent_timeout():
    """Nothing bounded an agent run before: agent_timeout was declared, documented
    in .env.example, and consumed by nothing, so the only limit was the per-call
    Ollama request_timeout (300s remote) — under a 12s shutdown drain."""
    from rag.agent import build_agent

    wf = build_agent()
    assert wf._timeout == float(settings.agent_timeout)


def test_the_dead_agent_settings_are_gone():
    """agent_max_iterations could never be wired: neither FunctionAgent nor
    ReActAgent has such a field. escalation_ticket_prefix had exactly one
    consumer, the deleted escalate tool."""
    assert not hasattr(settings, "agent_max_iterations")
    assert not hasattr(settings, "escalation_ticket_prefix")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_llm_config.py -v -k "tool_free or bounded or dead_agent"`
Expected: FAIL — `assert [<FunctionTool>, ...] == []`

- [ ] **Step 3: Empty the tool list and wire the timeout**

In `rag/agent.py`, delete these three imports:

```python
from rag.tools.escalate import escalate_to_compliance_tool
from rag.tools.get_section import get_section_tool
from rag.tools.search_policies import search_policies_tool
```

Replace the `ALL_TOOLS` assignment with:

```python
# Empty by design (spec D2), not by oversight.
# Measured: get_section and escalate_to_compliance were never invoked in
# production. escalate lost to the JSON escalation.needed field — the model must
# emit it anyway, so the tool call is a wasted round-trip — and get_section lost
# to satisfaction: once search returns usable sources the model stops, and three
# controlled probes (explicit order, order moved to the prompt tail, gating
# condition stripped from its docstring) all failed to force a second retrieval
# step. search_policies is gone because retrieval now runs in rag/search_first.py
# before the agent is built.
# Re-adding a tool costs ~270-410 tokens of schema on every request and puts
# tool-calling back — one of the five conditions in the MoE+CUDA crash matrix.
ALL_TOOLS = []
```

In `build_agent()`, change the docstring and pass the timeout:

```python
def build_agent() -> AgentWorkflow:
    """Build the tool-free compliance agent.

    NOT a ReAct agent, despite what this docstring said until 2026-09-24:
    AgentWorkflow.from_tools_or_functions picks FunctionAgent when
    llm.metadata.is_function_calling_model is True, and llama-index's Ollama
    reports True. Production has always used native tool calls, never ReAct text.
    """
    llm = get_llm()
    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=ALL_TOOLS,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        timeout=float(settings.agent_timeout),
        # Off: verbose=True prints ~20 [tick]/[run_agent_step] lines per question,
        # which buries the bot's own log — the WARNING lines an operator actually
        # needs (watermark holds, failed acks, force-advances) become unfindable
        # once 30 people are asking. The same detail is in Phoenix, structured and
        # searchable, which is where it belongs. Eval keeps its own verbose switch
        # (eval/agent_wrapper.py build_instrumented_agent) for debugging runs.
        verbose=False,
    )
    return agent
```

- [ ] **Step 4: Delete the tool modules and fix the package exports**

```bash
git rm rag/tools/get_section.py rag/tools/escalate.py
```

Replace the whole contents of `rag/tools/__init__.py` with:

```python
from rag.tools.search_policies import search_policies_tool

__all__ = ["search_policies_tool"]
```

(The `ALL_TOOLS` that lived here was a second, dead copy — `rag/agent.py` always defined its own and never imported this one.)

- [ ] **Step 5: Delete the dead settings**

In `config.py`, delete the `agent_max_iterations` line and the `escalation_ticket_prefix` line. Leave `agent_timeout` (now wired) and leave the `smtp_*` / `compliance_team_email` settings, which belong to unimplemented email escalation.

In `.env.example`, delete the `AGENT_MAX_ITERATIONS=8` and `ESCALATION_TICKET_PREFIX=ESC` lines. Leave `AGENT_TIMEOUT=120`.

Do **not** edit `.env` or `.env.remote-backup`: unknown keys are inert to pydantic-settings, and the backup is an untracked restore artefact.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_llm_config.py -v`
Expected: PASS.

- [ ] **Step 7: Confirm nothing still references the deleted modules**

Run: `grep -rn "get_section\|escalate_to_compliance\|escalation_ticket_prefix\|agent_max_iterations" --include='*.py' .`
Expected: matches only in `eval/` (fixed in Task 7) and `rag/observability.py:9`.

- [ ] **Step 7b: Correct the observability docstring**

`rag/observability.py:9` reads `- Every tool call (search_policies, get_section, clarify, escalate)`. Change it to:

```
- The pre-agent retrieval call (search_policies, via rag/search_first.py)
```

- [ ] **Step 8: Run the full unit suite**

Run: `PYTHONPATH=. pytest tests/unit -q`
Expected: PASS. `tests/unit/test_evaluators.py` does not reference the deleted tools.

- [ ] **Step 9: Commit**

```bash
git add -A rag/ config.py .env.example tests/unit/test_llm_config.py
git commit -m "refactor(agent): tool-free agent; delete the two dead tools and dead config

get_section and escalate_to_compliance were never invoked in production.
escalate lost to the JSON escalation.needed field; get_section lost to
satisfaction — once search returns usable sources the model stops, and three
probes failed to force a second retrieval step. search_policies moves out to
rag/search_first.py.

escalate_to_compliance was worse than unused: forced to fire, it returns a
fabricated ticket id as prose, the parse then fails, and bot.py logs the
escalation as outcome=answered.

Also: rag/tools/__init__.py held a second, dead ALL_TOOLS; agent_max_iterations
could never be wired (no such field on either agent class); agent_timeout was
declared and never applied, leaving an agent run unbounded under a 12s drain."
```

---

### Task 5: Wire `_run_rag` to search-first

**Files:**
- Modify: `channels/teams/bot.py:97-150` (`_run_rag`)
- Test: `tests/unit/test_bot_search_first.py` (create)

**Interfaces:**
- Consumes: `rag.search_first.prefetch`, `rag.search_first.compose_agent_input` (Task 2); `rag.agent.build_agent` (Task 4).
- Produces: `_run_rag` returns, in addition to its existing shapes, `{"answer": "", "citations": [], "escalation": {"needed": True, "reason": ...}, "parse_success": True}` on the no-match path.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_bot_search_first.py`:

```python
"""_run_rag after retrieval moved in front of the agent.

The point of the short-circuits is that they cost no LLM call: an unavailable
backend or a no-match search used to spend a full ~16s agent run to reach the
same conclusion.
"""

import pytest

import channels.teams.bot as bot
import rag.search_first as sf


@pytest.fixture
def no_agent(monkeypatch):
    """Fails loudly if anything builds an agent — that is what we are asserting."""
    def _boom():
        raise AssertionError("_run_rag built an agent on a short-circuit path")

    monkeypatch.setattr("rag.agent.build_agent", _boom)


def test_an_unavailable_search_short_circuits_without_an_llm_call(monkeypatch, no_agent):
    monkeypatch.setattr(sf, "prefetch", lambda q: sf.PrefetchResult("unavailable"))

    assert bot._run_rag("anything") == {"status": "unavailable"}


def test_a_no_match_search_escalates_without_an_llm_call(monkeypatch, no_agent):
    """CLAUDE.md requires this ('If search_policies returns NO_RELEVANT_POLICY_FOUND
    → escalate'). It used to be a prompt instruction the model could ignore; here
    it becomes a guarantee."""
    monkeypatch.setattr(sf, "prefetch", lambda q: sf.PrefetchResult("no_match"))

    result = bot._run_rag("anything")

    assert result["escalation"]["needed"] is True
    assert result["citations"] == []
    assert result["parse_success"] is True


def test_the_sources_reach_the_agent(monkeypatch):
    """The whole point: the agent answers from sources it did not have to ask for."""
    monkeypatch.setattr(
        sf, "prefetch", lambda q: sf.PrefetchResult("ok", "=== RETRIEVED POLICY SOURCES ===\n\n[Source 1] AUP")
    )
    seen = {}

    class _Agent:
        async def run(self, user_msg):
            seen["msg"] = user_msg
            return '{"answer": "a", "citations": [], "escalation": {"needed": false, "reason": ""}}'

    monkeypatch.setattr("rag.agent.build_agent", lambda: _Agent())

    bot._run_rag("Can I install software?")

    assert seen["msg"].startswith("Can I install software?")
    assert "[Source 1] AUP" in seen["msg"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_bot_search_first.py -v`
Expected: FAIL — `_run_rag` still calls `build_agent` on every path.

- [ ] **Step 3: Rewrite `_run_rag`'s retrieval handling**

In `channels/teams/bot.py`, inside `_run_rag`, add to the deferred-import block:

```python
    from rag.search_first import compose_agent_input, prefetch
```

Delete the line `sp._retrieval_unavailable = False` and the `import rag.tools.search_policies as sp` line (nothing else in the function uses `sp` after Step 4).

Immediately after the imports, add:

```python
    pre = prefetch(question)
    if pre.status == "unavailable":
        return {"status": "unavailable"}
    if pre.status == "no_match":
        return {
            "answer": "",
            "citations": [],
            "escalation": {
                "needed": True,
                "reason": "No relevant policy was found for this question.",
            },
            "parse_success": True,
        }
```

Change the inner coroutine to pass the composed message:

```python
    async def _run():
        agent = build_agent()
        return await agent.run(user_msg=compose_agent_input(question, pre.sources))
```

- [ ] **Step 4: Delete the now-unreachable post-run check**

Delete this block (currently `channels/teams/bot.py:143-149`), comment included:

```python
    # Retrieval failed inside the tool (LlamaIndex swallows tool exceptions) →
    # ...
    if sp._retrieval_unavailable:
        print("[worker] Unavailable (retrieval): ...")
        return {"status": "unavailable"}
```

It exists only because `AgentWorkflow` swallows tool exceptions, so a failure inside `search_policies` was visible only via the flag after the run. With retrieval hoisted out and no tools left, the only caller is `prefetch`, which sees the failure directly and now carries the log line (Task 2).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_bot_search_first.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the full unit suite**

Run: `PYTHONPATH=. pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add channels/teams/bot.py tests/unit/test_bot_search_first.py
git commit -m "feat(teams): _run_rag retrieves before building the agent

Unavailable and no-match now return without an LLM call at all — both used to
spend a full agent run to reach the same conclusion. The no-match path makes
CLAUDE.md's 'NO_RELEVANT_POLICY_FOUND -> escalate' a guarantee rather than a
prompt instruction the model could ignore.

The post-run sp._retrieval_unavailable check is deleted, not moved: it existed
because AgentWorkflow swallows tool exceptions, and with no tools left it is
unreachable. prefetch carries its log line."
```

---

### Task 6: The grounding backstop

Implements the spec's escalation layer 3. Today `parse_success` is read only by `eval/` and tests, and `renderer.py:79-80` renders an answer with no citations as bare prose — which violates the project's grounding constraint outright.

**Files:**
- Modify: `channels/teams/bot.py:742-752` (the render branch)
- Modify: `channels/teams/renderer.py:78-80`
- Test: `tests/unit/test_bot_search_first.py` (extend), `tests/unit/test_renderer.py` (extend)

**Interfaces:**
- Consumes: `_run_rag`'s result dict (Task 5).
- Produces: two new `compliance_request.outcome` values — `"escalated_parse_failure"` and `"escalated_ungrounded"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_bot_search_first.py`:

```python
# --- the grounding backstop -------------------------------------------------
#
# CLAUDE.md: "Agent must never answer without citing a retrieved chunk."
# Two ways that was violated in production, both demonstrated:
#   - a failed parse falls back to escalation.needed=False with the raw model
#     text as `answer`, so bot.py renders it and logs outcome="answered"
#   - an answer with zero citations renders as bare prose

def _sent(monkeypatch, tmp_path, result):
    """Run _answer with _run_rag stubbed; return (html, outcome)."""
    import contextlib

    import rag.observability as obs

    # Hermetic: never read or write the developer's real bot_state.json / bot.pid.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    monkeypatch.setattr(bot, "PID_FILE", tmp_path / "bot.pid")

    b = bot.TeamsBot(token_refresher=object())
    captured = {}
    monkeypatch.setattr(bot, "_run_rag", lambda q: result)
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(
        b, "_send_message",
        lambda chat_id, text, content_type="html", retry=False: captured.setdefault("html", text) or True,
    )

    class _Span:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

    span = _Span()

    class _Tracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name, **kwargs):
            yield span

    # _answer imports get_tracer INSIDE the function (the observability-first rule),
    # so the module attribute is what the call resolves — patch it, not bot's.
    monkeypatch.setattr(obs, "get_tracer", lambda: _Tracer())

    b._answer("chat1", "Can I install software?", "Ann")
    return captured.get("html", ""), span.attrs.get("compliance_request.outcome")


def test_a_failed_parse_is_escalated_not_answered(monkeypatch, tmp_path):
    html, outcome = _sent(monkeypatch, tmp_path, {
        "answer": "ESCALATED: Ticket #ESC-2026-0001. They will respond within 2 business days.",
        "citations": [],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": False,
    })

    assert outcome == "escalated_parse_failure"
    assert "ESC-2026-0001" not in html


def test_the_parse_failure_reason_never_carries_model_text(monkeypatch, tmp_path):
    """The renderer does not HTML-escape. Putting the raw response into `reason`
    would interpolate arbitrary model output straight into a Teams message."""
    html, _ = _sent(monkeypatch, tmp_path, {
        "answer": "<script>alert(1)</script> and <b>markup</b>",
        "citations": [],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": False,
    })

    assert "<script>" not in html


def test_an_answer_without_citations_is_escalated(monkeypatch, tmp_path):
    html, outcome = _sent(monkeypatch, tmp_path, {
        "answer": "You should ask IT before installing anything.",
        "citations": [],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": True,
    })

    assert outcome == "escalated_ungrounded"
    assert "You should ask IT" not in html


def test_a_cited_answer_is_still_answered(monkeypatch, tmp_path):
    _, outcome = _sent(monkeypatch, tmp_path, {
        "answer": "According to the AUP ...",
        "citations": [{"doc_title": "AUP", "section": "Use", "clause": "", "clause_number": "4.7", "quote": "q"}],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": True,
    })

    assert outcome == "answered"
```

Append to `tests/unit/test_renderer.py`:

```python
def test_render_answer_refuses_an_uncited_answer():
    """The old fallback returned the bare prose, violating CLAUDE.md's grounding
    constraint. Nothing should reach here now — the bot escalates first — so this
    pins the second line of defence."""
    from channels.teams.renderer import render_answer

    with pytest.raises(ValueError):
        render_answer({"answer": "some ungrounded prose", "citations": []})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_bot_search_first.py tests/unit/test_renderer.py -v -k "parse or citation or uncited or cited"`
Expected: FAIL — outcome is `"answered"`, and `render_answer` returns prose instead of raising.

- [ ] **Step 3: Add the backstop**

In `channels/teams/bot.py`, replace the render branch (currently lines 742-752) with:

```python
            # Grounding backstop. CLAUDE.md: "Agent must never answer without
            # citing a retrieved chunk." Two ways that was violated before, both
            # reproduced: a failed parse falls back to escalation.needed=False
            # with the raw model text as `answer`, and an answer with no
            # citations rendered as bare prose. Both are escalations, and both
            # get their own outcome so they stay countable in Phoenix instead of
            # hiding inside "answered".
            escalation = result.get("escalation", {})
            if not result.get("parse_success", True):
                span.set_attribute("compliance_request.outcome", "escalated_parse_failure")
                # A FIXED reason, never result["answer"] or raw_response: the
                # renderer does not HTML-escape, so model output must not be
                # interpolated into a message.
                html = render_escalation(
                    text,
                    {"escalation": {"reason": "The answer could not be read in the expected format."}},
                )
            elif escalation.get("needed"):
                span.set_attribute("compliance_request.outcome", "escalated")
                html = render_escalation(text, result)
            elif result.get("answer") and result.get("citations"):
                span.set_attribute("compliance_request.outcome", "answered")
                html = render_answer(result)
            elif result.get("answer"):
                span.set_attribute("compliance_request.outcome", "escalated_ungrounded")
                html = render_escalation(
                    text,
                    {"escalation": {"reason": "No policy source could be cited for this question."}},
                )
            else:
                span.set_attribute("compliance_request.outcome", "error")
                html = render_error(text, "No answer returned from the pipeline.")
```

- [ ] **Step 4: Remove the renderer's uncited fallback**

In `channels/teams/renderer.py`, replace:

```python
    if not citations:
        # Fallback to prose answer when there are no structured citations
        return f"<p>{answer}</p>"
```

with:

```python
    if not citations:
        # Second line of defence. The bot escalates an uncited answer before it
        # gets here (grounding backstop in bot.py), so reaching this point means
        # that guard was bypassed. Returning the prose would violate CLAUDE.md's
        # "never answer without citing a retrieved chunk".
        raise ValueError("render_answer called with no citations")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_bot_search_first.py tests/unit/test_renderer.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full unit suite**

Run: `PYTHONPATH=. pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add channels/teams/bot.py channels/teams/renderer.py tests/unit/
git commit -m "fix(teams): refuse to render an answer with no citation or a failed parse

parse_success was read only by eval and tests; bot.py ignored it, so an
unparseable response became an answer with the raw model text as its body. And
renderer.py returned bare prose whenever citations were empty — a direct
violation of 'never answer without citing a retrieved chunk'.

Both now escalate, each with its own Phoenix outcome so they stay countable
instead of hiding inside 'answered'. The parse-failure reason is a fixed string:
the renderer does not HTML-escape, so model output must never be interpolated."
```

---

### Task 7: Make eval measure what production does

`eval/agent_wrapper.py` imports production's `SYSTEM_PROMPT` but builds its own three-tool agent. After Tasks 3-5 it would run a prompt that says "sources are provided" against an agent that receives none.

**Files:**
- Modify: `eval/agent_wrapper.py`
- Modify: `eval/evaluators.py:363-375` and `:424-427`
- Modify: `eval/run_experiment.py:146-165`
- Test: `tests/unit/test_evaluators.py` (extend)

**Interfaces:**
- Consumes: `rag.search_first.prefetch`, `compose_agent_input` (Task 2); `rag.agent.ALL_TOOLS` (Task 4).
- Produces: `agent_metadata` keys `search_queries`, `num_searches`, `escalated`, `escalation_reason`. The keys `num_section_fetches` and `section_fetches` are **removed**.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_evaluators.py`:

```python
def test_the_agent_evaluators_no_longer_reference_deleted_tools():
    """agent_used_get_section scored 0.5/'skipped' on every run ever — the tool
    was never called. It is deleted along with the tool."""
    import eval.evaluators as ev

    assert not hasattr(ev, "agent_used_get_section")
    assert ev.AGENT_EVALUATORS == [ev.agent_search_count]


def test_search_count_still_scores_a_single_prefetch():
    """Retrieval now happens once, in code, before the agent. num_searches must
    keep counting it or every case would read as 'never searched — hallucinated'."""
    from eval.evaluators import agent_search_count

    result = agent_search_count({"agent_metadata": {"num_searches": 1}}, {})
    assert result["score"] == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_evaluators.py -v -k "deleted_tools or single_prefetch"`
Expected: FAIL — `agent_used_get_section` still exists.

- [ ] **Step 3: Rewrite the eval agent wrapper**

Replace the contents of `eval/agent_wrapper.py` with:

```python
"""Instrumented agent for evaluation.

Mirrors production exactly: retrieval runs in rag/search_first.py before a
tool-free agent. The only difference is the tool-call log, which eval needs for
its agent_metadata. Do not give this agent tools — a divergence here means eval
stops measuring the thing that ships.
"""

from llama_index.core.agent.workflow import AgentWorkflow

from config import settings
from rag.agent import ALL_TOOLS, SYSTEM_PROMPT, get_llm
from rag.response import parse_agent_response  # re-exported for backwards compat
from rag.search_first import compose_agent_input, prefetch
import rag.tools.search_policies as sp

__all__ = [
    "build_instrumented_agent",
    "compose_agent_input",
    "get_log",
    "clear_log",
    "parse_agent_response",
    "prefetch_logged",
]

_tool_call_log: list[dict] = []


def get_log() -> list[dict]:
    return _tool_call_log


def clear_log() -> None:
    _tool_call_log.clear()


def prefetch_logged(question: str):
    """prefetch(), plus the log entry eval's agent_metadata is built from."""
    result = prefetch(question)
    _tool_call_log.append(
        {
            "tool": "search_policies",
            "query": question,
            "status": result.status,
            "results": list(sp._last_search_results),
        }
    )
    return result


def build_instrumented_agent(verbose: bool = False) -> AgentWorkflow:
    """Tool-free, like production. Fresh agent per call — no state leakage."""
    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=ALL_TOOLS,
        llm=get_llm(),
        system_prompt=SYSTEM_PROMPT,
        timeout=float(settings.agent_timeout),
        verbose=verbose,
    )
```

- [ ] **Step 4: Update the experiment task**

In `eval/run_experiment.py`, the task function must call `prefetch_logged(question)` and pass `compose_agent_input(question, result.sources)` to the agent, exactly as `_run_rag` does. Then replace lines 146-165's metadata construction with:

```python
        escalated = bool(parsed["escalation"].get("needed"))

        return {
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "escalation": parsed["escalation"],
            "parse_success": parsed["parse_success"],
            "raw_response": parsed["raw_response"],
            "search_results": unique_results,
            "agent_metadata": {
                "search_queries": search_queries,
                "num_searches": len(search_queries),
                # Escalation is read from the JSON field, which is the only
                # escalation path there has ever been in production — the tool
                # that used to be counted here was never called.
                "escalated": escalated,
                "escalation_reason": parsed["escalation"].get("reason") or None,
```

Delete the `section_calls` and `escalation_calls` list comprehensions above it.

- [ ] **Step 5: Delete the dead evaluator**

In `eval/evaluators.py`, delete the whole `agent_used_get_section` function (lines 363-375) and change:

```python
AGENT_EVALUATORS = [
    agent_search_count,
]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_evaluators.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full unit suite**

Run: `PYTHONPATH=. pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add eval/ tests/unit/test_evaluators.py
git commit -m "test(eval): eval retrieves like production and drops the dead evaluator

agent_wrapper imported production's SYSTEM_PROMPT but built its own three-tool
agent; after search-first that would run a prompt saying 'sources are provided'
against an agent that receives none.

agent_used_get_section scored 0.5/'skipped' on every run ever. Escalation is now
counted from the JSON field, which is the only escalation path production has
ever had."
```

---

### Task 8: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the architecture block**

In `CLAUDE.md`, under `rag/`, change the `tools/` line to:

```
  tools/             # search_policies (run BEFORE the agent, by rag/search_first.py)
                     #   (clarify.py exists but is NOT imported/used)
  search_first.py    # prefetch() + compose_agent_input() — retrieval front-end
```

- [ ] **Step 2: Rewrite the Search flow paragraph**

Replace the existing **Search flow** paragraph with:

```
**Search flow:** `prefetch()` (rag/search_first.py) runs `search_policies` BEFORE the agent —
`embed_query → vector_search (RERANKER_CANDIDATES) → [BM25 RRF] → [rerank → top RERANKER_TOP_N]
→ relevance floor (RERANKER_MIN_SCORE, 0.0 = off) → format_sources()` with `[Source N]` headers —
and its sources are appended to the agent's user message. The agent is **tool-free**: retrieval
is no longer something it can skip. `POLICY_SEARCH_UNAVAILABLE` and `NO_RELEVANT_POLICY_FOUND`
short-circuit before any LLM call.
```

- [ ] **Step 3: Add the escalation and backstop note to the Message flow paragraph**

Append to the **Message flow (Teams)** paragraph:

```
Escalation is decided in three layers: retrieval returning no match escalates in code with no
LLM call; the model sets `escalation.needed` when sources do not answer the question; and a
grounding backstop refuses to render an answer whose parse failed or whose citations are empty
(outcomes `escalated_parse_failure` / `escalated_ungrounded`).
```

- [ ] **Step 4: Add two gotchas rows**

```
| Wondering why the agent "ignores" a tool | It is a **FunctionAgent**, not a ReActAgent — `from_tools_or_functions` picks on `is_function_calling_model`, which Ollama reports True. Native tool calls, no ReAct text (that gotcha is `OpenAILike`-only). The model is capable of chaining (measured), but **stops once retrieval returns usable sources**: three probes (explicit order, order at the prompt tail, gating condition stripped) all failed to force a second tool call. Do not add a tool expecting the model to reach for it. |
| A relevance setting that never fires | `MIN_CONFIDENCE_SCORE` applies **only when the reranker is OFF**, and production runs it ON — so it never ran. `RERANKER_MIN_SCORE` is the live floor, on the reranker's 0.0-1.0 score, and it is guarded on that key being *present*: the reranker's fallback path carries no score, and reading a missing one as 0.0 would turn a reranker outage into "no policy exists" for every question. |
```

- [ ] **Step 5: Update the config key list**

Add to the `.env` key block: `RERANKER_MIN_SCORE (0.0 = off; measure from Phoenix reranker.top_score before setting)`. Remove `AGENT_MAX_ITERATIONS` / `ESCALATION_TICKET_PREFIX` if listed.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: search-first flow, three-layer escalation, two new gotchas"
```

---

### Task 9: VM — set the threshold and run the eval gate

**This task runs on the VM, by the operator. It cannot be done by a subagent and it cannot be done locally:** every probe behind this design ran with the reranker down, so no local `rerank_score` exists. Stop here and hand back.

- [ ] **Step 1: Deploy the branch to the VM**

```bash
docker compose -f docker-compose-remote.yml up -d --build
```

- [ ] **Step 2: Read the score distribution from the VM's Phoenix**

Query the `compliance-bot` project for `reranker.top_score` across recent traces, split by the request's `compliance_request.outcome` (`answered` vs `escalated`). The threshold goes below the lowest score that produced a good answer, and above the bulk of the escalated ones. If the two distributions overlap heavily, the reranker's discrimination is collapsed (CLAUDE.md's "reranker scores compressed" row) — leave `RERANKER_MIN_SCORE=0.0` and say so, rather than picking a number that separates nothing.

- [ ] **Step 3: Set the value**

Set `RERANKER_MIN_SCORE` in the VM `.env` to the chosen value, and update the default in `config.py` to match, with a one-line comment recording the date and the sample size it came from.

- [ ] **Step 4: Upload the datasets and run the eval gate**

```bash
python scripts/make_dataset.py eval/datasets/<file>.json
python eval/run_experiment.py --tier tier2   --name search-first-v1
python eval/run_experiment.py --tier chatbot --name search-first-v1
```

- [ ] **Step 5: Compare against the pre-change baseline**

Pass/fail on **citation accuracy** — `citation_doc_accuracy`, `citation_section_accuracy`, `citation_clause_accuracy`. Token savings alone do not justify a merge.

Known signal to resolve here: on one covered question the new prompt produced **2 citations where the old prompt produced 3**, reproduced across three variations (tool-free, with tools, and with one cut repetition restored), so it is not noise and not caused by the trimming. Whether 2 or 3 is *correct* is what the labelled dataset answers. If citation accuracy regresses, the cut to restore first is the multi-source emphasis in `== RULES ==`.

- [ ] **Step 6: Report before merging**

Report the citation-accuracy deltas, the chosen threshold with its sample size, and the floor's rejection rate. Merging is a separate decision.

---

## Out of scope

Recorded in the spec, deliberately not built here:

- **Rephrase-and-retry on a rejected search** — pending two measurements (does rephrasing recover misses on `retrieval-test-v1`; what share of production questions the floor rejects). It is the dangerous direction: a second search that surfaces something tangential turns a correct escalation into a weakly-grounded answer.
- **Durable escalation records** — the hook point is `bot.py`'s escalation branch, not a tool.
- **Raising `num_ctx`** — tool-free removes one of the five crash conditions, but that hypothesis goes through `2026-09-17-ollama-moe-cuda-crash.md`'s matrix, not this plan.
- **Replacing `AgentWorkflow` with a direct `llm.achat`** — revisit once eval confirms tool-free.
- **`render_escalation` HTML escaping** — pre-existing, tracked separately.
