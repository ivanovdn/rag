# Query Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `QUERY_REWRITE=off|multi|multi_titles|hyde` step that searches the user's question plus LLM-written variants in one fused Qdrant query, and the tooling to measure it per question against `baseline-identity-v1`.

**Architecture:** A new `rag/query_rewrite.py` makes one temperature-0, plain-text LLM call and returns extra queries (never the original). `search_policies` embeds original + extras in one `/api/embed` call and passes the extras to `search_chunks`, which adds a dense (+ BM25) prefetch per query to the same RRF request. The reranker and the agent keep seeing only the original. With `off` (the default), the code path is the one that runs today, call for call.

**Tech Stack:** Python 3.12, llama-index `Ollama`/`OpenAILike` via `rag.agent.get_llm`, qdrant-client 1.19.1 Query API, OpenTelemetry spans, pytest, Phoenix REST (`/v1/experiments/<id>/json`).

**Spec:** `docs/superpowers/specs/2026-10-09-query-rephrasing-design.md`

## Global Constraints

- `QUERY_REWRITE` default is `off`; with `off`, `search_policies` must call `embed_query(query)` and `search_chunks(query, vector, top_k=...)` exactly as before (existing tests mock both with 1- and 3-positional-arg lambdas — they must keep passing unmodified).
- The original question is always searched; rephrasings only add prefetches.
- The reranker and the agent receive the ORIGINAL question only.
- A rewrite failure of any kind (error, timeout, empty/unusable output, titles lookup failure) never blocks and is never `infra_unavailable`: search runs with the original alone, recorded as `fallback`.
- Rewrite LLM call: temperature `settings.llm_temperature` (0.0), plain text, no tools, no `format`/structured output, fresh client per call (`get_llm(...)`, never cached), timeout `settings.query_rewrite_timeout` (20s), no `retry_transient`.
- `RERANKER_CANDIDATES` sizes every prefetch and the fused limit; never add a per-query count.
- Exactly one worker; module globals in `search_policies` follow the existing reset-then-read pattern.
- Imports at module top (exceptions already documented: `eval/run_experiment.py` imports `rag.*` inside functions; `channels/teams/bot.py` defers `rag.*` imports past `init_observability`).
- Tests never touch the network or `172.20.0.22`.
- No new dependencies.

## Review Focus

1. **Model echoes the original or numbers its lines** ("1. How …", "- …", quoted) → must be stripped/deduplicated, not searched twice or with list markers. Pinned in Task 3 (`test_parse_strips_markers_quotes_and_the_original`).
2. **Model returns more than 2 lines, or a preamble line ("Here are two versions:")** → keep at most 2, drop lines ending in `:`. Pinned in Task 3 (`test_parse_drops_preamble_and_caps_at_two`).
3. **`QUERY_REWRITE` on with BM25 off and reranker off** → result is RRF-fused, so the cosine floor must not judge it, and `score_type` must say `rrf`. Pinned in Task 1 (`cosine_floor_applies`) and Task 4 (`test_rewritten_search_reports_rrf_scores_even_with_bm25_off`).
4. **Embedding backend down while rewrite is on** → still `POLICY_SEARCH_UNAVAILABLE` with `failed_component="embeddings"`, exactly like today. Pinned in Task 4 (`test_embedding_outage_with_rewrite_on_is_still_unavailable`).
5. **Rewrite returns but the LLM is slow/hangs** → bounded by `query_rewrite_timeout`, then fallback. Pinned in Task 1 (`get_llm` honours `timeout`) and Task 3 (`test_any_llm_exception_falls_back_to_the_original`).

## File Structure

| File | Responsibility |
|---|---|
| `config.py` (modify) | `query_rewrite`, `query_rewrite_timeout`, `cosine_floor_applies` third term, `_INERT_WHEN` row |
| `rag/agent.py` (modify) | `get_llm(model, timeout)` override |
| `rag/embeddings.py` (modify) | `embed_queries(list)` — one backend call; `embed_query` delegates |
| `rag/vector_store.py` (modify) | `search_chunks(..., extra_queries=())`; `policy_titles()` |
| `rag/query_rewrite.py` (create) | prompts, `RewriteResult`, `parse_rephrasings`, `rewrite_query` + its span |
| `rag/tools/search_policies.py` (modify) | call rewrite, batch-embed, pass extras, `_last_rewrite`, `score_type` |
| `rag/run_identity.py` (modify) | `rewrite_identity()` |
| `channels/teams/bot.py` (modify) | rewrite identity on `compliance_request` |
| `eval/run_experiment.py` (modify) | tier1 through `search_policies`; rewrite in metadata and outputs |
| `eval/agent_wrapper.py` (modify) | log entry carries the rewrite |
| `eval/compare_runs.py` (create) | per-question rank comparison of experiments |
| `.env.example`, `CLAUDE.md`, `SETUP.md`, the spec (modify) | docs |

---

### Task 1: Settings, `get_llm` timeout, batch query embedding

**Already applied, uncommitted, on `feat/query-rewrite`** (`config.py`, `rag/agent.py`, `rag/embeddings.py`; suite green at 472). This task adds the tests, the `_INERT_WHEN` row and `.env.example`, then commits.

**Files:**
- Modify: `config.py` (field block above `# Reranker`; `cosine_floor_applies`; `_INERT_WHEN`)
- Modify: `rag/agent.py:140-180`
- Modify: `rag/embeddings.py` (end of file)
- Modify: `.env.example` (after the `ROUTER_*` block, line ~72)
- Test: `tests/unit/test_query_rewrite_config.py` (create)

**Interfaces:**
- Produces: `settings.query_rewrite: Literal["off","multi","multi_titles","hyde"]`, `settings.query_rewrite_timeout: int`, `get_llm(model: str | None = None, timeout: float | None = None)`, `embed_queries(queries: list[str]) -> list[list[float]]` (span name `embed_query`, `embedding.text_count = len(queries)`).

- [ ] **Step 1: Review the applied diff**

Run: `git diff config.py rag/agent.py rag/embeddings.py`
Expected: `query_rewrite` + `query_rewrite_timeout` fields; `cosine_floor_applies` returns `not self.reranker_enabled and not self.bm25_enabled and self.query_rewrite == "off"`; `get_llm` resolves `timeout = float(timeout if timeout is not None else settings.active_request_timeout)` and passes it as `timeout=` (OpenAILike) / `request_timeout=` (Ollama); `embed_query` returns `embed_queries([query])[0]`.

- [ ] **Step 2: Add the inert row** — in `config.py`, append to the `_INERT_WHEN` tuple:

```python
    (
        "QUERY_REWRITE_TIMEOUT",
        lambda s: s.query_rewrite == "off",
        "it bounds the query-rewrite LLM call, and QUERY_REWRITE is off.",
    ),
```

- [ ] **Step 3: Write the tests** — `tests/unit/test_query_rewrite_config.py`:

```python
"""Settings and plumbing the query rewrite relies on (design 2026-10-09)."""

import pytest
from pydantic import ValidationError

import rag.embeddings as embeddings_mod
from config import Settings, inert_env_keys
from rag.agent import get_llm


def test_rewrite_is_off_unless_configured():
    assert Settings(_env_file=None).query_rewrite == "off"


def test_an_unknown_rewrite_mode_refuses_to_start():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, query_rewrite="mutli")


@pytest.mark.parametrize("mode", ["multi", "multi_titles", "hyde"])
def test_a_rewritten_search_is_never_judged_by_the_cosine_floor(mode):
    """Several queries are RRF-fused even with BM25 off: 0.016 against 0.45
    would turn every question into "no policy exists"."""
    s = Settings(_env_file=None, reranker_enabled=False, bm25_enabled=False, query_rewrite=mode)
    assert s.cosine_floor_applies is False


def test_the_cosine_floor_still_applies_to_a_plain_dense_search():
    s = Settings(_env_file=None, reranker_enabled=False, bm25_enabled=False, query_rewrite="off")
    assert s.cosine_floor_applies is True


def test_the_rewrite_timeout_is_inert_while_rewrite_is_off():
    s = Settings(_env_file=None, query_rewrite="off")
    reported = inert_env_keys(s, env_text="QUERY_REWRITE_TIMEOUT=5\n", environ={})
    assert len(reported) == 1
    assert reported[0].startswith("QUERY_REWRITE_TIMEOUT is set but does nothing here")


def test_the_rewrite_timeout_is_live_when_rewrite_is_on():
    s = Settings(_env_file=None, query_rewrite="multi")
    assert inert_env_keys(s, env_text="QUERY_REWRITE_TIMEOUT=5\n", environ={}) == []


def test_get_llm_honours_a_timeout_override(monkeypatch):
    monkeypatch.setattr("rag.agent.settings.llm_backend", "ollama")
    assert get_llm(timeout=20).request_timeout == 20.0


def test_get_llm_defaults_to_the_answer_timeout(monkeypatch):
    monkeypatch.setattr("rag.agent.settings.llm_backend", "ollama")
    from rag.agent import settings

    assert get_llm().request_timeout == float(settings.active_request_timeout)


def test_several_queries_cost_one_embedding_call(monkeypatch):
    calls = []

    def _fake(texts, prefix=""):
        calls.append((list(texts), prefix))
        return [[float(i)] for i, _ in enumerate(texts)]

    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(embeddings_mod.settings, "embedding_query_prefix", "Q: ")
    monkeypatch.setattr(embeddings_mod, "_ollama_embed", _fake)

    assert embeddings_mod.embed_queries(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]
    assert calls == [(["a", "b", "c"], "Q: ")]
```

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/unit/test_query_rewrite_config.py tests/unit/test_retrieval_spans.py tests/unit/test_llm_config.py tests/unit/test_dead_config.py -q -p no:warnings`
Expected: all PASS.

- [ ] **Step 5: `.env.example`** — after the `ROUTER_CONFIDENCE_FLOOR=0.6` line add:

```bash

# Query rewrite before retrieval (docs/superpowers/specs/2026-10-09-query-rephrasing-design.md).
# off | multi | multi_titles | hyde. off until a per-question measurement says otherwise.
QUERY_REWRITE=off
# Seconds; bounds the rewrite LLM call. On timeout the original question is searched alone.
QUERY_REWRITE_TIMEOUT=20
```

Run: `.venv/bin/python -m pytest tests/unit -q -p no:warnings -k "env_example or dead_config"` → PASS.

- [ ] **Step 6: Commit**

```bash
git add config.py rag/agent.py rag/embeddings.py .env.example tests/unit/test_query_rewrite_config.py
git commit -m "feat(config): QUERY_REWRITE setting, rewrite-call timeout, batch query embedding"
```

---

### Task 2: `search_chunks` extra queries, and `policy_titles()`

**Files:**
- Modify: `rag/vector_store.py` (`search_chunks`, imports; new `policy_titles` after it)
- Test: `tests/unit/test_multi_query_retrieval.py` (create)

**Interfaces:**
- Produces: `search_chunks(query_text: str, query_vector: list[float], top_k: int, extra_queries: Sequence[tuple[str, list[float]]] = ()) -> list[ScoredPoint]`; span attribute `qdrant.query_count`; `policy_titles() -> list[str]` (sorted, distinct `doc_title`s of `settings.qdrant_collection`, cached per collection name, raises on Qdrant error).

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_multi_query_retrieval.py`:

```python
"""Rewritten queries ride in the SAME Qdrant request as the original.

Asserts on the request Qdrant receives; nothing reaches the network.
"""

import pytest

import rag.vector_store as vs


class _Resp:
    points = []


class _FakeClient:
    def __init__(self, scroll_pages=()):
        self.calls, self.scrolls = [], []
        self._pages = list(scroll_pages)

    def query_points(self, **kw):
        self.calls.append(kw)
        return _Resp()

    def scroll(self, **kw):
        self.scrolls.append(kw)
        return self._pages.pop(0)


@pytest.fixture
def client(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    return fake


def test_each_extra_query_adds_a_dense_and_a_sparse_prefetch(monkeypatch, client):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)

    vs.search_chunks("orig", [1.0], top_k=25, extra_queries=[("alt one", [2.0]), ("alt two", [3.0])])

    call = client.calls[0]
    pf = call["prefetch"]
    assert len(pf) == 6
    assert [p.query for p in pf[0::2]] == [[1.0], [2.0], [3.0]]
    assert [p.query.text for p in pf[1::2]] == ["orig", "alt one", "alt two"]
    assert {p.limit for p in pf} == {25} and call["limit"] == 25  # one knob sizes all
    assert call["query"].rrf.k == 60


def test_extra_queries_are_fused_even_with_bm25_off(monkeypatch, client):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)

    vs.search_chunks("orig", [1.0], top_k=6, extra_queries=[("alt", [2.0])])

    call = client.calls[0]
    assert [p.query for p in call["prefetch"]] == [[1.0], [2.0]]
    assert all(p.using is None for p in call["prefetch"])
    assert call["query"].rrf.k == 60


def test_without_extras_the_request_is_unchanged(monkeypatch, client):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)

    vs.search_chunks("orig", [1.0], top_k=6)

    call = client.calls[0]
    assert call["query"] == [1.0] and "prefetch" not in call


def test_policy_titles_are_distinct_sorted_and_read_once(monkeypatch):
    class _P:
        def __init__(self, t):
            self.payload = {"doc_title": t}

    fake = _FakeClient(scroll_pages=[([_P("B"), _P("A")], "next"), ([_P("A")], None)])
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(vs.settings, "qdrant_collection", "titles_test_collection")
    vs._policy_titles.cache_clear()

    assert vs.policy_titles() == ["A", "B"]
    assert vs.policy_titles() == ["A", "B"]
    assert len(fake.scrolls) == 2  # two pages, once — the second call is cached
    assert fake.scrolls[0]["with_payload"] == ["doc_title"]
    assert fake.scrolls[0]["with_vectors"] is False
```

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/unit/test_multi_query_retrieval.py -q -p no:warnings` → FAIL (`unexpected keyword argument 'extra_queries'`, no `policy_titles`).

- [ ] **Step 3: Implement `search_chunks`.** Add `from collections.abc import Sequence` and `from functools import lru_cache` to the imports. Change the signature to:

```python
def search_chunks(
    query_text: str,
    query_vector: list[float],
    top_k: int,
    extra_queries: Sequence[tuple[str, list[float]]] = (),
) -> list:
```

Append to the docstring:

```
    `extra_queries` are (text, vector) pairs from query rewriting, searched
    beside the original in the SAME request: each adds its own dense prefetch
    (and sparse, when BM25 is on), all sized `top_k`, all fused by RRF. More
    queries widen the union; the fused limit -- and so the reranker's workload --
    does not change. With extras the result is fused even when BM25 is off, so
    its score is RRF, not cosine (see Settings.cosine_floor_applies).
```

Add `"qdrant.query_count": 1 + len(extra_queries),` to the span's initial attributes. Replace `if settings.bm25_enabled:` with `if settings.bm25_enabled or extra_queries:`, and replace the literal two-element `prefetch=[...]` with a list built before the call:

```python
            prefetch = []
            for text, vector in [(query_text, query_vector), *extra_queries]:
                prefetch.append(Prefetch(query=vector, limit=limit))
                if settings.bm25_enabled:
                    prefetch.append(
                        Prefetch(
                            query=bm25_document(text),
                            using=SPARSE_VECTOR_NAME,
                            limit=limit,
                        )
                    )
            response = client.query_points(
                collection_name=settings.qdrant_collection,
                prefetch=prefetch,
                query=RrfQuery(rrf=Rrf(k=RRF_K)),
                limit=limit,
                with_payload=True,
            )
```

Leave the `qdrant.fusion` / `qdrant.rrf_k` / `qdrant.prefetch_limit` attributes inside the branch (they are true for both fused cases). Leave `qdrant.bm25_avg_len` there too (it describes the sparse prefetch, harmless when bm25 is off and extras exist — but guard it: `if settings.bm25_enabled: span.set_attribute("qdrant.bm25_avg_len", ...)`).

- [ ] **Step 4: Implement `policy_titles`** after `search_chunks`:

```python
def policy_titles() -> list[str]:
    """Distinct document titles in the active collection, sorted.

    The `multi_titles` rewrite prompt hands these to the LLM as the corpus's
    vocabulary. Cached per collection for the process: titles change only on
    ingest, and a restart follows any ingest that matters. Raises on any Qdrant
    error -- the caller treats that as a rewrite fallback, never a failed search.
    """
    return list(_policy_titles(settings.qdrant_collection))


@lru_cache(maxsize=4)
def _policy_titles(collection: str) -> tuple[str, ...]:
    client = get_qdrant_client()
    titles, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=["doc_title"],
            with_vectors=False,
        )
        titles.update(p.payload["doc_title"] for p in points if p.payload.get("doc_title"))
        if offset is None:
            return tuple(sorted(titles))
```

- [ ] **Step 5: Run** — `.venv/bin/python -m pytest tests/unit/test_multi_query_retrieval.py tests/unit/test_fused_retrieval.py tests/unit/test_retrieval_spans.py -q -p no:warnings` → all PASS.

- [ ] **Step 6: Commit**

```bash
git add rag/vector_store.py tests/unit/test_multi_query_retrieval.py
git commit -m "feat(retrieval): fuse rewritten queries into the same Qdrant request"
```

---

### Task 3: `rag/query_rewrite.py`

**Files:**
- Create: `rag/query_rewrite.py`
- Test: `tests/unit/test_query_rewrite.py`

**Interfaces:**
- Consumes: `get_llm(model=None, timeout=...)` (Task 1), `policy_titles()` (Task 2), `rag.observability.get_tracer`.
- Produces:
  - `REWRITE_PROMPTS: dict[str, str]` keyed `multi`, `multi_titles`, `hyde` (the `multi_titles` value contains the literal `{titles}` placeholder).
  - `@dataclass(frozen=True) RewriteResult(mode: str, queries: tuple[str, ...] = (), fallback: bool = False, error: str = "", latency_ms: int = 0)` with `.as_dict() -> dict`.
  - `parse_rephrasings(raw: str, original: str) -> tuple[str, ...]` (≤2 lines).
  - `parse_passage(raw: str) -> tuple[str, ...]` (0 or 1 passage).
  - `rewrite_query(question: str) -> RewriteResult` — never raises; `mode == "off"` returns `RewriteResult("off")` with no LLM call and no span.

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_query_rewrite.py`:

```python
"""The rewrite step: parse defensively, never block, never touch the network here."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rag.query_rewrite as qr

Q = "Can I disable the 5-minute auto-lock on my computer?"


# --- parsing --------------------------------------------------------------------


def test_parse_strips_markers_quotes_and_the_original():
    raw = f'1. "{Q}"\n2) Are Team Members permitted to disable automatic screen lock?\n- may a user override the inactivity lockout setting'
    assert qr.parse_rephrasings(raw, Q) == (
        "Are Team Members permitted to disable automatic screen lock?",
        "may a user override the inactivity lockout setting",
    )


def test_parse_drops_preamble_and_caps_at_two():
    raw = "Here are two versions:\nfirst version\nsecond version\nthird version"
    assert qr.parse_rephrasings(raw, Q) == ("first version", "second version")


def test_parse_deduplicates_case_insensitively():
    assert qr.parse_rephrasings("Same thing\nsame thing\n", Q) == ("Same thing",)


def test_parse_of_nothing_usable_is_empty():
    assert qr.parse_rephrasings("\n  \n", Q) == ()


def test_passage_is_one_whitespace_normalised_query():
    assert qr.parse_passage("  Team Members must\n not disable\tthe lock.  ") == (
        "Team Members must not disable the lock.",
    )
    assert qr.parse_passage("   ") == ()


# --- rewrite_query --------------------------------------------------------------


class _LLM:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.messages = text, exc, None

    def chat(self, messages):
        self.messages = messages
        if self.exc:
            raise self.exc
        return type("R", (), {"message": type("M", (), {"content": self.text})()})()


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(qr, "get_tracer", lambda: provider.get_tracer("t"))
    return exporter


def _use(monkeypatch, llm, mode):
    monkeypatch.setattr(qr.settings, "query_rewrite", mode)
    seen = {}

    def _get_llm(model=None, timeout=None):
        seen["timeout"] = timeout
        return llm

    monkeypatch.setattr(qr, "get_llm", _get_llm)
    return seen


def test_off_makes_no_llm_call_and_no_span(monkeypatch, spans):
    def _boom(*a, **kw):
        raise AssertionError("no LLM call when off")

    monkeypatch.setattr(qr.settings, "query_rewrite", "off")
    monkeypatch.setattr(qr, "get_llm", _boom)

    assert qr.rewrite_query(Q) == qr.RewriteResult("off")
    assert spans.get_finished_spans() == ()


def test_multi_returns_two_rephrasings_with_the_question_as_user_message(monkeypatch, spans):
    llm = _LLM("alt one\nalt two")
    seen = _use(monkeypatch, llm, "multi")
    monkeypatch.setattr(qr.settings, "query_rewrite_timeout", 20)

    r = qr.rewrite_query(Q)

    assert r.mode == "multi" and r.queries == ("alt one", "alt two") and not r.fallback
    assert seen["timeout"] == 20
    assert llm.messages[0].content == qr.REWRITE_PROMPTS["multi"]
    assert llm.messages[1].content == Q
    (span,) = spans.get_finished_spans()
    assert span.name == "query_rewrite"
    assert span.attributes["query_rewrite.mode"] == "multi"
    assert list(span.attributes["query_rewrite.queries"]) == ["alt one", "alt two"]
    assert span.attributes["query_rewrite.fallback"] is False


def test_multi_titles_puts_the_corpus_titles_in_the_prompt(monkeypatch, spans):
    llm = _LLM("alt one\nalt two")
    _use(monkeypatch, llm, "multi_titles")
    monkeypatch.setattr(qr, "policy_titles", lambda: ["Access Management Policy [Internal]", "Clear Desk And Clear Screen Policy [Internal]"])

    qr.rewrite_query(Q)

    system = llm.messages[0].content
    assert "- Access Management Policy [Internal]\n- Clear Desk And Clear Screen Policy [Internal]" in system
    assert "{titles}" not in system


def test_a_titles_lookup_failure_falls_back_without_calling_the_llm(monkeypatch, spans):
    def _down():
        raise ConnectionError("qdrant down")

    llm = _LLM("unused")
    _use(monkeypatch, llm, "multi_titles")
    monkeypatch.setattr(qr, "policy_titles", _down)

    r = qr.rewrite_query(Q)

    assert r.fallback and r.queries == () and "ConnectionError" in r.error
    assert llm.messages is None


def test_hyde_returns_one_passage(monkeypatch, spans):
    _use(monkeypatch, _LLM("Team Members must not\ndisable the screen lock."), "hyde")
    assert qr.rewrite_query(Q).queries == ("Team Members must not disable the screen lock.",)


@pytest.mark.parametrize("exc", [TimeoutError("slow"), ConnectionError("refused"), RuntimeError("Event loop is closed")])
def test_any_llm_exception_falls_back_to_the_original(monkeypatch, spans, exc):
    _use(monkeypatch, _LLM(exc=exc), "multi")

    r = qr.rewrite_query(Q)

    assert r.fallback and r.queries == () and type(exc).__name__ in r.error
    (span,) = spans.get_finished_spans()
    assert span.attributes["query_rewrite.fallback"] is True


def test_unusable_output_falls_back(monkeypatch, spans):
    _use(monkeypatch, _LLM(f"{Q}\n"), "multi")
    r = qr.rewrite_query(Q)
    assert r.fallback and r.error == "no usable rephrasing"


def test_the_prompts_forbid_answering_and_inventing():
    for mode in ("multi", "multi_titles"):
        assert "Do not answer" in qr.REWRITE_PROMPTS[mode]
        assert "Do not add" in qr.REWRITE_PROMPTS[mode]
```

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/unit/test_query_rewrite.py -q -p no:warnings` → FAIL (`No module named 'rag.query_rewrite'`).

- [ ] **Step 3: Implement** — `rag/query_rewrite.py`:

```python
"""Rephrase a question for retrieval only (design 2026-10-09).

The verbatim question is always searched; this adds variants that use the
policy's vocabulary where the user did not. Nothing here reaches the reranker
or the agent -- both keep the user's own words.

One plain-text LLM call at temperature 0: no tools and no structured output,
because constrained decoding is one of the five conditions of the Ollama MoE
crash (docs/superpowers/specs/2026-09-17-ollama-moe-cuda-crash.md). Never
raises: whatever goes wrong, the search runs with the original question alone
and the span says why.
"""

import re
import time
from dataclasses import asdict, dataclass

from llama_index.core.llms import ChatMessage, MessageRole
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from config import settings
from rag.agent import get_llm
from rag.observability import get_tracer
from rag.vector_store import policy_titles

_MULTI = """\
You rewrite an employee's question so it can be searched against internal company policy documents.

Write exactly 2 alternative versions of the question. Each version must:
- keep the original meaning and every specific detail (who, what, which system or situation);
- use the formal wording a written company policy would use for the same thing \
(for example "Team Member", "corporate workstation", "personal data breach", "approval", "prohibited");
- be a single line: a question or a search phrase.

Do not answer the question. Do not add facts, conditions, policies or details that are not in the question.
Output only the 2 lines, with no numbering and nothing else."""

_TITLES = """

The policy documents that can be searched are:
{titles}
Use their vocabulary where it fits. Do not name a policy unless the question is clearly about it."""

_HYDE = """\
Write one short paragraph (2-3 sentences) in the style of an internal company policy clause \
that would answer the employee's question. Use formal policy wording \
(for example "Team Members must ...", "... is prohibited unless approved by ...").
It is used only to search for the real clause: do not hedge, do not mention that it is \
hypothetical, and do not address the employee. Output only the paragraph."""

REWRITE_PROMPTS = {
    "multi": _MULTI,
    "multi_titles": _MULTI + _TITLES,
    "hyde": _HYDE,
}

_MAX_REPHRASINGS = 2
_MAX_QUERY_CHARS = 300
_MAX_PASSAGE_CHARS = 1000
_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


@dataclass(frozen=True)
class RewriteResult:
    mode: str
    queries: tuple[str, ...] = ()  # extra queries only -- the original is never in here
    fallback: bool = False  # rewriting was on but produced nothing usable
    error: str = ""
    latency_ms: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["queries"] = list(self.queries)
        return d


def parse_rephrasings(raw: str, original: str) -> tuple[str, ...]:
    """At most 2 distinct lines, minus list markers, quotes, preambles and the original."""
    seen = {original.strip().casefold()}
    out = []
    for line in raw.splitlines():
        line = _MARKER.sub("", line).strip().strip('"\'“”‘’').strip()
        if not line or line.endswith(":"):
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(line[:_MAX_QUERY_CHARS])
        if len(out) == _MAX_REPHRASINGS:
            break
    return tuple(out)


def parse_passage(raw: str) -> tuple[str, ...]:
    passage = " ".join(raw.split())
    return (passage[:_MAX_PASSAGE_CHARS],) if passage else ()


def rewrite_query(question: str) -> RewriteResult:
    mode = settings.query_rewrite
    if mode == "off":
        return RewriteResult("off")

    tracer = get_tracer()
    with tracer.start_as_current_span(
        "query_rewrite",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "query_rewrite.mode": mode,
        },
    ) as span:
        started = time.monotonic()
        try:
            system = REWRITE_PROMPTS[mode]
            if mode == "multi_titles":
                system = system.format(titles="\n".join(f"- {t}" for t in policy_titles()))
            llm = get_llm(timeout=settings.query_rewrite_timeout)
            response = llm.chat(
                [
                    ChatMessage(role=MessageRole.SYSTEM, content=system),
                    ChatMessage(role=MessageRole.USER, content=question),
                ]
            )
            raw = str(response.message.content or "")
            queries = parse_passage(raw) if mode == "hyde" else parse_rephrasings(raw, question)
            error = "" if queries else "no usable rephrasing"
        except Exception as exc:  # never blocks: any failure means "search the original alone"
            queries, error = (), f"{type(exc).__name__}: {exc}"[:200]
        result = RewriteResult(
            mode=mode,
            queries=queries,
            fallback=not queries,
            error=error,
            latency_ms=round((time.monotonic() - started) * 1000),
        )
        span.set_attribute("query_rewrite.queries", list(result.queries))
        span.set_attribute("query_rewrite.fallback", result.fallback)
        span.set_attribute("query_rewrite.latency_ms", result.latency_ms)
        if result.error:
            span.set_attribute("query_rewrite.error", result.error)
        return result
```

Note: `str.format` on `_MULTI + _TITLES` is safe only while `_MULTI` contains no `{`/`}`. The test `test_multi_titles_puts_the_corpus_titles_in_the_prompt` would raise `KeyError`/`IndexError` if someone adds braces — that is the guard.

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/unit/test_query_rewrite.py -q -p no:warnings` → PASS. Then `.venv/bin/python -m pytest tests/unit/test_no_undefined_names.py -q -p no:warnings` → PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/query_rewrite.py tests/unit/test_query_rewrite.py
git commit -m "feat(rag): query_rewrite — multi, multi_titles and hyde rephrasings that never block"
```

---

### Task 4: Wire the rewrite into `search_policies`

**Files:**
- Modify: `rag/tools/search_policies.py` (imports; globals; Step 1 of `search_policies`; `score_type`)
- Test: `tests/unit/test_search_with_rewrite.py` (create)

**Interfaces:**
- Consumes: `rewrite_query(question) -> RewriteResult` (Task 3), `embed_queries` (Task 1), `search_chunks(..., extra_queries=...)` (Task 2).
- Produces: module global `sp._last_rewrite: dict` — `RewriteResult.as_dict()` of the latest call (`{}` before any call; `{"mode": "off", ...}` when off).

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_search_with_rewrite.py`:

```python
"""search_policies with the rewrite on: one embed call, extras fused, original reranked.

Patched on `sp` (names bound at import). Nothing touches the network.
"""

import httpx
import pytest

import rag.tools.search_policies as sp
from rag.query_rewrite import RewriteResult


class _Hit:
    score = 0.016
    id = "c1"
    payload = {
        "doc_title": "Access Management Policy [Internal]",
        "doc_id": "access-management-policy-internal",
        "section": "Access Management",
        "clause": "Inactivity Logoff/Lockout",
        "clause_number": "4.14",
        "text": "Users are prohibited from disabling automatic lock settings.",
    }


@pytest.fixture
def wired(monkeypatch):
    calls = {}
    monkeypatch.setattr(sp.settings, "reranker_enabled", True)
    monkeypatch.setattr(sp.settings, "bm25_enabled", False)
    monkeypatch.setattr(sp.settings, "reranker_min_score", 0.0)
    monkeypatch.setattr(sp, "rewrite_query", lambda q: RewriteResult("multi", ("alt one", "alt two")))

    def _embed_queries(qs):
        calls["embed"] = list(qs)
        return [[float(i)] for i, _ in enumerate(qs)]

    def _search(q, v, top_k, extra_queries=()):
        calls["search"] = (q, v, top_k, list(extra_queries))
        return [_Hit()]

    def _rerank(query, results, top_n):
        calls["rerank_query"] = query
        return [dict(results[0], rerank_score=0.9)]

    monkeypatch.setattr(sp, "embed_queries", _embed_queries)
    monkeypatch.setattr(sp, "search_chunks", _search)
    monkeypatch.setattr(sp, "rerank", _rerank)
    return calls


def test_original_and_rephrasings_are_embedded_in_one_call(wired):
    sp.search_policies("orig question")
    assert wired["embed"] == ["orig question", "alt one", "alt two"]


def test_rephrasings_go_to_qdrant_as_extras_beside_the_original(wired):
    sp.search_policies("orig question")
    q, v, _top_k, extras = wired["search"]
    assert (q, v) == ("orig question", [0.0])
    assert extras == [("alt one", [1.0]), ("alt two", [2.0])]


def test_the_reranker_judges_against_the_original_question(wired):
    sp.search_policies("orig question")
    assert wired["rerank_query"] == "orig question"


def test_the_rewrite_is_kept_for_eval(wired):
    sp.search_policies("orig question")
    assert sp._last_rewrite["mode"] == "multi"
    assert sp._last_rewrite["queries"] == ["alt one", "alt two"]


def test_rewritten_search_reports_rrf_scores_even_with_bm25_off(wired):
    sp.search_policies("orig question")
    assert sp._last_search_results[0]["score_type"] == "rrf"


def test_a_fallback_rewrite_searches_exactly_as_rewrite_off(monkeypatch, wired):
    """No extras -> the pre-rewrite call shape: embed_query + 3-arg search_chunks."""
    monkeypatch.setattr(sp, "rewrite_query", lambda q: RewriteResult("multi", fallback=True, error="x"))
    monkeypatch.setattr(sp, "embed_queries", lambda qs: pytest.fail("batch path used on fallback"))
    monkeypatch.setattr(sp, "embed_query", lambda q: [9.0])
    monkeypatch.setattr(sp, "search_chunks", lambda q, v, top_k: [_Hit()])

    out = sp.search_policies("orig question")

    assert "[Source 1]" in out
    assert sp._last_rewrite["fallback"] is True
    assert sp._last_search_results[0]["score_type"] == "cosine"


def test_embedding_outage_with_rewrite_on_is_still_unavailable(monkeypatch, wired):
    recorded = []

    def _down(qs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(sp, "embed_queries", _down)
    monkeypatch.setattr(sp, "RETRY_BACKOFFS", ())
    monkeypatch.setattr(sp, "retry_transient", lambda fn: fn())
    monkeypatch.setattr(sp, "record_infra_unavailable", lambda *a: recorded.append(a))

    assert sp.search_policies("orig question") == sp.UNAVAILABLE
    assert sp._retrieval_unavailable is True
    assert recorded[0][0] == "embeddings"
```

`RETRY_BACKOFFS` and `retry_transient` are module-level names in `search_policies` (imported at top), and `httpx.ConnectError` is in `rag/resilience.py` `_TRANSIENT_TYPES`.

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/unit/test_search_with_rewrite.py -q -p no:warnings` → FAIL (`module has no attribute 'rewrite_query'`).

- [ ] **Step 3: Implement.** In `rag/tools/search_policies.py`:

Imports: change `from rag.embeddings import embed_query` to `from rag.embeddings import embed_queries, embed_query` and add `from rag.query_rewrite import rewrite_query`.

Globals, after `_retrieval_unavailable: bool = False`:

```python
# The latest call's rewrite (RewriteResult.as_dict()), read by eval like
# _last_search_results. Same single-worker reset-then-read contract.
_last_rewrite: dict = {}
```

In `search_policies`, add `_last_rewrite` to the `global` line. Replace the Step 1 embedding block (the `try: query_vector = retry_transient(lambda: embed_query(query))` block) with:

```python
    # Step 0: Rephrase for retrieval (QUERY_REWRITE; off by default). The result
    # only ever ADDS queries -- the original is searched regardless -- and a
    # failed rewrite has already fallen back to "no extras" inside rewrite_query.
    rewrite = rewrite_query(query)
    _last_rewrite = rewrite.as_dict()

    # Step 1: Retrieve candidates — dense only, or dense + sparse fused by Qdrant.
    # With no extras this is exactly the pre-rewrite call shape, call for call.
    try:
        if rewrite.queries:
            vectors = retry_transient(lambda: embed_queries([query, *rewrite.queries]))
            query_vector = vectors[0]
            extra_queries = list(zip(rewrite.queries, vectors[1:]))
        else:
            query_vector = retry_transient(lambda: embed_query(query))
            extra_queries = []
    except Exception as exc:
        if is_transient(exc):
            _last_search_results = []
            _retrieval_unavailable = True
            record_infra_unavailable("embeddings", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    try:
        if extra_queries:
            raw = retry_transient(
                lambda: search_chunks(query, query_vector, top_k=retrieve_k, extra_queries=extra_queries)
            )
        else:
            raw = retry_transient(lambda: search_chunks(query, query_vector, top_k=retrieve_k))
```

(keep the existing `except` of the Qdrant block unchanged). In the `results = [...]` comprehension replace

```python
            "score_type": "rrf" if settings.bm25_enabled else "cosine",
```

with

```python
            # Fused whenever there is more than one query, BM25 or not.
            "score_type": "rrf" if settings.bm25_enabled or extra_queries else "cosine",
```

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/unit/test_search_with_rewrite.py tests/unit/test_search_one_path.py tests/unit/test_search_floor.py tests/unit/test_search_first.py tests/unit/test_retrieval_spans.py tests/unit/test_request_span.py -q -p no:warnings` → all PASS (the existing ones unmodified — the off path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add rag/tools/search_policies.py tests/unit/test_search_with_rewrite.py
git commit -m "feat(retrieval): search_policies searches the question plus its rewrite"
```

---

### Task 5: Rewrite identity in experiments and production spans

**Files:**
- Modify: `rag/run_identity.py`
- Modify: `channels/teams/bot.py` (`_identity_attributes`)
- Modify: `eval/run_experiment.py` (`infra_meta`)
- Test: `tests/unit/test_run_identity.py` (extend)

**Interfaces:**
- Consumes: `REWRITE_PROMPTS` (Task 3), `settings.query_rewrite`.
- Produces: `rewrite_identity() -> dict[str, str]` = `{"query_rewrite": <mode>, "rewrite_prompt_sha12": <sha12 of REWRITE_PROMPTS[mode] template, "" when off>}`. Bot attributes `identity.query_rewrite`, `identity.rewrite_prompt_sha12`. Experiment metadata keys `query_rewrite`, `rewrite_prompt_sha12` (both tiers).

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_run_identity.py`:

```python
# --- query rewrite -----------------------------------------------------------------

import rag.query_rewrite as qr  # noqa: E402  (kept beside the tests that need it)


def test_rewrite_off_has_no_prompt_hash(monkeypatch):
    monkeypatch.setattr(ri.settings, "query_rewrite", "off")
    assert ri.rewrite_identity() == {"query_rewrite": "off", "rewrite_prompt_sha12": ""}


@pytest.mark.parametrize("mode", ["multi", "multi_titles", "hyde"])
def test_each_rewrite_mode_is_identified_by_its_own_prompt(monkeypatch, mode):
    monkeypatch.setattr(ri.settings, "query_rewrite", mode)
    assert ri.rewrite_identity() == {
        "query_rewrite": mode,
        "rewrite_prompt_sha12": ri.sha12(qr.REWRITE_PROMPTS[mode]),
    }


def test_editing_the_rewrite_prompt_changes_its_hash(monkeypatch):
    monkeypatch.setattr(ri.settings, "query_rewrite", "multi")
    before = ri.rewrite_identity()["rewrite_prompt_sha12"]
    monkeypatch.setitem(qr.REWRITE_PROMPTS, "multi", qr.REWRITE_PROMPTS["multi"] + " ")
    assert ri.rewrite_identity()["rewrite_prompt_sha12"] != before


def test_production_spans_carry_the_rewrite_identity(monkeypatch):
    monkeypatch.setattr(ri.settings, "query_rewrite", "multi")
    b = object.__new__(bot.TeamsBot)
    b._llm_digest_at_startup = ""
    attrs = b._identity_attributes()
    assert attrs["identity.query_rewrite"] == "multi"
    assert attrs["identity.rewrite_prompt_sha12"] == ri.rewrite_identity()["rewrite_prompt_sha12"]


def test_experiments_record_the_rewrite_identity_on_every_tier():
    src = inspect.getsource(rx.main)
    assert "**rewrite_identity()" in src
    assert src.index("**rewrite_identity()") < src.index('if args.tier == "tier1"')
```

(Move the `import rag.query_rewrite as qr` line up into the file's top import block instead of keeping it mid-file, to follow imports-at-top; drop the `noqa` then.)

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/unit/test_run_identity.py -q -p no:warnings` → FAIL (`no attribute 'rewrite_identity'`).

- [ ] **Step 3: Implement.**

`rag/run_identity.py` — add `import rag.query_rewrite as query_rewrite` to the imports and:

```python
def rewrite_identity() -> dict[str, str]:
    """Which query-rewrite mode ran, and the exact prompt template it used.

    The template, not the rendered prompt: `multi_titles` fills in titles read
    from the collection, and the collection is already in the metadata. Reading
    it here would put a Qdrant call on every request's span.
    """
    mode = settings.query_rewrite
    prompt = query_rewrite.REWRITE_PROMPTS.get(mode, "")
    return {"query_rewrite": mode, "rewrite_prompt_sha12": sha12(prompt) if prompt else ""}
```

`channels/teams/bot.py` `_identity_attributes` — extend the deferred import to `from rag.run_identity import prompt_identity, rewrite_identity, router_prompt_sha12` and add to the returned dict:

```python
            **{f"identity.{k}": v for k, v in rewrite_identity().items()},
```

`eval/run_experiment.py` `main()` — before `infra_meta = {`, add `from rag.run_identity import rewrite_identity` (local import, this file's convention), and inside `infra_meta` after `"git_commit": git_commit,` add:

```python
        # Applies to every tier: tier1 now retrieves through search_policies,
        # which is where the rewrite runs.
        **rewrite_identity(),
```

Also add `print(f"  Rewrite:     {settings.query_rewrite}")` after the `Reranker:` print.

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/unit/test_run_identity.py tests/unit/test_eval_metadata.py tests/unit/test_build_identity.py tests/unit/test_no_undefined_names.py -q -p no:warnings` → PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/run_identity.py channels/teams/bot.py eval/run_experiment.py tests/unit/test_run_identity.py
git commit -m "feat(identity): record the query-rewrite mode and prompt hash"
```

---

### Task 6: Tier1 retrieves through `search_policies`; eval outputs carry the rewrite

**Files:**
- Modify: `eval/run_experiment.py` (`make_tier1_task`; agent task's `_agent_metadata`)
- Modify: `eval/agent_wrapper.py` (`prefetch_logged`)
- Test: `tests/unit/test_tier1_one_path.py` (create)

**Interfaces:**
- Consumes: `sp.search_policies(query, top_k)`, `sp._last_search_results`, `sp._last_rewrite`, `sp._retrieval_unavailable`.
- Produces: tier1 task output `{"search_results": [...], "rewrite": dict}`; raises `RuntimeError` when retrieval is unavailable. Agent log entry gains `"rewrite": dict`; `agent_metadata` gains `"rewrite"` (the first search's rewrite dict, `{}` if none).

Why: tier1 today re-implements embed → `search_chunks` → rerank. With the rewrite inside `search_policies`, that copy would measure a pipeline that does not ship. Routing tier1 through `search_policies` also means tier1 now sees the relevance floors exactly as production does (`_last_search_results` stays populated on a reranker-floor rejection — `test_a_rejected_search_still_reports_what_it_found`).

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_tier1_one_path.py`:

```python
"""Tier1 measures the retrieval that ships: it calls search_policies."""

import inspect

import pytest

import eval.run_experiment as rx
import rag.tools.search_policies as sp


def test_tier1_has_no_private_copy_of_retrieval():
    src = inspect.getsource(rx.make_tier1_task)
    assert "search_policies(" in src
    for name in ("embed_query", "search_chunks", "rerank("):
        assert name not in src, name


def test_tier1_returns_what_search_policies_found_and_the_rewrite(monkeypatch):
    def _fake(query, top_k=6):
        sp._retrieval_unavailable = False
        sp._last_search_results = [{"doc_title": "D", "section": "S", "clause": "C", "clause_number": "1.1",
                                    "rerank_score": 0.9, "retrieval_score": 0.016, "score_type": "rrf"}]
        sp._last_rewrite = {"mode": "multi", "queries": ["a", "b"], "fallback": False, "error": "", "latency_ms": 7}
        return "formatted"

    monkeypatch.setattr(sp, "search_policies", _fake)
    out = rx.make_tier1_task(top_k=25)({"question": "q"})

    assert out["search_results"][0]["doc_title"] == "D"
    assert out["rewrite"]["queries"] == ["a", "b"]


def test_tier1_raises_when_retrieval_is_unavailable(monkeypatch):
    """A swallowed outage would score 0.0 and read as bad retrieval."""
    def _down(query, top_k=6):
        sp._retrieval_unavailable = True
        sp._last_search_results = []
        return sp.UNAVAILABLE

    monkeypatch.setattr(sp, "search_policies", _down)
    with pytest.raises(RuntimeError, match="unavailable"):
        rx.make_tier1_task(top_k=25)({"question": "q"})
```

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/unit/test_tier1_one_path.py -q -p no:warnings` → FAIL.

- [ ] **Step 3: Implement** — replace the body of `make_tier1_task` in `eval/run_experiment.py` with:

```python
def make_tier1_task(top_k: int):
    # Local import: init_observability() must run before any LlamaIndex/Ollama
    # import, and search_policies pulls llama_index.
    import rag.tools.search_policies as sp

    def retrieval_task(input):
        # The same call production makes -- rewrite, embed, fused search, rerank,
        # floors. Tier1 used to carry its own copy of that path, which stopped
        # being the shipped one the moment a step (the query rewrite) was added
        # inside search_policies. `top_k` only matters with the reranker off.
        sp.search_policies(input["question"], top_k=top_k)
        if sp._retrieval_unavailable:
            raise RuntimeError("retrieval unavailable (embeddings/qdrant) -- not a retrieval miss")
        return {
            "search_results": [dict(r) for r in sp._last_search_results],
            "rewrite": dict(sp._last_rewrite),
        }

    return retrieval_task
```

Note: `sp` here is a module attribute lookup at call time, so the test's `monkeypatch.setattr(sp, "search_policies", ...)` is seen. Remove the now-unused `retrieve_k` / `settings` lines from the old body.

`eval/agent_wrapper.py` `prefetch_logged` — add to the appended dict:

```python
            "rewrite": dict(sp._last_rewrite),
```

`eval/run_experiment.py` `_agent_metadata` (in `make_agent_task`, ~line 131) — add a key:

```python
                "rewrite": next((c.get("rewrite", {}) for c in tool_calls if c["tool"] == "search_policies"), {}),
```

(`tool_calls = list(get_log())` is already in scope there.)

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/unit -q -p no:warnings -k "tier1 or eval or agent_wrapper or one_candidate or no_undefined"` → PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/run_experiment.py eval/agent_wrapper.py tests/unit/test_tier1_one_path.py
git commit -m "fix(eval): tier1 retrieves through search_policies, and outputs carry the rewrite"
```

---

### Task 7: Per-question comparison of experiments

**Files:**
- Create: `eval/compare_runs.py`
- Test: `tests/unit/test_compare_runs.py`

**Interfaces:**
- Consumes: `eval.evaluators._extract_expected`, `eval.evaluators._match_result`; Phoenix `GET {base}/v1/experiments/{id}/json` → list of `{"input": {"question"}, "reference_output": {...}, "output": {"search_results": [...], "rewrite": {...}}, "error": ...}`.
- Produces: `first_match_rank(output: dict | None, reference: dict) -> int | None`; `compare(baseline: list[dict], candidate: list[dict]) -> dict` with keys `recovered`, `lost`, `better`, `worse`, `same` (each a list of `(question, base_rank, cand_rank)`), `latency_ms` (list of ints from `output.rewrite.latency_ms`), `fallbacks` (int); CLI `python eval/compare_runs.py <baseline_id> <candidate_id> [--phoenix-url URL]`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_compare_runs.py`:

```python
"""The decision rests on per-question movement, not means (design, Measurement 2)."""

from eval.compare_runs import compare, first_match_rank

REF = {"expected_citations": [{"doc_id": "Doc A", "section": "Sec", "clause": "Cl"}]}


def _run(q, ranks_doc, rewrite=None):
    results = [{"doc_title": d, "section": "Sec", "clause": "Cl"} for d in ranks_doc]
    return {"input": {"question": q}, "reference_output": REF,
            "output": {"search_results": results, "rewrite": rewrite or {}}}


def test_rank_is_one_based_and_none_when_absent():
    assert first_match_rank({"search_results": [{"doc_title": "X"}, {"doc_title": "Doc A", "section": "Sec", "clause": "Cl"}]}, REF) == 2
    assert first_match_rank({"search_results": [{"doc_title": "X"}]}, REF) is None
    assert first_match_rank(None, REF) is None


def test_compare_classifies_each_question():
    base = [_run("recovered", ["X"]), _run("lost", ["Doc A"]), _run("better", ["X", "Doc A"]),
            _run("worse", ["Doc A"]), _run("same", ["Doc A"])]
    cand = [_run("recovered", ["Doc A"], {"latency_ms": 900}), _run("lost", ["X"], {"latency_ms": 1100}),
            _run("better", ["Doc A"]), _run("worse", ["X", "Doc A"]), _run("same", ["Doc A"], {"fallback": True})]

    c = compare(base, cand)

    assert [q for q, *_ in c["recovered"]] == ["recovered"]
    assert [q for q, *_ in c["lost"]] == ["lost"]
    assert [q for q, *_ in c["better"]] == ["better"]
    assert [q for q, *_ in c["worse"]] == ["worse"]
    assert [q for q, *_ in c["same"]] == ["same"]
    assert sorted(c["latency_ms"]) == [900, 1100]
    assert c["fallbacks"] == 1


def test_questions_are_matched_by_text_not_position():
    base = [_run("a", ["Doc A"]), _run("b", ["X"])]
    cand = [_run("b", ["Doc A"]), _run("a", ["Doc A"])]
    c = compare(base, cand)
    assert [q for q, *_ in c["recovered"]] == ["b"]
    assert [q for q, *_ in c["same"]] == ["a"]
```

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/unit/test_compare_runs.py -q -p no:warnings` → FAIL (no module).

- [ ] **Step 3: Implement** — `eval/compare_runs.py`:

```python
#!/usr/bin/env python3
"""Compare two Phoenix experiments question by question.

    python eval/compare_runs.py <baseline_experiment_id> <candidate_experiment_id> \
        [--phoenix-url http://phoenix:6006]

With 61 questions, one recovered miss and one new miss average to "no
change". The query-rewrite decision (design 2026-10-09, Measurement 2 and the
stop rule) is made on this listing: misses recovered, working questions lost,
rank moves, and what the rewrite cost in latency. Read-only against Phoenix.
"""

import argparse
import statistics
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluators import _extract_expected, _match_result  # noqa: E402


def first_match_rank(output: dict | None, reference: dict) -> int | None:
    if not output:
        return None
    expectations = _extract_expected(reference)
    for rank, r in enumerate(output.get("search_results") or [], start=1):
        if any(_match_result(r, e) for e in expectations):
            return rank
    return None


def compare(baseline: list[dict], candidate: list[dict]) -> dict:
    base = {r["input"]["question"]: r for r in baseline}
    out = {k: [] for k in ("recovered", "lost", "better", "worse", "same")}
    latency, fallbacks = [], 0
    for run in candidate:
        q = run["input"]["question"]
        if q not in base:
            continue
        b = first_match_rank(base[q]["output"], base[q]["reference_output"])
        c = first_match_rank(run["output"], run["reference_output"])
        rewrite = (run["output"] or {}).get("rewrite") or {}
        if "latency_ms" in rewrite:
            latency.append(rewrite["latency_ms"])
        fallbacks += bool(rewrite.get("fallback"))
        if b is None and c is not None:
            key = "recovered"
        elif b is not None and c is None:
            key = "lost"
        elif b == c:
            key = "same"
        else:
            key = "better" if c < b else "worse"
        out[key].append((q, b, c))
    out["latency_ms"] = latency
    out["fallbacks"] = fallbacks
    return out


def _fetch(base_url: str, experiment_id: str) -> list[dict]:
    resp = requests.get(f"{base_url.rstrip('/')}/v1/experiments/{experiment_id}/json", timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--phoenix-url", default="http://localhost:6006")
    a = p.parse_args()

    c = compare(_fetch(a.phoenix_url, a.baseline), _fetch(a.phoenix_url, a.candidate))
    for key in ("recovered", "lost", "better", "worse"):
        print(f"\n{key.upper()} ({len(c[key])})")
        for q, b, n in c[key]:
            print(f"  {str(b):>4} -> {str(n):<4} {q}")
    print(f"\nSAME: {len(c['same'])}   rewrite fallbacks: {c['fallbacks']}")
    if c["latency_ms"]:
        lat = sorted(c["latency_ms"])
        p95 = lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))]
        print(f"rewrite latency ms: p50 {statistics.median(lat):.0f}  p95 {p95}  max {lat[-1]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/unit/test_compare_runs.py tests/unit/test_no_undefined_names.py -q -p no:warnings` → PASS. Check `test_no_undefined_names` scans `eval/*.py`; if it flags `__file__` (it flagged it in `run_experiment.py` on 2026-10-08), it will not here — `__file__` is read at module level only.

- [ ] **Step 5: Commit**

```bash
git add eval/compare_runs.py tests/unit/test_compare_runs.py
git commit -m "feat(eval): compare two experiments question by question"
```

---

### Task 8: Docs, full suite, mutation check

**Files:**
- Modify: `CLAUDE.md` (Architecture `rag/` block; Search flow paragraph; Config block)
- Modify: `SETUP.md` Step 11 (rewrite runs)
- Modify: `docs/superpowers/specs/2026-10-09-query-rephrasing-design.md` (span name)

- [ ] **Step 1: CLAUDE.md.** In the Architecture `rag/` block, after `search_first.py`, add `  query_rewrite.py   # QUERY_REWRITE: LLM rephrasings for retrieval only — off by default; never blocks`. In **Search flow**, after "`prefetch()` (rag/search_first.py) runs `search_policies` BEFORE the agent — " insert "`[rewrite_query → +2 rephrasings, QUERY_REWRITE, off by default]` → ". In the Config bash block add a line `QUERY_REWRITE=off|multi|multi_titles|hyde (off)  QUERY_REWRITE_TIMEOUT (20s)`. Keep CLAUDE.md ≤ 150 lines (`wc -l CLAUDE.md`).

- [ ] **Step 2: SETUP.md Step 11.** After the `-e GIT_COMMIT` paragraph add:

````markdown
### Query-rewrite comparison (design 2026-10-09)

Retrieval only, over the chatbot questions, once per mode — no answering model:

```bash
for MODE in off multi multi_titles hyde; do
  docker compose -f docker-compose-remote.yml run --rm $EVAL \
    -e GIT_COMMIT=$(git -C /home/sa.ivanov/rag-eval describe --always --dirty) \
    -e QUERY_REWRITE=$MODE \
    --entrypoint python bot eval/run_experiment.py \
    --tier tier1 --dataset chatbot-test-v1 --name rewrite-$MODE --phoenix-url http://phoenix:6006
done
```

Then compare each against `rewrite-off` (ids from Phoenix → Datasets & Experiments):

```bash
docker compose -f docker-compose-remote.yml run --rm $EVAL --entrypoint python bot \
  eval/compare_runs.py <rewrite-off id> <rewrite-multi id> --phoenix-url http://phoenix:6006
```

Decide on the RECOVERED / LOST lists, not the means. Stop rule: no mode that
recovers ≥ 2 of the 4 misses without losing a working question → rewriting is
not worth its latency on this corpus.
````

- [ ] **Step 3: Spec.** In the design doc's Rules item 6, replace "The rephrasings go on the `search_vectors` span (`query_rewrite.mode`, `.queries`, `.fallback`, `.latency_ms`)" with "The rephrasings go on their own `query_rewrite` span (`query_rewrite.mode`, `.queries`, `.fallback`, `.latency_ms`, `.error`), which also parents the rewrite's LLM span; `search_vectors` gains `qdrant.query_count`".

- [ ] **Step 4: Full suite.**

Run: `.venv/bin/python -m pytest tests/unit tests/load -p no:warnings -q --junitxml=/tmp/j.xml >/dev/null 2>&1; echo exit=$?; grep -o '<testsuite [^>]*' /tmp/j.xml | grep -oE '(tests|failures|errors)="[0-9]+"'`
Expected: `exit=0`, failures 0, errors 0 (scratchpad path instead of `/tmp` when running in Claude Code).

- [ ] **Step 5: Mutation check** — each mutation must turn at least one test red; restore after each:
  1. `search_policies`: pass `rewrite.queries[0]` to `rerank` instead of `query` → `test_the_reranker_judges_against_the_original_question` fails.
  2. `config.cosine_floor_applies`: drop `and self.query_rewrite == "off"` → `test_a_rewritten_search_is_never_judged_by_the_cosine_floor` fails.
  3. `query_rewrite.parse_rephrasings`: remove `seen = {original…}` seeding (use `set()`) → `test_parse_strips_markers_quotes_and_the_original` fails.
  4. `query_rewrite.rewrite_query`: re-raise in the `except` → `test_any_llm_exception_falls_back_to_the_original` fails.
  5. `vector_store.search_chunks`: build prefetch only for the original → `test_each_extra_query_adds_a_dense_and_a_sparse_prefetch` fails.
  6. `eval/run_experiment.make_tier1_task`: drop the `_retrieval_unavailable` check → `test_tier1_raises_when_retrieval_is_unavailable` fails.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md SETUP.md docs/superpowers/specs/2026-10-09-query-rephrasing-design.md
git commit -m "docs: query rewrite — architecture, config, and the VM comparison runbook"
```

---

## After the plan (owner, on the VM)

1. Merge, push, deploy (`QUERY_REWRITE` unset → `off`: production behaviour unchanged; the banner/identity shows `identity.query_rewrite=off`).
2. Run the four tier1 runs and the three comparisons from SETUP.md Step 11.
3. Apply the stop rule; record the outcome in the design doc.
