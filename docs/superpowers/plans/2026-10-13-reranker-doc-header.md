# Reranker document header Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A setting `RERANKER_DOC_HEADER` (default off) that makes the reranker score each candidate as `Document: … | Section: … | Clause: …\n<text>` — the same header the dense embedding was built from — so it can be measured end to end before anyone turns it on.

**Architecture:** One shared function builds the header; ingest uses it for the embedding text (byte-identical to today) and the reranker uses it for the scored text when the setting is on. Only the text sent to the reranker changes: the results it returns, the sources the agent sees, and the span's document content stay bare text. Eval metadata and the `rerank` span record the setting so runs and traces can be told apart.

**Tech Stack:** Python 3, pydantic-settings, httpx, OpenTelemetry, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-query-rephrasing-design.md`, section "Does the reranker do better when it sees the chunk's header? Slightly." (read-only A/B: 57→58/61 in top 6, recovered 1, lost 0, better 9 / worse 7).

**Base:** branched from `feat/query-rewrite` (not main) because the measurement needs that branch's tier1-through-`search_policies` and `eval/compare_runs.py`. Merging this branch therefore brings the query-rewrite branch with it; that branch ships `QUERY_REWRITE=off`, so production behaviour is unchanged either way.

## Global Constraints

- Default `reranker_doc_header = False`: no change to production behaviour until an operator sets it.
- The embedding text built by `ingest/pipeline.py` must stay byte-identical (a change would silently desynchronise new ingests from the 1602 stored vectors).
- The reranker never blocks: the existing fallback paths are untouched.
- Tests never touch the network or 172.20.0.22; no new dependencies.
- Imports at the top of the module.
- `CLAUDE.md` stays ≤150 lines.

## Review Focus

1. A chunk with an empty section or clause → header omits that part, exactly as ingest does (no `Section: ` with nothing after it).
2. A result dict missing `doc_title` (the reranker's contract only promises `text`) → no `KeyError`; reranking proceeds on whatever header parts exist.
3. Setting on → the returned results' `text` is still the bare chunk text (the agent's `[Source N]` and the span's document content must not gain a duplicated header).
4. Setting on with the reranker disabled → startup warns it is inert, instead of silently doing nothing.
5. An experiment run with the header on is distinguishable from one without, by metadata and by auto-name.

---

### Task 1: One shared header builder; ingest uses it unchanged

**Files:**
- Create: `rag/chunk_header.py`
- Modify: `ingest/pipeline.py:35-46`
- Test: `tests/unit/test_chunk_header.py`

**Interfaces:**
- Produces: `with_header(text: str, doc_title: str = "", section: str = "", clause: str = "") -> str`

- [ ] **Step 1: Write the failing test**

```python
"""The header the embedding was built from, in one place (design 2026-10-09).

Ingest and the reranker must build it identically: if they drift, the reranker
judges a different text than the one retrieval matched, and nothing errors.
"""

from rag.chunk_header import with_header


def test_full_header():
    assert with_header("Body.", "Backup Policy [Internal]", "Backups", "Frequency") == (
        "Document: Backup Policy [Internal] | Section: Backups | Clause: Frequency\nBody."
    )


def test_empty_parts_are_omitted_not_left_blank():
    assert with_header("Body.", "Doc", "", "Cl") == "Document: Doc | Clause: Cl\nBody."
    assert with_header("Body.", "Doc") == "Document: Doc\nBody."


def test_no_parts_is_the_bare_text():
    assert with_header("Body.") == "Body."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_chunk_header.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.chunk_header'`

- [ ] **Step 3: Write minimal implementation**

`rag/chunk_header.py`:

```python
"""The `Document | Section | Clause` header every chunk is embedded with.

Shared by ingest (the dense embedding text) and the reranker (RERANKER_DOC_HEADER),
so the two can never build it differently.
"""


def with_header(text: str, doc_title: str = "", section: str = "", clause: str = "") -> str:
    parts = []
    if doc_title:
        parts.append(f"Document: {doc_title}")
    if section:
        parts.append(f"Section: {section}")
    if clause:
        parts.append(f"Clause: {clause}")
    prefix = " | ".join(parts)
    return f"{prefix}\n{text}" if prefix else text
```

`ingest/pipeline.py` — replace the loop body (lines 35-46) with:

```python
    # Build embedding text: prepend metadata so doc/section/clause
    # participate in cosine similarity (not just the chunk body)
    embed_inputs = [with_header(c.text, c.doc_title, c.section, c.clause) for c in chunks]
```

and add `from rag.chunk_header import with_header` to the module's imports.

- [ ] **Step 4: Pin ingest byte-identity, then run**

Add to `tests/unit/test_chunk_header.py` (with `import ingest.pipeline as pl` at the top):

```python
def test_ingest_embeds_exactly_what_it_did_before(monkeypatch):
    """The 1602 stored vectors were built from this text; any drift silently
    desynchronises the next ingest from them. Expected strings are the old
    loop's output, written out by hand."""

    class C:
        def __init__(self, section, clause):
            self.doc_id, self.doc_title, self.text = "d", "Doc", "Body."
            self.section, self.clause = section, clause

    seen = {}
    monkeypatch.setattr(pl, "init_collection", lambda: None)
    monkeypatch.setattr(pl, "assert_sparse_vector", lambda *a, **kw: None)
    monkeypatch.setattr(pl, "parse_docx", lambda fp, link: [C("Sec", "Cl"), C("", "Cl")])
    monkeypatch.setattr(pl, "delete_document", lambda doc_id: None)
    monkeypatch.setattr(pl, "embed_texts", lambda xs: seen.setdefault("x", xs) and [[0.0]] * len(xs))
    monkeypatch.setattr(pl, "upsert_chunks", lambda chunks, vecs: None)

    pl.ingest_document(Path("x.docx"), "")

    assert seen["x"] == [
        "Document: Doc | Section: Sec | Clause: Cl\nBody.",
        "Document: Doc | Clause: Cl\nBody.",
    ]
```

(`from pathlib import Path` at the top as well. Run this test against the
UNCHANGED loop first — it must pass there, proving it pins today's output —
then again after the refactor.)

Run: `PYTHONPATH=. pytest tests/unit/test_chunk_header.py -q`
Expected: PASS (4)

- [ ] **Step 5: Commit**

```bash
git add rag/chunk_header.py ingest/pipeline.py tests/unit/test_chunk_header.py
git commit -m "refactor(ingest): one shared builder for the chunk header"
```

### Task 2: `RERANKER_DOC_HEADER` setting and the reranker reading it

**Files:**
- Modify: `config.py` (field beside `reranker_instruction`; new `_INERT_WHEN` row)
- Modify: `rag/reranker.py` (`_rerank_impl`, `rerank` span attributes)
- Modify: `.env.example` (commented key under the reranker block)
- Test: `tests/unit/test_reranker_doc_header.py`

**Interfaces:**
- Consumes: `with_header` (Task 1)
- Produces: `settings.reranker_doc_header: bool`; `rag.reranker._scored_texts(results: list[dict]) -> list[str]`; span attribute `reranker.doc_header: bool`

- [ ] **Step 1: Write the failing tests**

```python
"""RERANKER_DOC_HEADER: the reranker judges the text the embedding matched.

Read-only A/B 2026-10-09 (design doc): 57 -> 58/61 in top 6, recovered 1, lost 0.
Only what is SCORED changes; what is RETURNED stays bare text.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rag.reranker as rr
from config import Settings, inert_env_keys

R = {"doc_title": "Data Breach Response", "section": "Non-Compliance Implications",
     "clause": "Disciplinary Action", "text": "Failure to report may lead to dismissal."}


def test_off_by_default():
    assert Settings(_env_file=None).reranker_doc_header is False


def test_off_scores_the_bare_text(monkeypatch):
    monkeypatch.setattr(rr.settings, "reranker_doc_header", False)
    assert rr._scored_texts([R]) == [R["text"]]


def test_on_scores_the_embedding_header(monkeypatch):
    monkeypatch.setattr(rr.settings, "reranker_doc_header", True)
    assert rr._scored_texts([R]) == [
        "Document: Data Breach Response | Section: Non-Compliance Implications | "
        "Clause: Disciplinary Action\nFailure to report may lead to dismissal."
    ]


def test_on_tolerates_a_result_with_only_text(monkeypatch):
    monkeypatch.setattr(rr.settings, "reranker_doc_header", True)
    assert rr._scored_texts([{"text": "t"}]) == ["t"]


@pytest.fixture
def scored(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(rr, "get_tracer", lambda: provider.get_tracer("t"))
    monkeypatch.setattr(rr.settings, "reranker_backend", "vllm-score")
    monkeypatch.setattr(rr.settings, "reranker_doc_header", True)
    sent = {}

    def _score(query, documents):
        sent["documents"] = documents
        return [(0, 0.9)]

    monkeypatch.setattr(rr, "_call_score", _score)
    return sent, exporter


def test_the_header_reaches_the_wire_but_not_the_results(scored):
    sent, _ = scored
    out = rr.rerank("q", [dict(R)], top_n=1)
    assert "Section: Non-Compliance Implications" in sent["documents"][0]
    assert sent["documents"][0].startswith("<Document>: ")
    assert out[0]["text"] == R["text"]


def test_the_rerank_span_records_the_setting(scored):
    _, exporter = scored
    rr.rerank("q", [dict(R)], top_n=1)
    (span,) = exporter.get_finished_spans()
    assert span.attributes["reranker.doc_header"] is True


def test_the_header_setting_is_inert_with_the_reranker_off():
    s = Settings(_env_file=None, reranker_enabled=False)
    (msg,) = inert_env_keys(s, env_text="RERANKER_DOC_HEADER=true\n", environ={})
    assert msg.startswith("RERANKER_DOC_HEADER is set but does nothing here")


def test_the_header_setting_is_live_with_the_reranker_on():
    s = Settings(_env_file=None, reranker_enabled=True)
    assert inert_env_keys(s, env_text="RERANKER_DOC_HEADER=true\n", environ={}) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_reranker_doc_header.py -q`
Expected: FAIL — `reranker_doc_header` is not a Settings field / `_scored_texts` not defined / no inert report.

- [ ] **Step 3: Implement**

`config.py`, after `reranker_instruction`:

```python
    # Score each candidate as "Document: … | Section: … | Clause: …\n<text>",
    # the text its dense vector was built from, instead of the bare text. The
    # agent and the span still get the bare text. Off until measured end to end:
    # the read-only A/B (design 2026-10-09) recovered 1 of 61 and lost none, but
    # also reordered 7 inside the top 6 by lifting sibling clauses of a
    # title-matched document.
    reranker_doc_header: bool = False
```

`config.py`, append to `_INERT_WHEN`:

```python
    (
        "RERANKER_DOC_HEADER",
        lambda s: not s.reranker_enabled,
        "it changes what the reranker scores, and RERANKER_ENABLED is off.",
    ),
```

`rag/reranker.py`: add `from rag.chunk_header import with_header` to the imports; add

```python
def _scored_texts(results: list[dict]) -> list[str]:
    """The text each candidate is judged on. Never what is returned: results
    keep their bare `text`, so the agent's sources and the span are unchanged."""
    if not settings.reranker_doc_header:
        return [r.get("text", "") for r in results]
    return [
        with_header(r.get("text", ""), r.get("doc_title", ""), r.get("section", ""), r.get("clause", ""))
        for r in results
    ]
```

and in `_rerank_impl` replace `documents = _build_documents([r.get("text", "") for r in results])` with `documents = _build_documents(_scored_texts(results))`. In `rerank`'s span attributes add `"reranker.doc_header": settings.reranker_doc_header,`.

`.env.example`, under `RERANKER_CANDIDATES`:

```bash
# Score candidates with their Document | Section | Clause header (default false).
# Measured read-only only; see docs/superpowers/specs/2026-10-09-query-rephrasing-design.md
# RERANKER_DOC_HEADER=true
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_reranker_doc_header.py tests/unit/test_reranker_wire_format.py tests/unit/test_retrieval_spans.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py rag/reranker.py .env.example tests/unit/test_reranker_doc_header.py
git commit -m "feat(reranker): RERANKER_DOC_HEADER scores candidates with their chunk header (off)"
```

### Task 3: Eval runs record the setting

**Files:**
- Modify: `eval/run_experiment.py` (tier1 + agent-tier metadata, auto-name, banner)
- Test: `tests/unit/test_eval_metadata.py` (extend, matching its existing style)

**Interfaces:**
- Consumes: `settings.reranker_doc_header` (Task 2)
- Produces: metadata key `reranker_doc_header` (bool, or `None` with the reranker off); auto-name suffix `_hdr` when on

- [ ] **Step 1: Write the failing test**

Read `tests/unit/test_eval_metadata.py` first and follow how it obtains the metadata dict for each tier. Add, for both tier1 and chatbot:

```python
@pytest.mark.parametrize("tier", ["tier1", "chatbot"])
def test_the_reranker_header_setting_is_recorded(monkeypatch, tier):
    # obtain metadata for `tier` with reranker_enabled=True, reranker_doc_header=True
    # the same way the neighbouring tests do, then:
    assert metadata["reranker_doc_header"] is True
```

and one asserting the auto-name ends with `_hdr` when the setting is on and no `--name` is given.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_eval_metadata.py -q`
Expected: FAIL with `KeyError: 'reranker_doc_header'`

- [ ] **Step 3: Implement**

In both metadata dicts, beside `reranker_min_score`:

```python
"reranker_doc_header": settings.reranker_doc_header if settings.reranker_enabled else None,
```

In the auto-name branch for the reranker (line ~445), append `+ ("_hdr" if settings.reranker_doc_header else "")`. In the banner's `Reranker:` line, append `", doc header" if settings.reranker_doc_header` inside the parenthesis.

- [ ] **Step 4: Run the whole suite**

Run: `PYTHONPATH=. pytest tests/ -q`
Expected: all pass (538 + new), docs/live auto-skip.

- [ ] **Step 5: Commit**

```bash
git add eval/run_experiment.py tests/unit/test_eval_metadata.py
git commit -m "feat(eval): record RERANKER_DOC_HEADER in experiment metadata and name"
```

### Task 4: Runbook and CLAUDE.md

**Files:**
- Modify: `SETUP.md` Step 11 (beside the rewrite comparison runbook)
- Modify: `CLAUDE.md` (Config block: one line)

- [ ] **Step 1: SETUP.md** — add a "Reranker header comparison" block mirroring the rewrite runbook: tier1 with `-e RERANKER_DOC_HEADER=false`, then `=true`, each with `-e GIT_COMMIT`; then `python eval/compare_runs.py <off_id> <on_id> --phoenix-url http://phoenix:6006`; then one chatbot run with the header on, compared against the existing `baseline-identity-v1` chatbot run, reading the answers of every question compare_runs lists as better/worse. Stop rule: ship on only if tier1 recovers ≥1 and loses 0, and no chatbot answer gets worse on reading.

- [ ] **Step 2: CLAUDE.md** — in the Config code block, after the `QUERY_REWRITE` line:

```
RERANKER_DOC_HEADER (false) — reranker scores "Document | Section | Clause\n<text>"; agent still sees bare text
```

Check `wc -l CLAUDE.md` ≤ 150.

- [ ] **Step 3: Commit**

```bash
git add SETUP.md CLAUDE.md
git commit -m "docs: RERANKER_DOC_HEADER runbook and config line"
```
