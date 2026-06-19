# Tier-A test suite + corpus characterization

**Date:** 2026-06-19
**Status:** Approved (design), pending implementation
**Author:** Dmytro Ivanov (with Claude Code)

## Problem

The project has no automated tests — `tests/` contains only an empty
`__init__.py`, there is no `pytest` dependency, no pytest config, and no CI.
The fragile, regression-prone units (Word numbering resolution, agent-response
JSON parsing, eval citation matching, clause-name parsing, Teams HTML
rendering) are exactly the parts that have broken before (see the CLAUDE.md
"Gotchas" table and the recent "fix clause name parsing" commit) and have no
guard against re-breaking.

## Goal

Stand up a first **Tier-A** test suite: fast, deterministic unit tests for the
high-value pure-logic units, plus a **corpus characterization layer** that runs
the real DOCX parser over the actual policy corpus to measure parsing coverage
and catch real-world regressions.

### Non-goals (YAGNI — explicitly out of scope)
- Tier-B tests (mocked reranker / temp-SQLite feedback / rating-routing) — a
  separate spec, discussed next.
- Tier-C integration tests (live Qdrant / Ollama / Teams Graph API).
- CI wiring — the suite runs locally with `pytest`; CI comes later.
- Tests for `rag/bm25_index.py` and the RRF fusion in `rag/hybrid_search.py`
  (BM25 is **off in production** — a regression there harms no one today).
- `ingest/chunk_models.py` (pydantic self-validates) and `config.py` properties
  (trivial conditionals).

## Key constraint: the corpus is confidential and not in git

`policies/*.docx` is gitignored (*"may contain sensitive docs"*). The 52-doc
corpus exists only on developer machines, never in the repo or CI. Consequences
that shape the design:

- Tests that need a real `.docx` must **auto-skip** when `policies/` has no
  `.docx`, so the portable unit layer (and any future CI) stays green without
  the corpus.
- **No committed `.docx` fixtures** (neither synthetic nor real) and **no
  committed content snapshots** — that would leak policy text into git.
- **No hardcoded policy filenames** in committed test code (titles are
  "[Internal]"). Doc-dependent tests select files **dynamically** (first N
  found / parametrize over all present) and assert **invariants** that hold for
  any well-formed doc, not exact values.

## Architecture / layout

```
requirements-dev.txt        # pytest, pytest-cov  (python-docx is already a runtime dep)
pytest.ini                  # testpaths=tests, markers, quiet addopts
tests/
  conftest.py               # shared fixtures (chunk builder, policy-doc discovery + auto-skip)
  unit/                     # fully portable — NO docs needed — runs everywhere/CI in <1s
    test_response_parser.py
    test_evaluators.py
    test_docx_parser_helpers.py
    test_renderer.py
    test_teams_utils.py
  docs/                     # needs real policies/ — AUTO-SKIPS when absent
    test_numbering.py
    test_corpus_parsing.py
scripts/parse_coverage.py   # compute_parse_stats(chunks) + CLI report (shared with corpus test)
```

Rationale: `pytest.ini` rather than `pyproject.toml` (the project uses
`requirements.txt`, has no pyproject — keep footprint minimal). A separate
`requirements-dev.txt` keeps production/Docker images lean.

## Component detail

### `tests/unit/` — portable pure-logic tests (no docs, no services)

- **`test_response_parser.py`** — `rag/response.py` `parse_agent_response()` /
  `_extract_json()`:
  - clean JSON object → parsed dict with `answer`, `citations`, `escalation`
  - JSON wrapped in ` ```json … ``` ` fences → parsed
  - JSON embedded in surrounding prose (brace-matching extracts it)
  - malformed JSON → fallback dict (escalation needed / parse failure semantics)
  - plain text with no JSON → fallback
- **`test_evaluators.py`** — `eval/evaluators.py` scoring functions on synthetic
  `(output, expected)` inputs:
  - citation match keys on section / **clause name**, not `section_display`
  - `match_mode="any"` with alternate `expected_citations` passes when any matches
  - case-insensitive / substring matching behaves as intended
  - `hit` and `MRR` over synthetic ranked lists (including miss → 0)
  - escalation evaluator (expected-escalation vs answered)
- **`test_docx_parser_helpers.py`** — pure string/regex helpers in
  `ingest/docx_parser.py` (no `Document` needed):
  `extract_clause_number`, `extract_clause_name`, `extract_heading_level`,
  `extract_label`, `_split_oversized` (splits at sentence boundaries under
  `max_tokens`), `_estimate_tokens`, `_table_to_text`.
- **`test_renderer.py`** — `channels/teams/renderer.py`:
  - rendered HTML uses only allowed tags (`<p> <b> <i> <ul>/<li> <hr>`); no
    `<div style>`
  - multi-citation answer does **not** duplicate the answer text
  - escalation and error rendering produce expected structure
  - content is HTML-escaped
- **`test_teams_utils.py`** — `channels/teams/utils.py`:
  `safe_get_nested` (key present / missing / default) and `strip_html` (tag
  removal, entity decoding, whitespace collapse).

### `tests/docs/` — corpus-dependent tests (auto-skip when no `.docx`)

- **`test_numbering.py`** — `ingest/numbering.py` `NumberingResolver` driven by
  a small **dynamically selected sample** of real docs. Invariant assertions
  (doc-agnostic): the level-0 counter is monotonic across a document, nesting is
  consistent when `ilvl` increases/decreases, resolved number strings are
  non-empty and match the expected `lvlText` shape. Auto-skips with no corpus.
- **`test_corpus_parsing.py`** — `ingest/docx_parser.py` `parse_docx()`
  parametrized over **all present** `policies/*.docx`. Per-doc invariants:
  parses without exception, yields ≥1 chunk, no chunk has empty quote/text,
  `doc_id`/title populated, clause numbers (where present) resolve in order.
  Uses `compute_parse_stats()` so the pass/fail logic and the report share one
  implementation. Auto-skips with no corpus.

### `scripts/parse_coverage.py` — shared stats + CLI report

- **`compute_parse_stats(chunks: list[PolicyChunk]) -> dict`** — pure function
  computing parsing-completeness metrics for one doc's chunks: chunk count,
  fraction of chunks with a resolved clause number, fraction with a
  section/clause name, count of empty-quote chunks, count of oversized/split
  chunks. Imported by both the corpus test (assertions) and the CLI (DRY).
- **CLI** (`PYTHONPATH=. python scripts/parse_coverage.py`) — parses every
  `policies/*.docx`, prints a per-doc table of the above metrics plus a summary
  line, and flags anomalies (0-chunk docs, empty quotes, oversized splits). Any
  saved baseline is a **gitignored local artifact** (e.g. under `data/` or a
  gitignored path) since filenames are sensitive.

### `tests/conftest.py` — shared fixtures
- a `make_chunk(**overrides)` builder returning a valid `PolicyChunk` with
  sensible defaults, for the pure tests
- a `policy_docx_paths` fixture that returns the list of present
  `policies/*.docx`, calling `pytest.skip(...)` when empty
- a `sample_policy_docx` fixture returning the first N paths for the targeted
  numbering tests

### Infra files
- **`requirements-dev.txt`** — `pytest`, `pytest-cov`. (`python-docx` is already
  in `requirements.txt`.)
- **`pytest.ini`** — `testpaths = tests`, register the markers used, quiet
  `addopts`. (Code-coverage % via `pytest-cov` is available but not gated.)

## Portability tradeoff (accepted)

`tests/unit/` runs anywhere, including CI, in under a second — this is the bulk
of the gotcha-guards. `tests/docs/` runs only where the corpus is loaded
(developer machines), and auto-skips elsewhere. This is the direct, accepted
consequence of keeping confidential policy docs out of git.

## Verification (how we know the suite works)

1. `pip install -r requirements-dev.txt` then `pytest` → unit tests pass; on a
   machine with `policies/` loaded, the `docs/` tests run and pass; on a machine
   without the corpus, the `docs/` tests report **skipped** (not failed).
2. Each new test is observed to **fail when the behavior is broken** (a quick
   red check), so they assert real behavior rather than tautologies.
3. `PYTHONPATH=. python scripts/parse_coverage.py` prints the per-doc coverage
   table over the local corpus with no errors.

## Risks

- **Numbering invariants under-specify.** Doc-agnostic invariants are weaker
  than exact-value assertions; they may miss subtle numbering bugs. Mitigation:
  cover the documented gotcha (cross-`numId` level-0 continuation) explicitly as
  an invariant; revisit with a crafted case later if needed.
- **Corpus drift.** Parametrizing over all present docs means the test set
  varies by machine. Acceptable — the invariants hold regardless of which docs
  are present; the CLI report is the place to eyeball absolute numbers.
```
