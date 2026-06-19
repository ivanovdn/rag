# Tier-A Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a fast, deterministic Tier-A pytest suite for the high-value pure-logic units, plus an auto-skipping corpus-characterization layer that runs the real DOCX parser over the local policy corpus.

**Architecture:** `tests/unit/` holds portable tests that need no files or services (response parsing, evaluator matching, renderer, Teams utils, docx-parser string helpers, parse-stats). `tests/docs/` holds tests that require the gitignored `policies/*.docx` corpus and auto-skip when it's absent (full `parse_docx` + `NumberingResolver`). A shared `scripts/parse_coverage.py` provides `compute_parse_stats()` (used by both a unit test and the corpus test) and a CLI coverage report.

**Tech Stack:** Python 3.12, pytest 9 (already in `.venv`), python-docx 1.2.0, pydantic. No new runtime deps; `pytest`/`pytest-cov` go in a dev-only requirements file.

## Global Constraints

- **These tests target EXISTING, working code.** The classic TDD red→green inverts: each test should **PASS on first run**. The discipline that replaces "watch it fail first" is a **non-vacuity check** — after the file is green, temporarily change ONE expected literal to a wrong value, run, confirm that test FAILS, then restore it. Expected values are **independent literals the author computes by hand — never derived by calling the function under test**.
- **No new runtime dependencies.** `pytest`, `pytest-cov` live in `requirements-dev.txt` only.
- **Corpus is confidential and gitignored** (`policies/*.docx`): commit **no** `.docx` fixtures and **no** policy content. Committed test code must **not hardcode policy filenames** — discover docs dynamically. Corpus tests **auto-skip** when no `.docx` is present.
- **Imports at the top of every module** (project rule). None of the tested modules pull in LlamaIndex, so the `init_observability()`-first constraint does not apply here.
- Run tests from the repo root with `PYTHONPATH=.` (e.g. `PYTHONPATH=. pytest -q`).
- Commit convention: lowercase type prefix (`test:`, `feat:`, `docs:`). End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Do not commit until the user confirms** at each task boundary. Work on a feature branch, not `main` (Task 0).

---

### Task 0: Create feature branch

**Files:** none (git only)

- [ ] **Step 1: Branch off main**
```bash
git checkout -b chore/test-suite-tier-a
```
- [ ] **Step 2: Confirm clean start**
Run: `git status --short`
Expected: no output.

---

### Task 1: Test infrastructure scaffold

**Files:**
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/_corpus.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py` (empty)
- Create: `tests/docs/__init__.py` (empty)
- Create: `tests/unit/test_smoke.py`

**Interfaces:**
- Produces: `tests/_corpus.py::policy_docx_files() -> list[Path]` (consumed by Task 7); `conftest.py::make_chunk` fixture returning a `PolicyChunk` (consumed by Tasks 6); the `corpus` pytest marker (consumed by Task 7).

- [ ] **Step 1: Create `requirements-dev.txt`**
```
# Test-only dependencies (not installed in production images)
pytest>=8
pytest-cov>=5
```

- [ ] **Step 2: Create `pytest.ini`**
```ini
[pytest]
testpaths = tests
addopts = -q
markers =
    corpus: requires the local policy .docx corpus (auto-skipped when absent)
```

- [ ] **Step 3: Create `tests/_corpus.py`**
```python
"""Locate the local policy corpus. The .docx files are gitignored, so this
returns an empty list on machines/CI without them and callers skip."""
from pathlib import Path

from config import settings


def policy_docx_files() -> list[Path]:
    folder = Path(settings.policy_docs_folder)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.docx"))
```

- [ ] **Step 4: Create `tests/conftest.py`**
```python
import pytest

from ingest.chunk_models import PolicyChunk


@pytest.fixture
def make_chunk():
    """Factory for a valid PolicyChunk with sensible defaults; override any field."""
    def _make(**overrides) -> PolicyChunk:
        defaults = dict(
            chunk_id="c1",
            doc_id="doc",
            doc_title="Doc Title",
            doc_filename="doc.docx",
            doc_link="http://example/doc",
            text="some text",
        )
        defaults.update(overrides)
        return PolicyChunk(**defaults)
    return _make
```

- [ ] **Step 5: Create empty `tests/unit/__init__.py` and `tests/docs/__init__.py`**
```bash
: > tests/unit/__init__.py
: > tests/docs/__init__.py
```

- [ ] **Step 6: Create `tests/unit/test_smoke.py`**
```python
"""Sanity: the suite collects and every unit under test imports cleanly."""
import importlib

import pytest


@pytest.mark.parametrize("module", [
    "rag.response",
    "eval.evaluators",
    "ingest.docx_parser",
    "ingest.numbering",
    "channels.teams.renderer",
    "channels.teams.utils",
])
def test_module_imports(module):
    assert importlib.import_module(module) is not None
```

- [ ] **Step 7: Install dev deps and run the suite**
Run: `pip install -r requirements-dev.txt && PYTHONPATH=. pytest -q`
Expected: all smoke tests PASS (e.g. `6 passed`), no errors, no collection warnings.

- [ ] **Step 8: Commit**
```bash
git add requirements-dev.txt pytest.ini tests/_corpus.py tests/conftest.py tests/unit/__init__.py tests/docs/__init__.py tests/unit/test_smoke.py
git commit -m "test: scaffold pytest infra (config, fixtures, corpus discovery, smoke test)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `rag/response.py` parser tests

**Files:**
- Create: `tests/unit/test_response_parser.py`

**Interfaces:**
- Consumes: `rag.response.parse_agent_response(raw_response: str) -> dict` (keys: `answer`, `citations`, `escalation`, `parse_success`, `raw_response`) and `rag.response._extract_json(text: str) -> str | None`.

- [ ] **Step 1: Write `tests/unit/test_response_parser.py`**
```python
from rag.response import parse_agent_response, _extract_json


def test_parses_clean_json_object():
    raw = '{"answer": "Yes", "citations": [{"doc_title": "AUP"}], "escalation": {"needed": false, "reason": ""}}'
    result = parse_agent_response(raw)
    assert result["parse_success"] is True
    assert result["answer"] == "Yes"
    assert result["citations"] == [{"doc_title": "AUP"}]
    assert result["escalation"] == {"needed": False, "reason": ""}


def test_parses_json_in_code_fence():
    raw = '```json\n{"answer": "Fenced", "citations": []}\n```'
    result = parse_agent_response(raw)
    assert result["parse_success"] is True
    assert result["answer"] == "Fenced"


def test_extracts_json_embedded_in_prose():
    raw = 'Here is the result: {"answer": "Embedded", "citations": []} hope that helps'
    result = parse_agent_response(raw)
    assert result["parse_success"] is True
    assert result["answer"] == "Embedded"


def test_malformed_json_falls_back_to_raw():
    raw = '{"answer": "broken"'  # missing closing brace -> invalid JSON
    result = parse_agent_response(raw)
    assert result["parse_success"] is False
    assert result["answer"] == raw
    assert result["citations"] == []
    assert result["escalation"] == {"needed": False, "reason": ""}


def test_plain_text_falls_back():
    raw = "I could not find a relevant policy."
    result = parse_agent_response(raw)
    assert result["parse_success"] is False
    assert result["answer"] == raw


def test_extract_json_none_when_no_brace():
    assert _extract_json("no json here at all") is None
```

- [ ] **Step 2: Run green**
Run: `PYTHONPATH=. pytest tests/unit/test_response_parser.py -q`
Expected: `6 passed`.

- [ ] **Step 3: Non-vacuity check**
Temporarily change `assert result["answer"] == "Yes"` to `== "WRONG"`, run the file, confirm that one test FAILS, then restore it. Re-run → all pass.

- [ ] **Step 4: Commit**
```bash
git add tests/unit/test_response_parser.py
git commit -m "test: cover parse_agent_response (fences, embedded JSON, fallback)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Teams `utils` + `renderer` tests

**Files:**
- Create: `tests/unit/test_teams_utils.py`
- Create: `tests/unit/test_renderer.py`

**Interfaces:**
- Consumes: `channels.teams.utils.safe_get_nested(data, *keys, default="")`, `channels.teams.utils.strip_html(text) -> str`; `channels.teams.renderer.render_answer(result: dict) -> str`, `render_escalation(question: str, result: dict) -> str`, `render_error(question: str, error: str) -> str`. `render_answer` reads `result["citations"]` (list of dicts with `doc_title`, `section`, `clause`, `clause_number`, `quote`) and `result["answer"]`.

- [ ] **Step 1: Write `tests/unit/test_teams_utils.py`**
```python
from channels.teams.utils import safe_get_nested, strip_html


def test_safe_get_nested_returns_value():
    data = {"a": {"b": {"c": 42}}}
    assert safe_get_nested(data, "a", "b", "c") == 42


def test_safe_get_nested_missing_key_returns_default():
    data = {"a": {"b": {}}}
    assert safe_get_nested(data, "a", "b", "c") == ""
    assert safe_get_nested(data, "a", "x", "c", default="NA") == "NA"


def test_safe_get_nested_non_dict_intermediate_returns_default():
    data = {"a": "not-a-dict"}
    assert safe_get_nested(data, "a", "b", default=None) is None


def test_strip_html_removes_tags_and_collapses_whitespace():
    assert strip_html("<p>Hello   <b>world</b></p>") == "Hello world"


def test_strip_html_decodes_entities():
    assert strip_html("Tom &amp; Jerry &lt;3") == "Tom & Jerry <3"


def test_strip_html_empty_input():
    assert strip_html("") == ""
```

- [ ] **Step 2: Write `tests/unit/test_renderer.py`**
```python
from channels.teams.renderer import render_answer, render_escalation, render_error


def test_render_answer_no_citations_uses_prose():
    assert render_answer({"answer": "Just prose.", "citations": []}) == "<p>Just prose.</p>"


def test_render_answer_single_citation_structure():
    result = {"answer": "A", "citations": [
        {"doc_title": "AUP", "section": "Use", "clause": "Email", "clause_number": "4.7", "quote": "No spam."},
    ]}
    html = render_answer(result)
    assert "📄 AUP" in html
    assert "<b>Section:</b> Use" in html
    assert "<b>Clause 4.7:</b> Email" in html
    assert '<i>"No spam."</i>' in html
    assert "<div" not in html  # only Teams-safe tags


def test_render_answer_multi_citation_does_not_repeat_answer_text():
    result = {"answer": "DUPLICATE_ME", "citations": [
        {"doc_title": "A", "quote": "q1"},
        {"doc_title": "B", "quote": "q2"},
    ]}
    html = render_answer(result)
    assert "This is addressed in 2 policies" in html
    assert html.count("<hr>") == 1
    assert "DUPLICATE_ME" not in html  # the prose answer is NOT duplicated when citations exist


def test_render_answer_clause_equal_to_number_shows_number_only():
    result = {"answer": "", "citations": [
        {"doc_title": "A", "clause": "4.7", "clause_number": "4.7", "quote": "q"},
    ]}
    html = render_answer(result)
    assert "<b>Clause 4.7</b>" in html
    assert "Clause 4.7:" not in html  # not rendered as the name form


def test_render_escalation_includes_question_and_reason():
    html = render_escalation("Can I install software?", {"escalation": {"reason": "No policy found."}})
    assert "Escalated to Compliance Team" in html
    assert "Can I install software?" in html
    assert "No policy found." in html


def test_render_error_includes_error_text():
    html = render_error("Q?", "boom")
    assert "Compliance lookup failed" in html
    assert "boom" in html
```

- [ ] **Step 3: Run green**
Run: `PYTHONPATH=. pytest tests/unit/test_teams_utils.py tests/unit/test_renderer.py -q`
Expected: `12 passed`.

- [ ] **Step 4: Non-vacuity check**
Temporarily change `assert html.count("<hr>") == 1` to `== 9`, run, confirm FAIL, restore. Re-run → all pass.

- [ ] **Step 5: Commit**
```bash
git add tests/unit/test_teams_utils.py tests/unit/test_renderer.py
git commit -m "test: cover Teams utils (safe_get_nested, strip_html) and renderer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `eval/evaluators.py` matching tests

**Files:**
- Create: `tests/unit/test_evaluators.py`

**Interfaces:**
- Consumes: from `eval.evaluators`: `hit_evaluator(output, expected) -> dict`, `mrr_evaluator`, `citation_clause_accuracy`, `json_parse_success`. Each returns `{"score": float, "label": str, "explanation": str}`. `output` carries `search_results` / `citations` (lists of dicts with `doc_title`, `section`, `clause`). `expected` uses either Tier-1 keys (`expected_doc`/`expected_section`/`expected_clause`) or `expected_citations: [{doc_id, section, clause}]`, plus optional top-level `match_mode` (`"all"` default / `"any"`). Matching: doc is exact (case-insensitive); section/clause are substring (case-insensitive).

- [ ] **Step 1: Write `tests/unit/test_evaluators.py`**
```python
from eval.evaluators import (
    hit_evaluator,
    mrr_evaluator,
    citation_clause_accuracy,
    json_parse_success,
)


def _output(search_results=None, citations=None, **extra):
    out = {"search_results": search_results or [], "citations": citations or []}
    out.update(extra)
    return out


def test_hit_all_mode_pass():
    out = _output(search_results=[{"doc_title": "AUP", "section": "Acceptable Use", "clause": "Email"}])
    exp = {"expected_doc": "AUP", "expected_section": "Acceptable Use", "expected_clause": "Email"}
    assert hit_evaluator(out, exp)["score"] == 1.0


def test_hit_matches_section_as_substring_not_section_display():
    # expected "Private Information" matches inside the chunk's "7. Private Information"
    out = _output(search_results=[{"doc_title": "AUP", "section": "7. Private Information", "clause": "Blogging"}])
    exp = {"expected_doc": "AUP", "expected_section": "Private Information", "expected_clause": "Blogging"}
    assert hit_evaluator(out, exp)["score"] == 1.0


def test_hit_all_mode_miss():
    out = _output(search_results=[{"doc_title": "Other", "section": "X", "clause": "Y"}])
    exp = {"expected_doc": "AUP", "expected_section": "Acceptable Use", "expected_clause": "Email"}
    assert hit_evaluator(out, exp)["score"] == 0.0


def test_hit_any_mode_passes_when_one_alt_matches():
    out = _output(search_results=[{"doc_title": "AUP", "section": "Acceptable Use", "clause": "Email"}])
    exp = {"match_mode": "any", "expected_citations": [
        {"doc_id": "Wrong Doc", "section": "Z", "clause": "Z"},
        {"doc_id": "AUP", "section": "Acceptable Use", "clause": "Email"},
    ]}
    assert hit_evaluator(out, exp)["score"] == 1.0


def test_mrr_first_match_at_rank_two():
    out = _output(search_results=[
        {"doc_title": "X", "section": "", "clause": ""},
        {"doc_title": "AUP", "section": "Use", "clause": "Email"},
    ])
    exp = {"expected_doc": "AUP", "expected_section": "Use", "expected_clause": "Email"}
    assert mrr_evaluator(out, exp)["score"] == 0.5


def test_citation_clause_any_mode_substring():
    out = _output(citations=[{"doc_title": "AUP", "section": "Acceptable Use", "clause": "Email Use"}])
    exp = {"match_mode": "any", "expected_citations": [
        {"doc_id": "AUP", "section": "Acceptable Use", "clause": "Email"},
    ]}
    assert citation_clause_accuracy(out, exp)["score"] == 1.0


def test_citation_clause_skips_when_no_expected_clause():
    out = _output(citations=[])
    exp = {"expected_doc": "AUP", "expected_section": "Use"}
    result = citation_clause_accuracy(out, exp)
    assert result["score"] == 1.0
    assert result["label"] == "skip"


def test_json_parse_success_true_and_false():
    assert json_parse_success({"parse_success": True}, {})["score"] == 1.0
    assert json_parse_success({"parse_success": False, "raw_response": "x"}, {})["score"] == 0.0


def test_evaluator_handles_none_output():
    result = hit_evaluator(None, {})
    assert result["score"] == 0.0
    assert result["label"] == "error"
```

- [ ] **Step 2: Run green**
Run: `PYTHONPATH=. pytest tests/unit/test_evaluators.py -q`
Expected: `9 passed`.

- [ ] **Step 3: Non-vacuity check**
Temporarily change `test_mrr_first_match_at_rank_two`'s expected `0.5` to `0.99`, run, confirm FAIL, restore. Re-run → all pass.

- [ ] **Step 4: Commit**
```bash
git add tests/unit/test_evaluators.py
git commit -m "test: cover evaluator matching (match_mode any/all, substring, MRR)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `ingest/docx_parser.py` pure-helper tests

**Files:**
- Create: `tests/unit/test_docx_parser_helpers.py`

**Interfaces:**
- Consumes: from `ingest.docx_parser`: `extract_clause_number(text) -> str | None`, `extract_label(text) -> str | None`, `extract_heading_level(paragraph) -> int | None` (reads `paragraph.style.name`), `extract_clause_name(para) -> str` (reads `para.runs[*].bold/.text`), `_estimate_tokens(text) -> int`, `_split_oversized(text, max_tokens) -> list[str]`, `_table_to_text(table) -> str` (reads `table.rows[*].cells[*].text`).
- Note: the docx-object helpers are exercised with lightweight `SimpleNamespace` fakes — no real `.docx`, no files.

- [ ] **Step 1: Write `tests/unit/test_docx_parser_helpers.py`**
```python
from types import SimpleNamespace

from ingest.docx_parser import (
    extract_clause_number,
    extract_label,
    extract_heading_level,
    extract_clause_name,
    _estimate_tokens,
    _split_oversized,
    _table_to_text,
)


# --- pure string helpers ---

def test_extract_clause_number_variants():
    assert extract_clause_number("4.7 Software Installation") == "4.7"
    assert extract_clause_number("3. Introduction") == "3"
    assert extract_clause_number("4.2.1 Nested item") == "4.2.1"


def test_extract_clause_number_none_without_trailing_space():
    assert extract_clause_number("Introduction") is None
    assert extract_clause_number("4.7No space after number") is None


def test_extract_label():
    assert extract_label("Acceptable Use: limited and occasional") == "Acceptable Use"
    assert extract_label("lowercase start: nope") is None


def test_estimate_tokens_four_chars_per_token():
    assert _estimate_tokens("a" * 40) == 10


def test_split_oversized_under_limit_returns_single():
    text = "Short sentence."
    assert _split_oversized(text, max_tokens=100) == [text]


def test_split_oversized_splits_on_sentence_boundaries():
    text = "This is a sentence with several words. " * 20
    parts = _split_oversized(text, max_tokens=20)
    assert len(parts) > 1
    assert all(p.strip() for p in parts)


# --- helpers that read docx-like objects (lightweight fakes) ---

def _fake_run(text, bold):
    return SimpleNamespace(text=text, bold=bold)


def _fake_para(runs=(), style_name="Normal", text=""):
    return SimpleNamespace(runs=list(runs), style=SimpleNamespace(name=style_name), text=text)


def test_extract_heading_level():
    assert extract_heading_level(_fake_para(style_name="Heading 2")) == 2
    assert extract_heading_level(_fake_para(style_name="Normal")) is None


def test_extract_clause_name_from_leading_bold_runs():
    para = _fake_para(runs=[
        _fake_run("Blogging and Social Media:", True),
        _fake_run(" the rest of the text", False),
    ])
    assert extract_clause_name(para) == "Blogging and Social Media"


def test_extract_clause_name_empty_when_first_run_not_bold():
    para = _fake_para(runs=[_fake_run("not bold", False)])
    assert extract_clause_name(para) == ""


def test_table_to_text_pairs_headers_with_cells():
    def cell(t):
        return SimpleNamespace(text=t)

    def row(*texts):
        return SimpleNamespace(cells=[cell(t) for t in texts])

    table = SimpleNamespace(rows=[row("H1", "H2"), row("a", "b")])
    assert _table_to_text(table) == "H1: a | H2: b"
```

- [ ] **Step 2: Run green**
Run: `PYTHONPATH=. pytest tests/unit/test_docx_parser_helpers.py -q`
Expected: `10 passed`.

- [ ] **Step 3: Non-vacuity check**
Temporarily change the expected `"4.7"` in `test_extract_clause_number_variants` to `"9.9"`, run, confirm FAIL, restore. Re-run → all pass.

- [ ] **Step 4: Commit**
```bash
git add tests/unit/test_docx_parser_helpers.py
git commit -m "test: cover docx_parser pure helpers (clause number/name, split, table)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `scripts/parse_coverage.py` + parse-stats unit test

**Files:**
- Create: `scripts/parse_coverage.py`
- Create: `tests/unit/test_parse_stats.py`

**Interfaces:**
- Produces: `scripts.parse_coverage.compute_parse_stats(chunks: list[PolicyChunk]) -> dict` with keys `chunk_count`, `with_clause_number`, `with_section_or_clause`, `empty_text`, `oversized`, `pct_clause_number`, `pct_section_or_clause`; `find_policy_docx(folder=None) -> list[Path]`; and a CLI `main()`. (Consumed by Task 7's corpus test.)
- Consumes: `ingest.docx_parser.parse_docx`, `ingest.docx_parser._estimate_tokens`, `config.settings`.

- [ ] **Step 1: Create `scripts/parse_coverage.py`**
```python
"""Parsing-coverage report over the local policy corpus.

Provides compute_parse_stats() (shared with the corpus test) and a CLI that
prints per-doc parsing metrics. The corpus is gitignored, so this only does
anything useful on a machine that has the .docx files locally.

Usage:
    PYTHONPATH=. python scripts/parse_coverage.py
"""
from pathlib import Path

from config import settings
from ingest.chunk_models import PolicyChunk
from ingest.docx_parser import parse_docx, _estimate_tokens


def find_policy_docx(folder=None) -> list[Path]:
    folder = Path(folder or settings.policy_docs_folder)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.docx"))


def compute_parse_stats(chunks: list[PolicyChunk]) -> dict:
    """Parsing-completeness metrics for one or more docs' chunks."""
    total = len(chunks)
    with_clause_number = sum(1 for c in chunks if c.clause_number)
    with_section_or_clause = sum(1 for c in chunks if c.section or c.clause)
    empty_text = sum(1 for c in chunks if not c.text.strip())
    oversized = sum(1 for c in chunks if _estimate_tokens(c.text) > settings.chunk_max_tokens)
    return {
        "chunk_count": total,
        "with_clause_number": with_clause_number,
        "with_section_or_clause": with_section_or_clause,
        "empty_text": empty_text,
        "oversized": oversized,
        "pct_clause_number": round(with_clause_number / total, 3) if total else 0.0,
        "pct_section_or_clause": round(with_section_or_clause / total, 3) if total else 0.0,
    }


def main() -> None:
    paths = find_policy_docx()
    if not paths:
        print(f"No .docx found in {settings.policy_docs_folder} (corpus is local-only).")
        return

    print(f"{'doc':48} {'chunks':>6} {'clause%':>8} {'sec/cl%':>8} {'empty':>6} {'oversz':>7}")
    print("-" * 86)
    all_chunks: list[PolicyChunk] = []
    for path in paths:
        chunks = parse_docx(path, doc_link="")
        all_chunks.extend(chunks)
        s = compute_parse_stats(chunks)
        print(f"{path.stem[:48]:48} {s['chunk_count']:>6} {s['pct_clause_number']:>8} "
              f"{s['pct_section_or_clause']:>8} {s['empty_text']:>6} {s['oversized']:>7}")

    g = compute_parse_stats(all_chunks)
    print("-" * 86)
    print(f"TOTAL: {len(paths)} docs, {g['chunk_count']} chunks | "
          f"clause#={g['pct_clause_number']} sec/clause={g['pct_section_or_clause']} "
          f"empty={g['empty_text']} oversized={g['oversized']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tests/unit/test_parse_stats.py`** (uses the `make_chunk` fixture from `conftest.py`)
```python
from scripts.parse_coverage import compute_parse_stats


def test_compute_parse_stats_counts(make_chunk):
    chunks = [
        make_chunk(text="alpha", clause_number="1.1", section="S"),
        make_chunk(text="beta", clause_number="", section="", clause=""),
        make_chunk(text="   ", clause_number="2.0", clause="C"),  # blank text
    ]
    s = compute_parse_stats(chunks)
    assert s["chunk_count"] == 3
    assert s["with_clause_number"] == 2
    assert s["with_section_or_clause"] == 2
    assert s["empty_text"] == 1
    assert s["pct_clause_number"] == round(2 / 3, 3)


def test_compute_parse_stats_empty_list():
    s = compute_parse_stats([])
    assert s["chunk_count"] == 0
    assert s["pct_clause_number"] == 0.0
```

- [ ] **Step 3: Run green**
Run: `PYTHONPATH=. pytest tests/unit/test_parse_stats.py -q`
Expected: `2 passed`.

- [ ] **Step 4: Non-vacuity check**
Temporarily change `assert s["empty_text"] == 1` to `== 0`, run, confirm FAIL, restore. Re-run → pass.

- [ ] **Step 5: Smoke-run the CLI (corpus-dependent — note if skipped)**
Run: `PYTHONPATH=. python scripts/parse_coverage.py`
Expected (with corpus present): a per-doc table + a `TOTAL:` line, no traceback. Without the corpus: the "No .docx found" message. Either is acceptable — record which occurred.

- [ ] **Step 6: Commit**
```bash
git add scripts/parse_coverage.py tests/unit/test_parse_stats.py
git commit -m "feat: add parse-coverage report (compute_parse_stats + CLI) with unit test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Corpus-dependent tests (auto-skipping)

**Files:**
- Create: `tests/docs/test_corpus_parsing.py`
- Create: `tests/docs/test_numbering.py`

**Interfaces:**
- Consumes: `tests._corpus.policy_docx_files()`, `ingest.docx_parser.parse_docx`, `scripts.parse_coverage.compute_parse_stats`, `ingest.numbering.NumberingResolver`, `docx.Document`.
- Both modules: marked `corpus` and `skipif(not <docs present>)` so they auto-skip when the corpus is absent. Doc identities come from runtime discovery — no filenames are hardcoded.

- [ ] **Step 1: Write `tests/docs/test_corpus_parsing.py`**
```python
import pytest

from ingest.docx_parser import parse_docx
from scripts.parse_coverage import compute_parse_stats
from tests._corpus import policy_docx_files

_DOCX = policy_docx_files()
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not _DOCX, reason="policy corpus not present (gitignored/local-only)"),
]


@pytest.mark.parametrize("docx_path", _DOCX, ids=[p.stem for p in _DOCX])
def test_doc_parses_to_valid_chunks(docx_path):
    chunks = parse_docx(docx_path, doc_link="http://test/policy")
    assert len(chunks) >= 1, f"{docx_path.name} produced no chunks"
    for c in chunks:
        assert c.text.strip(), f"empty chunk text in {docx_path.name}"
        assert c.doc_id, f"missing doc_id in {docx_path.name}"
        assert c.doc_title, f"missing doc_title in {docx_path.name}"


def test_corpus_stats_are_sane():
    all_chunks = []
    for path in _DOCX:
        all_chunks.extend(parse_docx(path, doc_link=""))
    stats = compute_parse_stats(all_chunks)
    assert stats["chunk_count"] >= len(_DOCX)  # >= 1 chunk per doc
    assert stats["empty_text"] == 0
```

- [ ] **Step 2: Write `tests/docs/test_numbering.py`**
```python
import pytest
from docx import Document

from ingest.numbering import NumberingResolver
from tests._corpus import policy_docx_files

_DOCX = policy_docx_files()
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not _DOCX, reason="policy corpus not present (gitignored/local-only)"),
]


def _level0_decimals(path):
    """Walk a doc, assert per-paragraph numbering invariants, and return the
    sequence of leading integers for level-0 decimal paragraphs."""
    doc = Document(path)
    resolver = NumberingResolver(doc)
    seq = []
    for element in doc.element.body:
        info = resolver.resolve(element)
        if info is None:
            continue
        # structural invariants for every numbered paragraph
        assert info["ilvl"] >= 0
        assert isinstance(info["numFmt"], str) and info["numFmt"]
        if info["numFmt"] == "decimal":
            # decimal lvlText is reliably non-empty (e.g. "%1."); bullets may not be
            assert info["resolved"] != "", f"empty resolved decimal number in {path.name}"
            if info["ilvl"] == 0:
                leading = info["resolved"].strip().rstrip(".").split(".")[0]
                if leading.isdigit():
                    seq.append(int(leading))
    return seq


@pytest.mark.parametrize("docx_path", _DOCX, ids=[p.stem for p in _DOCX])
def test_numbering_resolves_structurally(docx_path):
    _level0_decimals(docx_path)  # assertions happen inside


def test_level0_decimal_numbering_is_continuous():
    """The resolver continues top-level decimal numbering across numIds, so the
    level-0 sequence should be non-decreasing and actually progress — never
    reset to 1 per numId (the documented gotcha)."""
    checked = 0
    for path in _DOCX:
        seq = _level0_decimals(path)
        if len(seq) >= 2:
            checked += 1
            assert seq == sorted(seq), f"level-0 numbering not non-decreasing in {path.name}: {seq}"
            assert seq[-1] > seq[0], f"level-0 numbering did not progress in {path.name}: {seq}"
    if checked == 0:
        pytest.skip("no doc with >=2 level-0 decimal sections")
```

- [ ] **Step 3: Run the corpus layer**
Run: `PYTHONPATH=. pytest tests/docs -q`
Expected: on a machine **with** the corpus, the doc tests run and PASS (one parametrized case per doc); on a machine **without** it, every case reports **skipped** (never failed). Record which occurred.

- [ ] **Step 4: Confirm auto-skip works regardless of corpus presence**
Run: `PYTHONPATH=. pytest -m "not corpus" -q`
Expected: runs only the `tests/unit/` suite (no corpus cases collected), all PASS.

- [ ] **Step 5: Commit**
```bash
git add tests/docs/test_corpus_parsing.py tests/docs/test_numbering.py
git commit -m "test: add auto-skipping corpus parsing + numbering invariants

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Docs + full-suite verification

**Files:**
- Modify: `SETUP.md` (insert a `## Testing` section before the `## Project Status` section)
- Modify: `CLAUDE.md` (the "Not yet implemented" line)

**Interfaces:** none.

- [ ] **Step 1: Add a `## Testing` section to `SETUP.md`**
Locate the line `## Project Status` and insert the following block immediately before it:
```markdown
## Testing

Tier-A unit tests (pure logic, no services) plus an auto-skipping corpus layer.

```bash
pip install -r requirements-dev.txt

# Unit tests only — fast, no policy docs needed (CI-safe)
PYTHONPATH=. pytest -m "not corpus" -q

# Everything, including corpus parsing/numbering (needs local policies/*.docx)
PYTHONPATH=. pytest -q

# Parsing-coverage report over the local corpus
PYTHONPATH=. python scripts/parse_coverage.py
```

The `corpus` tests require the gitignored `policies/*.docx` and **auto-skip**
when they're absent (e.g. on CI or a fresh checkout).

---
```

- [ ] **Step 2: Update the "Not yet implemented" line in `CLAUDE.md`**
Change:
```markdown
**Not yet implemented:** email escalation, pytest suite.
```
to:
```markdown
**Not yet implemented:** email escalation. (Tier-A pytest suite exists under `tests/`; Tier-B/C and CI still pending.)
```

- [ ] **Step 3: Full-suite run**
Run: `PYTHONPATH=. pytest -q`
Expected: all unit tests pass; corpus tests pass (corpus present) or skip (absent) — no failures.

- [ ] **Step 4: Commit**
```bash
git add SETUP.md CLAUDE.md
git commit -m "docs: document the test suite and mark pytest suite implemented

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the executor

- If any test won't go green against the real code, **stop and surface it** — it may be a genuine bug in the code under test (that's a finding worth keeping), not a test to force-pass.
- The corpus tests' pass/skip outcome depends on whether `policies/*.docx` exists on the machine; always report which path executed.
- Honor the "commit only at task boundaries, with the user's nod" rule.
- After all tasks: branch `chore/test-suite-tier-a` is ready for the finishing-a-development-branch skill.

## Risks

- **Numbering continuity invariant** (`test_level0_decimal_numbering_is_continuous`) assumes each policy is a single top-level decimal hierarchy. If a real doc legitimately contains two independent top-level numbered lists, the resolver is *designed* to continue them — so a failure here is most likely a real resolver regression, but could require excluding an unusual doc. Surface it rather than weakening the assertion blindly.
- **Corpus drift:** the parametrized doc set varies by machine; invariants are written to hold for any well-formed doc, and absolute numbers live in the CLI report, not assertions.
```
