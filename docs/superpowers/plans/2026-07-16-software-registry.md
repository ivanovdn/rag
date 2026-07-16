# Software Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the compliance bot answer "is this software allowed/forbidden?" and "what can I use for X?" from two curated lists, via a new `check_software` agent tool backed by a separate Qdrant collection.

**Architecture:** A committed JSON registry (`software/software_registry.json`) is ingested into a dedicated `software_registry` Qdrant collection. A hybrid `check_software` tool does fuzzy name lookup (in-memory index built by scrolling the collection) first, then a score-gated semantic fallback for category questions. The tool is a third tool on the existing `AgentWorkflow` (router unchanged). Software answers reuse the `ComplianceAnswer` schema with two new optional citation fields and a status-aware Teams renderer branch. "Not listed" is a distinct deterministic outcome (canned "ask IT" reply, no rating).

**Tech Stack:** Python, pydantic-settings, LlamaIndex `AgentWorkflow`, Qdrant, Ollama embeddings, `rapidfuzz` (fuzzy), `pdfplumber` (bootstrap only), pytest.

## Global Constraints

- `temperature=0.0` always — do not change the LLM temperature.
- The agent must never state a status that did not come from a retrieved registry row — no invented allowed/forbidden verdicts. "Not forbidden" ≠ "allowed".
- `init_observability()` must run FIRST in every entry point, before any LlamaIndex/Ollama import (see `scripts/ingest_all.py` for the pattern).
- **Imports always at the top of the module** — except the documented pytest-collection exception in `tests/live/` (import `rag.*` inside the test function).
- Teams HTML is limited to `<p> <b> <i> <ul>/<li> <hr> <code> <br>` — no `<div style>`; never leak raw errors into HTML.
- Software uses the **same embedding model and dim** as policies — do NOT change `QDRANT_VECTOR_DIM` or `EMBEDDING_MODEL`.
- `rapidfuzz` scores are on a **0–100** scale.
- The **router is unchanged** — do not add a `software` category.
- Match surrounding code conventions; the deployed `.env` is the source of truth, `config.py` defaults are lighter dev fallbacks.

---

## File Structure

**Create:**
- `software/software_registry.json` — canonical registry data (committed).
- `software/Allowed Software.pdf`, `software/Forbidden Software.pdf` — moved from `data/` (raw artifacts; gitignored).
- `scripts/build_software_registry.py` — one-time pdfplumber PDF→JSON bootstrap.
- `scripts/ingest_software.py` — JSON→Qdrant ingest CLI.
- `rag/software_registry.py` — pure logic: `SoftwareRow`, normalize, load, name index, fuzzy lookup, embed-text builder.
- `rag/tools/check_software.py` — the hybrid tool + module flags + result formatting.
- `tests/unit/test_software_config.py`
- `tests/unit/test_software_registry.py`
- `tests/unit/test_check_software.py`
- `tests/unit/test_renderer_software.py`
- `tests/docs/test_software_registry_data.py`
- `tests/live/test_software_lookup.py`
- `tests/_software.py` — probe: is the software collection ingested?

**Modify:**
- `config.py` — software settings.
- `.env.example` — software env keys.
- `requirements.txt` — `rapidfuzz`, `pdfplumber`.
- `rag/vector_store.py` — `collection_name` param on `search_vectors`/`scroll_by_filter`; add `init_software_collection`, `upsert_software_rows`, `scroll_all`.
- `rag/agent.py` — `Citation` optional fields; `SYSTEM_PROMPT` tool-selection + software rules; `ALL_TOOLS` gated by `software_lookup_enabled`.
- `channels/teams/renderer.py` — software citation branch; `render_software_not_listed`.
- `channels/teams/bot.py` — `_run_rag` flags + `software_not_found` outcome; `_send_reply` handling.
- `.gitignore` — `software/*.pdf`.
- `CLAUDE.md` — architecture, commands, config, gotchas, future-source note.

---

## Task 1: Dependencies & config

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py:36-45` (Qdrant block) and end of `Settings`
- Modify: `.env.example`
- Test: `tests/unit/test_software_config.py`

**Interfaces:**
- Produces: `settings.software_lookup_enabled: bool`, `settings.software_collection: str`, `settings.software_docs_folder: str`, `settings.software_registry_path: str`, `settings.software_fuzzy_threshold: float`, `settings.software_min_semantic_score: float`.

- [ ] **Step 1: Add dependencies**

In `requirements.txt`, under `# Document parsing & ingestion` add `pdfplumber>=0.11`, and under a new `# Fuzzy matching` line add `rapidfuzz>=3.9`:

```
# Document parsing & ingestion
python-docx>=1.2
python-slugify>=8.0
pdfplumber>=0.11

# Fuzzy matching (software name lookup)
rapidfuzz>=3.9
```

Then install:

Run: `pip install rapidfuzz pdfplumber`
Expected: successful install (no errors).

- [ ] **Step 2: Write the failing config test**

Create `tests/unit/test_software_config.py`:

```python
from config import Settings


def test_software_defaults():
    s = Settings()
    assert s.software_lookup_enabled is True
    assert s.software_collection == "software_registry"
    assert s.software_docs_folder == "./software"
    assert s.software_registry_path == "./software/software_registry.json"
    assert s.software_fuzzy_threshold == 85.0
    assert s.software_min_semantic_score == 0.5
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_software_config.py -v`
Expected: FAIL (AttributeError / missing fields).

- [ ] **Step 4: Add the settings**

In `config.py`, immediately after the Qdrant block (after line 45, the `active_qdrant_url` property), add:

```python
    # Software registry (allowed/forbidden software lookup)
    software_lookup_enabled: bool = True
    software_collection: str = "software_registry"
    software_docs_folder: str = "./software"
    software_registry_path: str = "./software/software_registry.json"
    software_fuzzy_threshold: float = 85.0     # rapidfuzz score 0-100; >= this is a confident name match
    software_min_semantic_score: float = 0.5   # cosine floor for the category (semantic) fallback
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/unit/test_software_config.py -v`
Expected: PASS.

- [ ] **Step 6: Add env keys**

In `.env.example`, after the `QDRANT_COLLECTION=compliance_policies` line, add:

```bash
# Software registry (allowed/forbidden software lookup)
SOFTWARE_LOOKUP_ENABLED=true
SOFTWARE_COLLECTION=software_registry
SOFTWARE_DOCS_FOLDER=./software
SOFTWARE_REGISTRY_PATH=./software/software_registry.json
SOFTWARE_FUZZY_THRESHOLD=85.0
SOFTWARE_MIN_SEMANTIC_SCORE=0.5
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt config.py .env.example tests/unit/test_software_config.py
git commit -m "feat(software): add config + deps for software registry"
```

---

## Task 2: Registry core logic (`rag/software_registry.py`)

Pure, dependency-light logic (no Qdrant, no LLM). This is the heart of the fuzzy lookup.

**Files:**
- Create: `rag/software_registry.py`
- Test: `tests/unit/test_software_registry.py`

**Interfaces:**
- Consumes: `rapidfuzz` (Task 1); `settings.software_registry_path`.
- Produces:
  - `class SoftwareRow(BaseModel)` with fields `name:str, status:str, category:str="", note:str="", alternative:str="", aliases:list[str]=[], source_list:str=""`.
  - `normalize_name(s: str) -> str`
  - `load_registry(path: str | None = None) -> list[SoftwareRow]`
  - `build_name_index(rows: list[SoftwareRow]) -> dict[str, SoftwareRow]`
  - `@dataclass LookupResult(row: SoftwareRow | None, score: float, exact: bool = False, suggestion: SoftwareRow | None = None)`
  - `lookup_name(query: str, index: dict[str, SoftwareRow], threshold: float) -> LookupResult`
  - `software_embed_text(row: SoftwareRow) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_software_registry.py`:

```python
import pytest

from rag.software_registry import (
    SoftwareRow,
    normalize_name,
    build_name_index,
    lookup_name,
    software_embed_text,
)


@pytest.fixture
def rows():
    return [
        SoftwareRow(name="Docker", status="allowed",
                    note="A license is required for commercial usage",
                    source_list="Allowed Software"),
        SoftwareRow(name="WinRar", status="forbidden",
                    category="High-risk russian software",
                    alternative="Windows 11 natively supports .tar, .tgz, .7z, .rar etc.",
                    source_list="Forbidden Software"),
        SoftwareRow(name="JetBrains Software", status="forbidden",
                    category="High-risk russian software",
                    alternative="MS Visual Studio, VS Code",
                    aliases=["IntelliJ IDEA", "PyCharm", "WebStorm", "Rider"],
                    source_list="Forbidden Software"),
    ]


def test_normalize_collapses_whitespace_and_case():
    assert normalize_name("  Docker   Desktop ") == "docker desktop"


def test_exact_name_match(rows):
    idx = build_name_index(rows)
    r = lookup_name("docker", idx, threshold=85.0)
    assert r.row is not None and r.row.name == "Docker" and r.exact is True


def test_alias_resolves_to_parent_row(rows):
    idx = build_name_index(rows)
    r = lookup_name("PyCharm", idx, threshold=85.0)
    assert r.row is not None and r.row.name == "JetBrains Software"


def test_fuzzy_typo_matches_above_threshold(rows):
    idx = build_name_index(rows)
    # genuine typo (extra 'r'), NOT just a case variant — a case variant would
    # normalize to an exact key hit; this must exercise the fuzzy path (exact=False).
    r = lookup_name("WinRarr", idx, threshold=85.0)
    assert r.row is not None and r.row.name == "WinRar" and r.exact is False


def test_unknown_returns_no_row_but_suggests_closest(rows):
    idx = build_name_index(rows)
    r = lookup_name("Dockerr Composer XYZ", idx, threshold=99.0)
    assert r.row is None
    assert r.suggestion is not None  # closest candidate offered as "did you mean"


def test_embed_text_forbidden_includes_category_and_alternative(rows):
    txt = software_embed_text(rows[1])
    assert "WinRar" in txt and "russian" in txt and "Alternative:" in txt


def test_embed_text_allowed_is_name_and_note(rows):
    txt = software_embed_text(rows[0])
    assert txt.startswith("Docker") and "license" in txt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_software_registry.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the module**

Create `rag/software_registry.py`:

```python
"""Pure logic for the software allowed/forbidden registry: row model, name
normalization, in-memory fuzzy name index, and embedding-text construction.

No Qdrant or LLM here — the check_software tool wires this to the backends.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from config import settings


class SoftwareRow(BaseModel):
    name: str
    status: str  # "allowed" | "forbidden"
    category: str = ""
    note: str = ""
    alternative: str = ""
    aliases: list[str] = Field(default_factory=list)
    source_list: str = ""


@dataclass
class LookupResult:
    row: SoftwareRow | None
    score: float
    exact: bool = False
    suggestion: SoftwareRow | None = None


def normalize_name(s: str) -> str:
    """Lowercase and collapse internal whitespace."""
    return " ".join(s.lower().split())


def load_registry(path: str | None = None) -> list[SoftwareRow]:
    """Load and validate the committed JSON registry."""
    p = Path(path or settings.software_registry_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return [SoftwareRow(**row) for row in data]


def build_name_index(rows: list[SoftwareRow]) -> dict[str, SoftwareRow]:
    """Map each normalized name AND alias -> its row."""
    index: dict[str, SoftwareRow] = {}
    for row in rows:
        index[normalize_name(row.name)] = row
        for alias in row.aliases:
            index[normalize_name(alias)] = row
    return index


def lookup_name(query: str, index: dict[str, SoftwareRow], threshold: float) -> LookupResult:
    """Exact match, else fuzzy match >= threshold. Below threshold -> row=None with
    the closest candidate as `suggestion` (for a 'did you mean' hint)."""
    q = normalize_name(query)
    if not index:
        return LookupResult(row=None, score=0.0)
    if q in index:
        return LookupResult(row=index[q], score=100.0, exact=True)

    match = process.extractOne(q, list(index.keys()), scorer=fuzz.WRatio)
    if match is None:
        return LookupResult(row=None, score=0.0)
    matched_key, score = match[0], match[1]
    if score >= threshold:
        return LookupResult(row=index[matched_key], score=score, exact=False)
    return LookupResult(row=None, score=score, suggestion=index[matched_key])


def software_embed_text(row: SoftwareRow) -> str:
    """Text embedded for the semantic (category) fallback. Forbidden rows fold in
    category + description + alternative; allowed rows are name (+ note)."""
    if row.status == "forbidden":
        parts = [row.name]
        if row.category:
            parts.append(row.category)
        if row.note:
            parts.append(row.note)
        if row.alternative:
            parts.append(f"Alternative: {row.alternative}")
        return " | ".join(parts)
    return f"{row.name} | {row.note}" if row.note else row.name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_software_registry.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add rag/software_registry.py tests/unit/test_software_registry.py
git commit -m "feat(software): registry row model + fuzzy name lookup"
```

---

## Task 3: Bootstrap the registry data (`scripts/build_software_registry.py` + JSON)

Parse the PDFs into the canonical JSON, review it by hand, commit it. The committed JSON is the source of truth; the script is a best-effort bootstrap (the Forbidden table has merged cells that need manual cleanup).

**Files:**
- Create: `scripts/build_software_registry.py`
- Create: `software/software_registry.json` (generated, hand-reviewed, committed)
- Move: `data/Allowed Software.pdf` → `software/Allowed Software.pdf`; `data/Forbidden Software.pdf` → `software/Forbidden Software.pdf`
- Modify: `.gitignore`
- Test: `tests/docs/test_software_registry_data.py`

**Interfaces:**
- Consumes: `SoftwareRow`, `load_registry` (Task 2); `pdfplumber` (Task 1).
- Produces: the committed `software/software_registry.json` conforming to `SoftwareRow`.

- [ ] **Step 1: Move the PDFs and ignore raw artifacts**

```bash
mkdir -p software
git mv "data/Allowed Software.pdf" "software/Allowed Software.pdf" 2>/dev/null || mv "data/Allowed Software.pdf" "software/Allowed Software.pdf"
git mv "data/Forbidden Software.pdf" "software/Forbidden Software.pdf" 2>/dev/null || mv "data/Forbidden Software.pdf" "software/Forbidden Software.pdf"
```

Add to `.gitignore`:

```
# Software list raw artifacts (JSON is the committed canonical source)
software/*.pdf
```

- [ ] **Step 2: Write the bootstrap script**

Create `scripts/build_software_registry.py`:

```python
"""One-time bootstrap: parse the Allowed/Forbidden Software PDFs into
software/software_registry.json. Best-effort — the Forbidden table has merged
cells; review and hand-correct the JSON (especially `alternative` and `aliases`)
before committing. The JSON, not the PDF, is the canonical source of truth.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber  # noqa: E402

from config import settings  # noqa: E402


def _clean(cell: str | None) -> str:
    return " ".join((cell or "").split())


def _extract_aliases(name: str) -> list[str]:
    """Pull example tool names out of a descriptive forbidden name, e.g.
    'Any torrent client (uTorrent, BitTorrent, qBittorrent)' -> [uTorrent, ...].
    Also splits multi-line bundles (JetBrains) on newlines."""
    aliases: list[str] = []
    paren = re.search(r"\(([^)]*)\)", name)
    if paren:
        inner = re.sub(r"(?i)\b(for example|e\.g\.|etc\.?)\b", "", paren.group(1))
        aliases += [a.strip() for a in re.split(r"[,;]", inner) if a.strip()]
    return aliases


def parse_allowed(pdf_path: Path) -> list[dict]:
    rows: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for r in table:
                    name = _clean(r[0]) if r else ""
                    if not name or name.lower() == "name":
                        continue
                    note = _clean(r[1]) if len(r) > 1 else ""
                    rows.append({
                        "name": name, "status": "allowed", "category": "",
                        "note": note, "alternative": "", "aliases": [],
                        "source_list": "Allowed Software",
                    })
    return rows


def parse_forbidden(pdf_path: Path) -> list[dict]:
    rows: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for r in table:
                    name = _clean(r[0]) if r else ""
                    if not name or name.lower() == "name":
                        continue
                    category = _clean(r[1]) if len(r) > 1 else ""
                    description = _clean(r[2]) if len(r) > 2 else ""
                    alternative = _clean(r[3]) if len(r) > 3 else ""
                    remarks = _clean(r[4]) if len(r) > 4 else ""
                    note = " ".join(p for p in (description, remarks) if p)
                    rows.append({
                        "name": name, "status": "forbidden", "category": category,
                        "note": note, "alternative": alternative,
                        "aliases": _extract_aliases(name),
                        "source_list": "Forbidden Software",
                    })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Bootstrap software_registry.json from PDFs")
    parser.add_argument("--folder", default=settings.software_docs_folder)
    parser.add_argument("--out", default=settings.software_registry_path)
    args = parser.parse_args()

    folder = Path(args.folder)
    rows = parse_allowed(folder / "Allowed Software.pdf") + parse_forbidden(folder / "Forbidden Software.pdf")
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.out}. REVIEW merged cells + aliases before committing.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate and hand-review the JSON**

Run: `PYTHONPATH=. python scripts/build_software_registry.py`
Expected: `Wrote N rows to ./software/software_registry.json`.

Then open `software/software_registry.json` and hand-correct against the PDFs (merged `alternative` cells in the Forbidden table repeat/omit; ensure `aliases` list the individual JetBrains products and the VPN/torrent examples). These specific rows MUST be present and correct (asserted by the test in Step 4):
- `Docker` → status `allowed`, note mentions "license".
- `TeamViewer` → status `forbidden`.
- `JetBrains Software` (forbidden) → `alternative` contains "VS Code"; `aliases` include "PyCharm".
- The personal-VPN forbidden row → `alternative` contains "Cisco AnyConnect".
- `Cisco AnyConnect (VPN)` → status `allowed`.

- [ ] **Step 4: Write the data-validation test**

Create `tests/docs/test_software_registry_data.py`:

```python
import pytest

from config import settings
from pathlib import Path
from rag.software_registry import load_registry

_PRESENT = Path(settings.software_registry_path).is_file()
pytestmark = pytest.mark.skipif(not _PRESENT, reason="software registry JSON not present")


def test_registry_loads_and_has_both_lists():
    rows = load_registry()
    assert len(rows) >= 50
    statuses = {r.status for r in rows}
    assert statuses == {"allowed", "forbidden"}
    for r in rows:
        assert r.name.strip()
        assert r.status in ("allowed", "forbidden")
        assert r.source_list in ("Allowed Software", "Forbidden Software")


def _find(rows, name):
    return next((r for r in rows if r.name.lower() == name.lower()), None)


def test_known_rows_are_correct():
    rows = load_registry()
    docker = _find(rows, "Docker")
    assert docker and docker.status == "allowed" and "license" in docker.note.lower()

    tv = _find(rows, "TeamViewer")
    assert tv and tv.status == "forbidden"

    jb = _find(rows, "JetBrains Software")
    assert jb and "vs code" in jb.alternative.lower()
    assert any("pycharm" in a.lower() for a in jb.aliases)

    vpn = next((r for r in rows if r.status == "forbidden" and "vpn" in r.name.lower()), None)
    assert vpn and "cisco anyconnect" in vpn.alternative.lower()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=. pytest tests/docs/test_software_registry_data.py -v`
Expected: PASS (skips only if the JSON is missing).

- [ ] **Step 6: Commit**

```bash
git add software/software_registry.json scripts/build_software_registry.py tests/docs/test_software_registry_data.py .gitignore
git commit -m "feat(software): bootstrap script + canonical registry JSON"
```

---

## Task 4: Vector store — software collection support

**Files:**
- Modify: `rag/vector_store.py`
- Test: `tests/unit/test_vector_store_software.py`

**Interfaces:**
- Consumes: `settings.software_collection`, `settings.qdrant_vector_dim`.
- Produces:
  - `search_vectors(query_vector, top_k=None, collection_name=None)` — `collection_name` defaults to `settings.qdrant_collection` (backward compatible).
  - `scroll_by_filter(filter_conditions, limit=10, collection_name=None)` — same default.
  - `init_software_collection() -> None`
  - `upsert_software_rows(rows: list[dict], embeddings: list[list[float]]) -> None`
  - `scroll_all(collection_name: str, limit: int = 2000) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_vector_store_software.py`:

```python
import types

import rag.vector_store as vs
from config import settings


class _FakeClient:
    def __init__(self):
        self.query_calls = []
        self.scroll_calls = []

    def query_points(self, collection_name, query, limit, with_payload):
        self.query_calls.append(collection_name)
        return types.SimpleNamespace(points=[])

    def scroll(self, collection_name, limit, with_payload):
        self.scroll_calls.append(collection_name)
        pt = types.SimpleNamespace(payload={"name": "Docker", "status": "allowed"})
        return ([pt], None)


def test_search_vectors_defaults_to_policy_collection(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    vs.search_vectors([0.1, 0.2], top_k=3)
    assert fake.query_calls == [settings.qdrant_collection]


def test_search_vectors_honors_collection_name(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    vs.search_vectors([0.1, 0.2], top_k=3, collection_name="software_registry")
    assert fake.query_calls == ["software_registry"]


def test_scroll_all_returns_payloads(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    rows = vs.scroll_all("software_registry")
    assert rows == [{"name": "Docker", "status": "allowed"}]
    assert fake.scroll_calls == ["software_registry"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_vector_store_software.py -v`
Expected: FAIL (TypeError: unexpected keyword `collection_name` / no `scroll_all`).

- [ ] **Step 3: Implement the changes**

In `rag/vector_store.py`, add `uuid` to the imports at the top:

```python
import uuid
```

Replace `search_vectors` (lines 103-114) with:

```python
def search_vectors(
    query_vector: list[float], top_k: int | None = None, collection_name: str | None = None
) -> list:
    """Search for similar vectors, returns list of ScoredPoint."""
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=collection_name or settings.qdrant_collection,
        query=query_vector,
        limit=top_k or settings.retrieval_top_k,
        with_payload=True,
    )
    return response.points
```

Replace `scroll_by_filter` (lines 117-126) with:

```python
def scroll_by_filter(
    filter_conditions: Filter, limit: int = 10, collection_name: str | None = None
) -> list:
    """Scroll through points matching a filter."""
    client = get_qdrant_client()
    results, _ = client.scroll(
        collection_name=collection_name or settings.qdrant_collection,
        scroll_filter=filter_conditions,
        limit=limit,
        with_payload=True,
    )
    return results
```

Append at the end of the file:

```python
def init_software_collection() -> None:
    """Create the software_registry collection with its payload indexes if absent."""
    client = get_qdrant_client()
    name = settings.software_collection
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_dim,
                distance=Distance.COSINE,
            ),
        )
        for field in ("name", "status", "category"):
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )


def upsert_software_rows(rows: list[dict], embeddings: list[list[float]]) -> None:
    """Upsert software rows (plain dict payloads) with deterministic UUID ids
    derived from the name, so re-ingest overwrites rather than duplicates."""
    client = get_qdrant_client()
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, row["name"])),
            vector=embedding,
            payload=row,
        )
        for row, embedding in zip(rows, embeddings)
    ]
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=settings.software_collection, points=points[i : i + batch_size])


def scroll_all(collection_name: str, limit: int = 2000) -> list[dict]:
    """Return all payloads in a collection (small collections only)."""
    client = get_qdrant_client()
    results, _ = client.scroll(collection_name=collection_name, limit=limit, with_payload=True)
    return [p.payload for p in results]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/unit/test_vector_store_software.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Confirm no regression on existing store callers**

Run: `PYTHONPATH=. pytest tests/unit -q`
Expected: PASS (no existing test broke from the signature change).

- [ ] **Step 6: Commit**

```bash
git add rag/vector_store.py tests/unit/test_vector_store_software.py
git commit -m "feat(software): vector_store collection_name param + software collection helpers"
```

---

## Task 5: Ingest script (`scripts/ingest_software.py`)

**Files:**
- Create: `scripts/ingest_software.py`

**Interfaces:**
- Consumes: `load_registry`, `software_embed_text` (Task 2); `init_software_collection`, `upsert_software_rows` (Task 4); `embed_texts` (`rag.embeddings`); `init_observability`.
- Produces: an ingested `software_registry` Qdrant collection.

- [ ] **Step 1: Write the ingest CLI**

Create `scripts/ingest_software.py`:

```python
"""CLI: ingest software/software_registry.json into the software_registry collection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()  # must precede any LlamaIndex/embedding import

from config import settings  # noqa: E402
from rag.embeddings import embed_texts  # noqa: E402
from rag.software_registry import load_registry, software_embed_text  # noqa: E402
from rag.vector_store import init_software_collection, upsert_software_rows  # noqa: E402


def main():
    rows = load_registry(settings.software_registry_path)
    if not rows:
        print("No software rows found.")
        return
    init_software_collection()
    embed_inputs = [software_embed_text(r) for r in rows]
    embeddings = embed_texts(embed_inputs)
    upsert_software_rows([r.model_dump() for r in rows], embeddings)
    print(f"Ingested {len(rows)} software rows into '{settings.software_collection}'.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the ingest against a running Qdrant + embedding backend**

Run: `docker compose up -d` (local Qdrant) then `PYTHONPATH=. python scripts/ingest_software.py`
Expected: `Ingested N software rows into 'software_registry'.`

(If no local embedding/Qdrant stack is available, this manual verification step is deferred to the environment that has it — the tool's unit tests in Task 6 do not require it.)

- [ ] **Step 3: Commit**

```bash
git add scripts/ingest_software.py
git commit -m "feat(software): ingest script (JSON -> software_registry collection)"
```

---

## Task 6: The `check_software` tool

**Files:**
- Create: `rag/tools/check_software.py`
- Test: `tests/unit/test_check_software.py`

**Interfaces:**
- Consumes: `SoftwareRow`, `build_name_index`, `lookup_name` (Task 2); `scroll_all`, `search_vectors` (Task 4); `embed_query` (`rag.embeddings`); `retry_transient`, `is_transient`, `RETRY_BACKOFFS` (`rag.resilience`); `record_infra_unavailable` (`rag.observability`).
- Produces:
  - `check_software(name: str) -> str` and `check_software_tool` (a `FunctionTool`).
  - Module flags read by the bot: `_software_unavailable: bool`, `_software_not_found: bool`, `_software_suggestion: dict | None`, `_software_query: str`.
  - Sentinels: `"SOFTWARE_LOOKUP_UNAVAILABLE"`, `"SOFTWARE_NOT_LISTED"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_check_software.py`:

```python
import types

import pytest

import rag.tools.check_software as cs
from rag.software_registry import SoftwareRow

ROWS = [
    SoftwareRow(name="Docker", status="allowed", note="license required", source_list="Allowed Software"),
    SoftwareRow(name="TeamViewer", status="forbidden",
                category="Prohibited software - remote desktop control",
                note="risk of unauthorized access",
                alternative="Microsoft Teams remote desktop sharing and control",
                source_list="Forbidden Software"),
    SoftwareRow(name="Any personal/non-corporate VPN", status="forbidden",
                category="Prohibited software - personal VPN",
                note="Personal VPN must not be used for business",
                alternative="Corporate Cisco AnyConnect VPN",
                aliases=["ExpressVPN", "NordVPN"],
                source_list="Forbidden Software"),
]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    cs._name_index = None
    cs._software_unavailable = False
    cs._software_not_found = False
    cs._software_suggestion = None
    monkeypatch.setattr(cs, "scroll_all", lambda coll: [r.model_dump() for r in ROWS])


def _hit(row, score):
    return types.SimpleNamespace(payload=row.model_dump(), score=score)


def test_exact_allowed_returns_status(monkeypatch):
    out = cs.check_software("Docker")
    assert "[Software 1] Docker" in out
    assert "Status: allowed" in out
    assert cs._software_not_found is False


def test_forbidden_includes_alternative():
    out = cs.check_software("TeamViewer")
    assert "Status: forbidden" in out
    assert "Alternative: Microsoft Teams" in out


def test_alias_resolves():
    out = cs.check_software("NordVPN")
    assert "Status: forbidden" in out
    assert "Cisco AnyConnect" in out


def test_category_query_uses_semantic_when_name_misses(monkeypatch):
    # name lookup misses "vpn"; semantic returns the VPN forbidden row above the floor
    monkeypatch.setattr(cs, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(cs, "search_vectors",
                        lambda qv, top_k, collection_name: [_hit(ROWS[2], 0.9)])
    out = cs.check_software("VPN")
    assert "Status: forbidden" in out and "Cisco AnyConnect" in out


def test_not_listed_when_semantic_below_floor(monkeypatch):
    monkeypatch.setattr(cs, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(cs, "search_vectors",
                        lambda qv, top_k, collection_name: [_hit(ROWS[0], 0.1)])  # below floor
    out = cs.check_software("SomeNicheToolXYZ")
    assert out.startswith("SOFTWARE_NOT_LISTED")
    assert cs._software_not_found is True


def test_transient_embed_failure_sets_unavailable(monkeypatch):
    def boom(_q):
        raise ConnectionError("embedding server down")
    monkeypatch.setattr(cs, "embed_query", boom)
    out = cs.check_software("SomethingNotAName")  # forces semantic path
    assert out == "SOFTWARE_LOOKUP_UNAVAILABLE"
    assert cs._software_unavailable is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_check_software.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the tool**

Create `rag/tools/check_software.py`:

```python
"""check_software tool: hybrid allowed/forbidden lookup.

Fuzzy name match first (in-memory index built by scrolling the software_registry
collection), then a score-gated semantic fallback for category questions
("what VPN can I use?"). Mirrors search_policies' resilience pattern: transient
backend failures set a module flag and return a sentinel instead of raising.
"""

from llama_index.core.tools import FunctionTool

from config import settings
from rag.embeddings import embed_query
from rag.observability import record_infra_unavailable
from rag.resilience import RETRY_BACKOFFS, is_transient, retry_transient
from rag.software_registry import (
    SoftwareRow,
    build_name_index,
    lookup_name,
)
from rag.vector_store import scroll_all, search_vectors

_name_index: dict[str, SoftwareRow] | None = None

# Read by channels/teams/bot.py after the agent run:
_software_unavailable: bool = False
_software_not_found: bool = False
_software_suggestion: dict | None = None
_software_query: str = ""

UNAVAILABLE = "SOFTWARE_LOOKUP_UNAVAILABLE"
NOT_LISTED = "SOFTWARE_NOT_LISTED"


def _get_name_index() -> dict[str, SoftwareRow]:
    global _name_index
    if _name_index is None:
        rows = [SoftwareRow(**p) for p in scroll_all(settings.software_collection)]
        _name_index = build_name_index(rows)
    return _name_index


def format_software_results(rows: list[SoftwareRow]) -> str:
    lines = ["=== SOFTWARE REGISTRY RESULTS ==="]
    for i, r in enumerate(rows):
        lines.append("")
        lines.append(f"[Software {i + 1}] {r.name}")
        lines.append(f"Status: {r.status}")
        lines.append(f"List: {r.source_list}")
        if r.category:
            lines.append(f"Category: {r.category}")
        if r.note:
            lines.append(f"Note: {r.note}")
        if r.alternative:
            lines.append(f"Alternative: {r.alternative}")
    return "\n".join(lines)


def check_software(name: str) -> str:
    """
    Look up whether a specific software/tool is ALLOWED or FORBIDDEN for company use,
    or find the approved alternative for a category of tool.

    Args:
        name: The specific software name (e.g. "Docker", "TeamViewer") OR a short
              category phrase (e.g. "personal VPN", "IDE"). Pass the name/category
              only — NOT the whole question.

    Returns:
        Formatted registry result(s) with Status (allowed/forbidden), source List,
        Category, Note, and Alternative. Returns "SOFTWARE_NOT_LISTED" (optionally with
        a "DID_YOU_MEAN" hint) if not found, or "SOFTWARE_LOOKUP_UNAVAILABLE" on a
        transient backend failure.
    """
    global _software_unavailable, _software_not_found, _software_suggestion, _software_query
    _software_unavailable = False
    _software_not_found = False
    _software_suggestion = None
    _software_query = name

    # Step 1: fuzzy name lookup (build the index from Qdrant on first use)
    try:
        index = retry_transient(_get_name_index)
    except Exception as exc:
        if is_transient(exc):
            _software_unavailable = True
            record_infra_unavailable("software_qdrant", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    result = lookup_name(name, index, settings.software_fuzzy_threshold)
    if result.row is not None:
        return format_software_results([result.row])

    # Step 2: score-gated semantic fallback (category questions)
    try:
        query_vector = retry_transient(lambda: embed_query(name))
    except Exception as exc:
        if is_transient(exc):
            _software_unavailable = True
            record_infra_unavailable("software_embeddings", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    try:
        raw = retry_transient(
            lambda: search_vectors(query_vector, top_k=5, collection_name=settings.software_collection)
        )
    except Exception as exc:
        if is_transient(exc):
            _software_unavailable = True
            record_infra_unavailable("software_qdrant", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    hits = [h for h in raw if h.score >= settings.software_min_semantic_score]
    if hits:
        return format_software_results([SoftwareRow(**h.payload) for h in hits])

    # Step 3: not found — offer the closest fuzzy candidate as a hint
    _software_not_found = True
    if result.suggestion is not None:
        _software_suggestion = {"name": result.suggestion.name, "status": result.suggestion.status}
        return f"{NOT_LISTED}\nDID_YOU_MEAN: {result.suggestion.name} ({result.suggestion.status})"
    return NOT_LISTED


check_software_tool = FunctionTool.from_defaults(fn=check_software)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_check_software.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add rag/tools/check_software.py tests/unit/test_check_software.py
git commit -m "feat(software): check_software hybrid tool with resilience"
```

---

## Task 7: Agent integration

**Files:**
- Modify: `rag/agent.py`
- Test: `tests/unit/test_agent_software.py`

**Interfaces:**
- Consumes: `check_software_tool` (Task 6); `settings.software_lookup_enabled`.
- Produces: `Citation` with optional `status` and `alternative`; `build_agent()` includes `check_software` when enabled.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_agent_software.py`:

```python
from rag.agent import Citation, ALL_TOOLS


def test_citation_software_fields_default_empty():
    c = Citation(doc_title="Allowed Software", section="", clause="",
                 clause_number="", quote="license required")
    assert c.status == ""
    assert c.alternative == ""


def test_citation_accepts_software_status():
    c = Citation(doc_title="Forbidden Software", section="remote desktop", clause="",
                 clause_number="", quote="risk", status="forbidden",
                 alternative="Microsoft Teams")
    assert c.status == "forbidden"
    assert c.alternative == "Microsoft Teams"


def test_check_software_tool_registered():
    names = {getattr(t.metadata, "name", None) for t in ALL_TOOLS}
    assert "check_software" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_agent_software.py -v`
Expected: FAIL (unexpected kwarg `status`; `check_software` not in ALL_TOOLS).

- [ ] **Step 3: Add the Citation fields**

In `rag/agent.py`, add the import near the other tool imports (top of file):

```python
from rag.tools.check_software import check_software_tool
```

In the `Citation` model, after the `quote` field (line 30), add:

```python
    status: str = Field(
        default="",
        description="For software answers only: 'allowed' or 'forbidden'. Empty string for policy citations.",
    )
    alternative: str = Field(
        default="",
        description="For forbidden-software answers only: the approved alternative tool. Empty string otherwise.",
    )
```

- [ ] **Step 4: Register the tool (gated)**

Replace the `ALL_TOOLS` list (lines 148-152) with:

```python
ALL_TOOLS = [
    search_policies_tool,
    get_section_tool,
    escalate_to_compliance_tool,
]
if settings.software_lookup_enabled:
    # inserted before escalate so the agent sees both search tools first
    ALL_TOOLS.insert(1, check_software_tool)
```

- [ ] **Step 5: Extend the system prompt**

In `rag/agent.py`, in `SYSTEM_PROMPT`, change the first line of `== HOW TO RESPOND ==` step 1 (line 72) from:

```
1. Call search_policies FIRST for every question. Never answer without searching.
```

to:

```
1. Call a search tool FIRST for every question. Never answer without searching.
   - For questions about whether a specific software/tool is allowed or forbidden, or
     "what tool/software can I use for X", call check_software (pass the software NAME or
     a short category phrase like "personal VPN" or "IDE" — not the whole sentence).
   - For questions about a policy, rule, or procedure, call search_policies (pass the
     original question verbatim, per step 0).
   - A question may need both tools; call both when it does.
```

Then insert a new section immediately before `== ANSWER FORMAT ==` (before line 81):

```
== SOFTWARE QUESTIONS (check_software) ==

- State the Status (allowed or forbidden) EXACTLY as returned. Never infer a status.
  "Not forbidden" does NOT mean allowed.
- Cite the source List ("Allowed Software" / "Forbidden Software") as the document.
- For a forbidden tool, include the Alternative from the result.
- If check_software returns SOFTWARE_NOT_LISTED, do not guess. Return a short answer
  saying it is not on the approved or forbidden list and to request approval from IT;
  set escalation.needed=false.
- In the citations array for a software answer, set: doc_title = the List, section = the
  Category (empty for allowed), status = "allowed"/"forbidden", alternative = the
  Alternative (empty if none), quote = the Note. Leave clause and clause_number empty.
```

Then in `== OUTPUT FORMAT ==`, extend the example citation objects to document the new optional fields by adding these two lines inside each citation object (after `"quote": ...`):

```
      "status": "",
      "alternative": ""
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_agent_software.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add rag/agent.py tests/unit/test_agent_software.py
git commit -m "feat(software): register check_software tool + software answer schema/prompt"
```

---

## Task 8: Teams rendering

**Files:**
- Modify: `channels/teams/renderer.py`
- Test: `tests/unit/test_renderer_software.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `render_answer` handles software citations; `render_software_not_listed(name: str, suggestion: dict | None = None) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_renderer_software.py`:

```python
from channels.teams.renderer import render_answer, render_software_not_listed


def test_render_allowed_software():
    result = {"answer": "", "citations": [
        {"doc_title": "Allowed Software", "section": "", "clause": "", "clause_number": "",
         "quote": "A license is required for commercial usage", "status": "allowed", "alternative": ""},
    ]}
    html = render_answer(result)
    assert "✅ Allowed" in html
    assert "Allowed Software" in html
    assert '"A license is required for commercial usage"' in html
    assert "<div" not in html


def test_render_forbidden_software_with_alternative():
    result = {"answer": "", "citations": [
        {"doc_title": "Forbidden Software", "section": "remote desktop control", "clause": "",
         "clause_number": "", "quote": "risk of unauthorized access", "status": "forbidden",
         "alternative": "Microsoft Teams remote desktop sharing"},
    ]}
    html = render_answer(result)
    assert "⛔ Forbidden" in html
    assert "remote desktop control" in html
    assert "<b>Alternative:</b> Microsoft Teams remote desktop sharing" in html


def test_policy_citation_still_renders_old_way():
    result = {"answer": "", "citations": [
        {"doc_title": "AUP", "section": "Use", "clause": "Email", "clause_number": "4.7",
         "quote": "No spam."},
    ]}
    html = render_answer(result)
    assert "📄 AUP" in html
    assert "✅" not in html and "⛔" not in html


def test_not_listed_with_suggestion():
    html = render_software_not_listed("Dockerr", {"name": "Docker", "status": "allowed"})
    assert "isn't on the approved or forbidden software list" in html
    assert "Docker" in html and "Allowed" in html
    assert "support@trinetix.com" in html
    assert "<div" not in html


def test_not_listed_without_suggestion():
    html = render_software_not_listed("ZzzTool", None)
    assert "ZzzTool" in html
    assert "support@trinetix.com" in html
    assert "Did you mean" not in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_renderer_software.py -v`
Expected: FAIL (`render_software_not_listed` missing; software layout absent).

- [ ] **Step 3: Implement rendering**

In `channels/teams/renderer.py`, add a helper above `render_answer`:

```python
def _render_software_citation(c: dict) -> str:
    status = c.get("status", "").lower()
    doc = c.get("doc_title", "")
    category = c.get("section", "")
    quote = c.get("quote", "")
    alt = c.get("alternative", "")

    if status == "allowed":
        head = f"<b>✅ Allowed — {doc}</b>"
    else:
        head = f"<b>⛔ Forbidden{f' — {category}' if category else ''}</b>"

    lines = [f"<p>{head}</p>"]
    if quote:
        lines.append(f'<p><i>"{quote}"</i></p>')
    if alt:
        lines.append(f"<p><b>Alternative:</b> {alt}</p>")
    return "\n".join(lines)
```

Replace the body of `render_answer` (lines 73-116) with:

```python
    citations = result.get("citations", [])
    answer = result.get("answer", "")

    if not citations:
        # Fallback to prose answer when there are no structured citations
        return f"<p>{answer}</p>"

    has_software = any(c.get("status") for c in citations)
    policy_citations = [c for c in citations if not c.get("status")]

    parts = []
    # Only show the multi-policy header for a pure-policy multi-citation answer
    if len(policy_citations) > 1 and not has_software:
        parts.append(f"<p>This is addressed in {len(policy_citations)} policies:</p>")

    for i, c in enumerate(citations):
        if i > 0:
            parts.append("<hr>")

        if c.get("status"):
            parts.append(_render_software_citation(c))
            continue

        doc = c.get("doc_title", "")
        section = c.get("section", "")
        clause = c.get("clause", "")
        clause_num = c.get("clause_number", "")
        quote = c.get("quote", "")

        location_lines = []
        if doc:
            location_lines.append(f"<b>📄 {doc}</b>")
        if section:
            location_lines.append(f"<b>Section:</b> {section}")
        clause_is_just_number = clause and clause_num and clause.strip() == clause_num.strip()
        if clause_is_just_number:
            location_lines.append(f"<b>Clause {clause_num}</b>")
        elif clause and clause_num:
            location_lines.append(f"<b>Clause {clause_num}:</b> {clause}")
        elif clause:
            location_lines.append(f"<b>Clause:</b> {clause}")
        elif clause_num:
            location_lines.append(f"<b>Clause {clause_num}</b>")

        parts.append(f"<p>{'<br>'.join(location_lines)}</p>")

        if quote:
            parts.append(f'<p><i>"{quote}"</i></p>')

    return "\n".join(parts)
```

Add near the other canned-message renderers (e.g. after `render_unintelligible`):

```python
# Shown when software is on neither the allowed nor forbidden list. Editable; Teams-safe HTML only.
def render_software_not_listed(name: str, suggestion: dict | None = None) -> str:
    parts = [f"<p><b>'{name}' isn't on the approved or forbidden software list.</b></p>"]
    if suggestion:
        parts.append(
            f"<p>Did you mean <b>{suggestion['name']}</b> ({suggestion['status'].capitalize()})?</p>"
        )
    parts.append(
        "<p>Request approval from IT before installing "
        "(<b>support@trinetix.com</b>).</p>"
    )
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_renderer_software.py tests/unit/test_renderer.py -v`
Expected: PASS (existing renderer tests still pass; new ones pass).

- [ ] **Step 5: Commit**

```bash
git add channels/teams/renderer.py tests/unit/test_renderer_software.py
git commit -m "feat(software): Teams rendering for software answers + not-listed reply"
```

---

## Task 9: Bot wiring (`_run_rag` + `_send_reply`)

**Files:**
- Modify: `channels/teams/bot.py`
- Test: `tests/unit/test_bot_software.py`

**Interfaces:**
- Consumes: `check_software` module flags (Task 6); `render_software_not_listed` (Task 8).
- Produces: `_run_rag` returns `{"status": "software_not_found", "name", "suggestion"}` when the software tool found nothing and no policy citation was produced; `_send_reply` renders it with no rating prompt.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_bot_software.py`:

```python
from channels.teams.bot import TeamsBot


class _Bot(TeamsBot):
    def __init__(self):
        # bypass TeamsBot.__init__ (no token refresher / state needed for this test)
        self.sent = []

    def _send_message(self, chat_id, text, content_type="html"):
        self.sent.append(text)
        return True


def test_software_not_found_renders_not_listed_and_no_rating(monkeypatch):
    import channels.teams.bot as botmod

    monkeypatch.setattr(botmod.settings, "router_enabled", False)
    monkeypatch.setattr(
        botmod, "_run_rag",
        lambda q: {"status": "software_not_found", "name": "ZzzTool",
                   "suggestion": {"name": "Docker", "status": "allowed"}},
    )

    bot = _Bot()
    bot._send_reply("chat1", "is ZzzTool allowed?", sender_name="Alice")

    joined = "\n".join(bot.sent)
    assert "isn't on the approved or forbidden software list" in joined
    assert "Was this helpful?" not in joined  # no rating prompt
    assert "chat1" not in botmod._pending_ratings  # no pending feedback row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_bot_software.py -v`
Expected: FAIL (software_not_found not handled → falls through to render_error / rating prompt path).

- [ ] **Step 3: Update `_run_rag`**

In `channels/teams/bot.py`, update the deferred imports and flag resets inside `_run_rag`. Change lines 49-55 from:

```python
    import asyncio
    import rag.tools.search_policies as sp
    from rag.agent import build_agent
    from rag.response import parse_agent_response
    from rag.resilience import retry_transient, is_transient, RETRY_BACKOFFS
    from rag.observability import record_infra_unavailable

    sp._retrieval_unavailable = False
```

to:

```python
    import asyncio
    import rag.tools.search_policies as sp
    import rag.tools.check_software as cs
    from rag.agent import build_agent
    from rag.response import parse_agent_response
    from rag.resilience import retry_transient, is_transient, RETRY_BACKOFFS
    from rag.observability import record_infra_unavailable

    sp._retrieval_unavailable = False
    cs._software_unavailable = False
    cs._software_not_found = False
```

Then replace the tail of `_run_rag` (lines 74-79) from:

```python
    # Retrieval failed inside the tool (LlamaIndex swallows tool exceptions) →
    # the flag was set in search_policies; surface the unavailable outcome.
    if sp._retrieval_unavailable:
        return {"status": "unavailable"}

    return parse_agent_response(str(response))
```

to:

```python
    # Retrieval failed inside a tool (LlamaIndex swallows tool exceptions) →
    # the flag was set in the tool; surface the unavailable outcome.
    if sp._retrieval_unavailable or cs._software_unavailable:
        return {"status": "unavailable"}

    parsed = parse_agent_response(str(response))

    # Software not on either list, and the agent produced no policy citation either →
    # deterministic "not listed" outcome (no rating prompt). A blended answer that DID
    # cite a policy still renders normally below.
    if cs._software_not_found and not parsed.get("citations"):
        return {
            "status": "software_not_found",
            "name": cs._software_query,
            "suggestion": cs._software_suggestion,
        }

    return parsed
```

- [ ] **Step 4: Update `_send_reply`**

In `channels/teams/bot.py`, add `render_software_not_listed` to the renderer import block (around lines 17-24) — add the name to the existing `from channels.teams.renderer import (...)` list.

Then, immediately after the `unavailable` handling block (after line 282, before `# Render response`), add:

```python
        # Software not on either list — deterministic guidance, no rating prompt.
        if result.get("status") == "software_not_found":
            html = render_software_not_listed(result.get("name", ""), result.get("suggestion"))
            sent = self._send_message(chat_id, html)
            if sent:
                print("Software not-listed notice sent")
            return bool(sent)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_bot_software.py tests/unit/test_bot_routing.py -v`
Expected: PASS (new test passes; existing bot routing tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add channels/teams/bot.py tests/unit/test_bot_software.py
git commit -m "feat(software): bot wiring for software_not_found + unavailable outcomes"
```

---

## Task 10: Live accuracy test + documentation

**Files:**
- Create: `tests/_software.py`
- Create: `tests/live/test_software_lookup.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `check_software` (Task 6); `llm_reachable` (`tests/_llm.py`).
- Produces: `software_registry_ready() -> bool`.

- [ ] **Step 1: Write the readiness probe**

Create `tests/_software.py`:

```python
"""Probe whether the software_registry collection is ingested, so live software
tests auto-skip on machines without the ingested stack."""

from config import settings


def software_registry_ready() -> bool:
    try:
        from rag.vector_store import get_qdrant_client

        client = get_qdrant_client()
        if not client.collection_exists(settings.software_collection):
            return False
        return client.count(settings.software_collection).count > 0
    except Exception:
        return False
```

- [ ] **Step 2: Write the live accuracy test**

Create `tests/live/test_software_lookup.py`:

```python
import pytest

from tests._llm import llm_reachable
from tests._software import software_registry_ready

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(
        not (llm_reachable() and software_registry_ready()),
        reason="needs a reachable LLM + ingested software_registry (local-only)",
    ),
]


def test_check_software_accuracy():
    """Tuning signal, not a hard gate: assert direct-tool lookups on known rows.
    Imported lazily (see the router live-test note) to avoid pulling llama-index at
    pytest collection on offline runs."""
    from rag.tools.check_software import check_software

    docker = check_software("Docker")
    assert "allowed" in docker.lower()

    tv = check_software("TeamViewer")
    assert "forbidden" in tv.lower()

    vpn = check_software("VPN")  # category question → semantic path
    assert "cisco anyconnect" in vpn.lower()

    unknown = check_software("SomeNonexistentToolXYZ123")
    assert unknown.startswith("SOFTWARE_NOT_LISTED")
```

- [ ] **Step 3: Run the live test (auto-skips without the stack)**

Run: `PYTHONPATH=. pytest tests/live/test_software_lookup.py -v`
Expected: PASS if a local LLM + ingested collection exist; otherwise SKIPPED (clean, no error).

- [ ] **Step 4: Run the full unit + docs suite**

Run: `PYTHONPATH=. pytest tests/unit tests/docs -q`
Expected: PASS (all green; corpus/software-data tests skip only if data absent).

- [ ] **Step 5: Update CLAUDE.md**

In `CLAUDE.md`:

Under the intro, add software to the corpus description:
```
Answers employee questions **strictly from approved internal policy DOCX files** and a curated **software allow/forbidden registry**; if an answer can't be grounded, it escalates to Compliance.
```

Under `## Commands`, add:
```bash
# Ingest software registry (bootstrap JSON from PDFs first, then ingest)
PYTHONPATH=. python scripts/build_software_registry.py   # PDFs -> software/software_registry.json (one-time, hand-review)
PYTHONPATH=. python scripts/ingest_software.py           # JSON -> software_registry collection
```

Under `## Architecture`, add to the `rag/` block:
```
  software_registry.py # SoftwareRow + normalize + fuzzy name index + embed-text (pure logic)
  tools/check_software # hybrid: fuzzy name lookup -> score-gated semantic fallback (call for software Qs)
```
and add a line:
```
software/            # Allowed/Forbidden Software PDFs (gitignored) + software_registry.json (canonical, committed)
```

Add a new short paragraph after the "Search flow" paragraph:
```
**Software lookup:** `check_software` (agent tool, not the router) answers "is X allowed/forbidden?" and "what can I use for X?" from `software_registry` (separate Qdrant collection). Fuzzy name lookup first (rapidfuzz, in-memory index scrolled from the collection); if no confident name hit, a semantic fallback gated by `SOFTWARE_MIN_SEMANTIC_SCORE`. Software answers reuse `ComplianceAnswer` with optional `status`/`alternative` citation fields. Not-on-either-list → deterministic `software_not_found` reply (canned "ask IT" + fuzzy "did you mean"), no rating prompt. Source of truth is the committed JSON; **future work: pluggable Excel/Confluence/SharePoint loader** for both policies and software.
```

Add to the Config section:
```bash
SOFTWARE_LOOKUP_ENABLED (kill switch) / SOFTWARE_COLLECTION / SOFTWARE_FUZZY_THRESHOLD (0-100) / SOFTWARE_MIN_SEMANTIC_SCORE (cosine floor for category fallback)
```

Add these rows to the Gotchas table:
```
| Software semantic fallback returns 5 unrelated rows for any query | Gate hits by `SOFTWARE_MIN_SEMANTIC_SCORE`; below floor → SOFTWARE_NOT_LISTED. |
| `software_not_found` suppresses a valid blended policy answer | Short-circuit only fires when the parsed answer has NO citations. |
| Forbidden PDF merged cells mis-parse (alternatives/aliases) | `build_software_registry.py` is best-effort; the committed JSON is hand-reviewed and guarded by `tests/docs/test_software_registry_data.py`. |
| `check_software` name index stale after re-ingest | Index is cached module-level (`_name_index`); a bot restart rebuilds it by scrolling the collection. |
```

- [ ] **Step 6: Commit**

```bash
git add tests/_software.py tests/live/test_software_lookup.py CLAUDE.md
git commit -m "feat(software): live accuracy test + docs"
```

---

## Self-Review (completed during authoring)

**Spec coverage:** Data model & source of truth → Task 3 (+ Task 2 `SoftwareRow`). Storage & ingest → Tasks 4, 5. `check_software` hybrid + resilience → Task 6. Agent integration (tool, prompt, schema) → Task 7. Rendering + not-found + unavailable outcomes → Tasks 8, 9. Config/deps/data location → Tasks 1, 3. Testing (unit/docs/live) → every task + Task 10. Known limitation + future-source note → documented in CLAUDE.md (Task 10) and carried in the spec. Router unchanged → respected (no router task). All spec sections map to a task.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has an expected result.

**Type consistency:** `SoftwareRow` fields, `LookupResult`, `check_software(name)`, module flag names (`_software_unavailable`, `_software_not_found`, `_software_suggestion`, `_software_query`), sentinels (`SOFTWARE_LOOKUP_UNAVAILABLE`, `SOFTWARE_NOT_LISTED`), `search_vectors(..., collection_name=)`, `scroll_all`, `render_software_not_listed(name, suggestion)`, and the `{"status": "software_not_found", "name", "suggestion"}` dict are used identically across Tasks 2, 4, 6, 7, 8, 9, 10.

**Known deferrals (from spec, intentionally not built):** allowed-side category tagging; Excel/Confluence/SharePoint source loaders; router-level software category (rejected).
