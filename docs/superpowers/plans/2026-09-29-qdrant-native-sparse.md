# Qdrant-Native Sparse Vectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move BM25 from a gitignored JSON file into Qdrant as native sparse vectors, so the lexical index lives in the collection, survives re-ingestion, needs no mount, and is fused server-side.

**Architecture:** A new collection carries an unnamed 768-dim dense vector (unchanged) plus a named `bm25` sparse vector with `modifier: idf`. Qdrant performs BM25 encoding itself via its built-in `qdrant/bm25` model, so no encoder ships in this repo. Retrieval becomes one `query_points` call with two `Prefetch` branches fused by RRF; because fusion is server-side, both the BM25-on and BM25-off paths return `list[ScoredPoint]`, collapsing three duplicated call sites into one each. A point-copying migration script builds the new collection from the existing one, leaving dense vectors byte-identical.

**Tech Stack:** Python 3.12, `qdrant-client` 1.17.0, Qdrant server 1.17.1, pytest, OpenTelemetry / OpenInference. **No new dependencies.**

**Spec:** `docs/superpowers/specs/2026-09-29-qdrant-native-sparse-design.md`

**Branch:** `feat/qdrant-native-sparse` (already created, spec committed at `96b3e3e`)

## Global Constraints

- **Do not change server-side configuration on `172.20.0.22`** — it is a shared host this project does not own. Every change here is client-side.
- **No new dependencies.** `fastembed` is deliberately absent and must stay absent; adding it would silently relocate BM25 encoding from the server to the client.
- **`cloud_inference=True`** must be set on the Qdrant client. The client default is `False`, which means "encode locally via fastembed".
- **`avg_len` must be passed in per-document `options` on every upsert and every query.** Qdrant 1.17.1 accepts a collection-level `Bm25Config` and silently discards it.
- **Sparse vector name is `bm25`; model name is `qdrant/bm25`.** Only built-in models work self-hosted; any other name fails with `InferenceService URL not configured`.
- **The dense vector stays unnamed** (empty-string key `""` in a point's vector dict), size `settings.qdrant_vector_dim` (768), `Distance.COSINE`.
- **The OpenTelemetry span keeps the name `search_vectors`** even after the function is renamed, so Phoenix comparisons against historical runs stay valid.
- **`init_observability()` must run FIRST in every entry point**, before any LlamaIndex/Ollama import.
- **Tests must not touch the network or `172.20.0.22`.** Mock the Qdrant client; never construct a real one.
- **Never let `settings` reach a pytest assertion or `AttributeError` message.** The `Settings` repr contains real `.env` secrets (`hf_token`, `teams_client_secret`, `teams_refresh_token`, `smtp_password`). Bind the value to a local variable first, then assert on the local.
- **Imports at the top of the module**, never inside functions. Existing function-local imports in the files you touch should move to the top as you collapse their branches.
- `temperature=0.0`, `num_ctx=4096`, `num_predict=1024`, exactly one worker thread, never cache the LLM client — all unchanged by this work, none of it should be touched.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `config.py` | all settings | add `bm25_avg_len: float = 50.0` |
| `rag/vector_store.py` | Qdrant client, collection schema, upsert, retrieval, preflight | the bulk of the work |
| `rag/tools/search_policies.py` | search orchestration: retrieve → rerank → floor → format | collapse two branches into one; fix the cosine-threshold guard |
| `ingest/pipeline.py` | parse → embed → upsert | delete two BM25 blocks |
| `scripts/migrate_collection.py` | **new** — copy points into a sparse-enabled collection | create |
| `scripts/start_teams_bot.py` | bot entry point | call the preflight |
| `scripts/run_eval.py` | offline retrieval eval | collapse two branches |
| `eval/run_experiment.py` | Phoenix experiments | collapse two branches; record new metadata |
| `rag/bm25_index.py` | pure-Python BM25 (230 lines) | **delete** |
| `rag/hybrid_search.py` | client-side RRF (166 lines) | **delete** |
| `scripts/build_bm25_from_qdrant.py` | index rebuild tool (149 lines) | **delete** |
| `.gitignore`, `.env.example`, `CLAUDE.md`, `rag/observability.py` | docs and config hygiene | cleanup |

**Task order matters.** Task 4 renames `search_vectors` to `search_chunks`; Tasks 5 and 8 remove the last importers of `hybrid_search`; Task 9 can only delete the modules once nothing imports them. Do not reorder.

---

### Task 1: Config setting and the `cloud_inference` flag

**Files:**
- Modify: `config.py:111-113` (the `# Hybrid search` block)
- Modify: `rag/vector_store.py:64-68` (`get_qdrant_client`)
- Test: `tests/unit/test_sparse_config.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `settings.bm25_avg_len` (`float`, default `50.0`); `get_qdrant_client()` constructs `QdrantClient(url=..., timeout=10, cloud_inference=True)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sparse_config.py`:

```python
"""Config and client wiring for Qdrant-native sparse vectors.

cloud_inference is the load-bearing one. Its name is misleading: it means
"do not encode locally, send the text to the server". The client's default is
False, which means "encode locally via fastembed". Server-side BM25 works today
only because fastembed is not installed — anyone installing it for an unrelated
reason would silently move encoding from the server to the client, changing
retrieval with no error and no log line.
"""

import rag.vector_store as vs
from config import settings


def test_bm25_avg_len_defaults_to_the_measured_corpus_average():
    # Bound to a local first: a failing assert on settings.<attr> would embed the
    # full Settings repr, which carries real .env secrets, into pytest output.
    avg_len = settings.bm25_avg_len
    assert avg_len == 50.0


def test_qdrant_client_pins_server_side_inference(monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vs, "QdrantClient", _FakeClient)
    monkeypatch.setattr(vs, "_client", None)  # module-level singleton

    vs.get_qdrant_client()

    assert captured["cloud_inference"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_sparse_config.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'bm25_avg_len'` and `KeyError: 'cloud_inference'`.

- [ ] **Step 3: Add the setting**

In `config.py`, in the `# Hybrid search` block, after `hybrid_bm25_candidates`:

```python
    # Hybrid search
    bm25_enabled: bool = True
    hybrid_vector_candidates: int = 20
    hybrid_bm25_candidates: int = 20
    # BM25 length normalisation, passed to Qdrant in per-document `options`.
    # Qdrant's default is 256; this corpus measured 1602 chunks at mean 49.7
    # tokens (median 38, p90 109, max 350) on 2026-09-29. At 256 the term
    # (1 - b + b*dl/avg_len) stays near 0.25 for every chunk, so `b` goes inert
    # and long chunks are never penalised.
    bm25_avg_len: float = 50.0
```

- [ ] **Step 4: Set the client flag**

In `rag/vector_store.py`, replace `get_qdrant_client`:

```python
def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        # cloud_inference=True means "do not embed models.Document locally, send
        # the text to the server". Despite the name it is correct for a
        # self-hosted server: Qdrant 1.17.1 resolves the built-in qdrant/bm25
        # model itself, with no InferenceService configured (verified
        # 2026-09-29). The client default is False, i.e. encode locally via
        # fastembed — which works today only because fastembed is not installed.
        _client = QdrantClient(
            url=settings.active_qdrant_url, timeout=10, cloud_inference=True
        )
    return _client
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_sparse_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add config.py rag/vector_store.py tests/unit/test_sparse_config.py
git commit -m "feat(qdrant): add bm25_avg_len and pin server-side inference"
```

---

### Task 2: Collection schema with a sparse vector

**Files:**
- Modify: `rag/vector_store.py:71-111` (`init_collection`)
- Test: `tests/unit/test_sparse_config.py` (extend)

**Interfaces:**
- Consumes: `settings.bm25_avg_len` from Task 1.
- Produces: `SPARSE_VECTOR_NAME = "bm25"`, `BM25_MODEL = "qdrant/bm25"`, and `init_collection(collection_name: str | None = None) -> None`. Task 7's migration script calls `init_collection(target)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_sparse_config.py`:

```python
from qdrant_client.models import Distance, Modifier


class _RecordingClient:
    """Records create_collection / create_payload_index calls; creates nothing."""

    def __init__(self, exists=False):
        self._exists = exists
        self.created = None
        self.indexes = []

    def collection_exists(self, name):
        return self._exists

    def create_collection(self, **kwargs):
        self.created = kwargs

    def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs)


def test_init_collection_declares_the_sparse_bm25_vector(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.init_collection()

    sparse = client.created["sparse_vectors_config"]
    assert set(sparse) == {"bm25"}
    assert sparse["bm25"].modifier == Modifier.IDF


def test_init_collection_keeps_the_dense_vector_unnamed_and_768(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(vs.settings, "qdrant_vector_dim", 768)

    vs.init_collection()

    dense = client.created["vectors_config"]
    # A bare VectorParams (not a dict) is what makes the vector unnamed. Naming it
    # would force every existing search call to specify `using=`.
    assert dense.size == 768
    assert dense.distance == Distance.COSINE


def test_init_collection_targets_a_named_collection(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.init_collection("compliance_policies_v2")

    assert client.created["collection_name"] == "compliance_policies_v2"
    assert all(i["collection_name"] == "compliance_policies_v2" for i in client.indexes)
    assert len(client.indexes) == 6


def test_init_collection_is_a_noop_when_the_collection_exists(monkeypatch):
    client = _RecordingClient(exists=True)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.init_collection()

    assert client.created is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_sparse_config.py -v`
Expected: FAIL — `KeyError: 'sparse_vectors_config'`, and `init_collection() takes 0 positional arguments`.

- [ ] **Step 3: Implement**

In `rag/vector_store.py`, extend the `qdrant_client.models` import with `Modifier` and `SparseVectorParams`, add the two constants below `_DOCUMENT_CONTENT_MAX_CHARS`, and replace `init_collection`:

```python
# The sparse vector's name inside the collection, and Qdrant's built-in BM25
# model. Only built-in models resolve on a self-hosted server; any other name
# fails with "InferenceService URL not configured", which would require
# server-side config on a shared host we do not own.
SPARSE_VECTOR_NAME = "bm25"
BM25_MODEL = "qdrant/bm25"


def init_collection(collection_name: str | None = None) -> None:
    """Create the collection with its payload indexes if it does not exist.

    Takes an explicit name so scripts/migrate_collection.py can build a target
    other than the configured one from this single schema definition, rather
    than duplicating it and letting the two drift.
    """
    name = collection_name or settings.qdrant_collection
    client = get_qdrant_client()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            # Bare VectorParams, not a dict: that is what keeps the dense vector
            # unnamed, so no search call needs `using=`. Verified that an unnamed
            # dense vector coexists with a named sparse one.
            vectors_config=VectorParams(
                size=settings.qdrant_vector_dim,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        for field, schema in (
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("section", PayloadSchemaType.KEYWORD),
            ("section_number", PayloadSchemaType.KEYWORD),
            ("clause", PayloadSchemaType.KEYWORD),
            ("clause_number", PayloadSchemaType.KEYWORD),
            ("section_display", PayloadSchemaType.TEXT),
        ):
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema
            )
```

Note: `k` and `b` are deliberately absent from `SparseVectorParams`. Qdrant 1.17.1 accepts a collection-level `Bm25Config` and silently discards it — verified by writing one and reading the collection back to find only `{"modifier": "idf"}`. They are passed per-document in Task 3.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_sparse_config.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add rag/vector_store.py tests/unit/test_sparse_config.py
git commit -m "feat(qdrant): declare a sparse bm25/idf vector on the collection"
```

---

### Task 3: Write sparse vectors at ingest

**Files:**
- Modify: `rag/vector_store.py:114-131` (`upsert_chunks`)
- Modify: `ingest/pipeline.py:21-27` and `:48-53` (delete both BM25 blocks)
- Test: `tests/unit/test_sparse_ingest.py` (create)

**Interfaces:**
- Consumes: `SPARSE_VECTOR_NAME`, `BM25_MODEL`, `settings.bm25_avg_len`.
- Produces: `bm25_document(text: str) -> Document` in `rag/vector_store.py` — the single place `avg_len` is attached. Task 4 calls it for the query side.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sparse_ingest.py`:

```python
"""Ingest writes a sparse vector alongside the dense one, unconditionally.

Unconditionally is the point. The old code built the lexical index only when
BM25 was already enabled, which is why enabling the flag later gave the bot an
empty index. Now the flag is query-side only (spec D7), so flipping it needs no
re-ingest.
"""

import rag.vector_store as vs


class _CapturingClient:
    def __init__(self):
        self.batches = []

    def upsert(self, **kwargs):
        self.batches.append(kwargs)


class _Chunk:
    """Minimal stand-in for PolicyChunk: only the fields upsert_chunks reads."""

    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text

    def model_dump(self):
        return {"text": self.text, "doc_title": "Backup Policy"}


def test_upsert_writes_both_an_unnamed_dense_and_a_named_sparse_vector(monkeypatch):
    client = _CapturingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.upsert_chunks([_Chunk("c1", "retention period is ninety days")], [[0.1, 0.2]])

    point = client.batches[0]["points"][0]
    assert set(point.vector) == {"", "bm25"}
    assert point.vector[""] == [0.1, 0.2]
    assert point.vector["bm25"].model == "qdrant/bm25"
    assert point.vector["bm25"].text == "retention period is ninety days"


def test_sparse_text_is_the_chunk_body_not_the_metadata_prefixed_string(monkeypatch):
    """Spec D6: the dense embedding prepends Document/Section/Clause; sparse does
    not. The encoder is already the variable under test, and changing the indexed
    text at the same time would make a bad eval result undiagnosable."""
    client = _CapturingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.upsert_chunks([_Chunk("c1", "body text only")], [[0.1]])

    assert client.batches[0]["points"][0].vector["bm25"].text == "body text only"


def test_avg_len_is_attached_to_every_document(monkeypatch):
    """Qdrant 1.17.1 silently discards a collection-level Bm25Config, so
    per-document options are the only place avg_len takes effect."""
    client = _CapturingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(vs.settings, "bm25_avg_len", 77.0)

    vs.upsert_chunks([_Chunk("c1", "text")], [[0.1]])

    assert client.batches[0]["points"][0].vector["bm25"].options == {"avg_len": 77.0}


def test_sparse_is_written_even_when_bm25_is_disabled(monkeypatch):
    client = _CapturingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)

    vs.upsert_chunks([_Chunk("c1", "text")], [[0.1]])

    assert "bm25" in client.batches[0]["points"][0].vector


def test_ingest_pipeline_no_longer_references_the_file_based_index():
    from pathlib import Path

    source = Path("ingest/pipeline.py").read_text(encoding="utf-8")
    assert "bm25" not in source.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_sparse_ingest.py -v`
Expected: FAIL — the point's `vector` is a bare list, so `set(point.vector)` raises `TypeError`; and `pipeline.py` still contains `bm25`.

- [ ] **Step 3: Implement the document helper and the upsert change**

In `rag/vector_store.py`, add `Document` to the `qdrant_client.models` import, add the helper after the two constants, and replace `upsert_chunks`:

```python
def bm25_document(text: str) -> Document:
    """Text handed to Qdrant for server-side BM25 encoding.

    avg_len goes in per-document `options` on EVERY call, at ingest and at query
    time alike: Qdrant 1.17.1 accepts a collection-level Bm25Config and silently
    discards it (verified 2026-09-29 — the collection reads back carrying only
    {"modifier": "idf"}), so this is the only place k/b/avg_len take effect.
    """
    return Document(
        text=text,
        model=BM25_MODEL,
        options={"avg_len": settings.bm25_avg_len},
    )


def upsert_chunks(chunks: "list[PolicyChunk]", embeddings: list[list[float]]) -> None:
    """Upsert chunks with their dense and sparse vectors into Qdrant.

    The sparse vector is written unconditionally, regardless of BM25_ENABLED:
    that flag is query-side only now. Gating the write on it is what let an
    index and a collection drift apart silently.
    """
    client = get_qdrant_client()
    points = [
        PointStruct(
            id=chunk.chunk_id,
            # "" is the unnamed dense vector; "bm25" is the named sparse one.
            vector={"": embedding, SPARSE_VECTOR_NAME: bm25_document(chunk.text)},
            payload=chunk.model_dump(),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + batch_size],
        )
```

- [ ] **Step 4: Delete both BM25 blocks from the ingest pipeline**

In `ingest/pipeline.py`, delete lines 21-27 (the `if settings.bm25_enabled:` block calling `remove_document_from_bm25`) and lines 48-53 (the block calling `add_chunks_to_bm25`), leaving:

```python
    doc_id = chunks[0].doc_id

    delete_document(doc_id)
```

and ending the function at:

```python
    embeddings = embed_texts(embed_inputs)
    upsert_chunks(chunks, embeddings)

    return len(chunks)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_sparse_ingest.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add rag/vector_store.py ingest/pipeline.py tests/unit/test_sparse_ingest.py
git commit -m "feat(ingest): write sparse vectors unconditionally, drop the file index sync"
```

---

### Task 4: Fused retrieval via the Query API

**Files:**
- Modify: `rag/vector_store.py:145-171` (`search_vectors` → `search_chunks`)
- Modify: `rag/tools/search_policies.py:74` (call site, non-BM25 branch only)
- Modify: `scripts/run_eval.py:146` (call site)
- Modify: `eval/run_experiment.py:88` (call site)
- Modify: `tests/unit/test_retrieval_spans.py` (rename call sites)
- Test: `tests/unit/test_fused_retrieval.py` (create)

**Interfaces:**
- Consumes: `bm25_document`, `SPARSE_VECTOR_NAME`.
- Produces: `search_chunks(query_text: str, query_vector: list[float], top_k: int | None = None) -> list` returning `list[ScoredPoint]` in **both** modes. `search_vectors` no longer exists. Tasks 5 and 8 consume this.

This task renames and rewires only. Callers keep their `if settings.bm25_enabled:` branching; Tasks 5 and 8 remove it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_fused_retrieval.py`:

```python
"""Retrieval issues ONE Qdrant call in both modes.

Fusion moved server-side, so both branches return list[ScoredPoint] with the
same payload shape. That is what lets search_policies, run_eval and
run_experiment each collapse from two code paths to one.

Every test here asserts on the REQUEST Qdrant receives. Nothing reaches the
network.
"""

from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

import rag.vector_store as vs


@pytest.fixture
def span_exporter(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(vs, "get_tracer", lambda: tracer)
    return exporter


class _Point:
    def __init__(self, score, payload=None, point_id="c1"):
        self.score = score
        self.payload = payload or {}
        self.id = point_id


class _Response:
    def __init__(self, points):
        self.points = points


class _FakeClient:
    def __init__(self, points=()):
        self._points = list(points)
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self._points)


@pytest.fixture
def client(monkeypatch):
    fake = _FakeClient([_Point(0.91), _Point(0.42)])
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(vs.settings, "qdrant_collection", "compliance_policies_v2")
    return fake


def test_bm25_on_sends_two_prefetch_branches_fused_by_rrf(monkeypatch, client, span_exporter):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs.settings, "hybrid_vector_candidates", 20)
    monkeypatch.setattr(vs.settings, "hybrid_bm25_candidates", 15)

    vs.search_chunks("retention period", [0.1, 0.2], top_k=6)

    call = client.calls[0]
    dense, sparse = call["prefetch"]
    assert dense.query == [0.1, 0.2]
    assert dense.limit == 20
    assert sparse.using == "bm25"
    assert sparse.limit == 15
    assert sparse.query.model == "qdrant/bm25"
    assert sparse.query.text == "retention period"
    assert call["query"].fusion == "rrf"
    assert call["limit"] == 6


def test_bm25_off_sends_a_plain_dense_query_with_no_fusion(monkeypatch, client):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)

    vs.search_chunks("retention period", [0.1, 0.2], top_k=6)

    call = client.calls[0]
    assert "prefetch" not in call or call["prefetch"] is None
    assert call["query"] == [0.1, 0.2]


def test_both_modes_return_scored_points_unchanged(monkeypatch, client):
    """The whole point of server-side fusion: one return type, no caller branch."""
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    fused = vs.search_chunks("q", [0.1], top_k=2)
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)
    dense = vs.search_chunks("q", [0.1], top_k=2)

    assert [p.score for p in fused] == [p.score for p in dense] == [0.91, 0.42]


def test_avg_len_is_threaded_into_the_query_document(monkeypatch, client):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs.settings, "bm25_avg_len", 77.0)

    vs.search_chunks("q", [0.1], top_k=6)

    assert client.calls[0]["prefetch"][1].query.options == {"avg_len": 77.0}


def test_span_keeps_its_historical_name_and_gains_fusion_attributes(
    monkeypatch, client, span_exporter
):
    """Renaming the span would silently invalidate every Phoenix comparison
    against runs recorded before this migration."""
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)

    vs.search_chunks("q", [0.1], top_k=6)

    span = span_exporter.get_finished_spans()[0]
    assert span.name == "search_vectors"
    assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "RETRIEVER"
    assert span.attributes["qdrant.collection"] == "compliance_policies_v2"
    assert span.attributes["qdrant.bm25_enabled"] is True
    assert span.attributes["qdrant.fusion"] == "rrf"
    assert span.attributes["qdrant.returned_count"] == 2
    assert span.attributes["qdrant.top_score"] == 0.91


def test_default_limit_falls_back_to_retrieval_top_k(monkeypatch, client, span_exporter):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)
    monkeypatch.setattr(vs.settings, "retrieval_top_k", 10)

    vs.search_chunks("q", [0.1])

    assert client.calls[0]["limit"] == 10
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_fused_retrieval.py -v`
Expected: FAIL — `AttributeError: module 'rag.vector_store' has no attribute 'search_chunks'`.

- [ ] **Step 3: Implement `search_chunks`**

In `rag/vector_store.py`, add `Fusion`, `FusionQuery` and `Prefetch` to the `qdrant_client.models` import, and replace `search_vectors` entirely:

```python
def search_chunks(
    query_text: str, query_vector: list[float], top_k: int | None = None
) -> list:
    """Retrieve candidate chunks: dense only, or dense + sparse fused by Qdrant.

    Returns list[ScoredPoint] in BOTH modes, because fusion happens server-side —
    so no caller needs a branch. The score means different things (an RRF score
    when bm25_enabled, a cosine similarity otherwise); callers record which in
    `score_type`, and anything comparing a score against a threshold must check
    which scale it is on. See search_policies' min_confidence_score guard.
    """
    tracer = get_tracer()
    limit = top_k or settings.retrieval_top_k
    client = get_qdrant_client()
    # The span name stays "search_vectors" although the function was renamed, so
    # Phoenix comparisons against runs recorded before this migration stay valid.
    with tracer.start_as_current_span(
        "search_vectors",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
            "qdrant.collection": settings.qdrant_collection,
            "qdrant.limit": limit,
            "qdrant.bm25_enabled": settings.bm25_enabled,
        },
    ) as span:
        if settings.bm25_enabled:
            span.set_attribute("qdrant.fusion", "rrf")
            span.set_attribute("qdrant.bm25_avg_len", settings.bm25_avg_len)
            response = client.query_points(
                collection_name=settings.qdrant_collection,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        limit=settings.hybrid_vector_candidates,
                    ),
                    Prefetch(
                        query=bm25_document(query_text),
                        using=SPARSE_VECTOR_NAME,
                        limit=settings.hybrid_bm25_candidates,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            response = client.query_points(
                collection_name=settings.qdrant_collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
        points = response.points
        span.set_attribute("qdrant.returned_count", len(points))
        if points:
            span.set_attribute("qdrant.top_score", points[0].score)
        span.set_attributes(_document_span_attributes(points))
        return points
```

- [ ] **Step 4: Update the three call sites**

`rag/tools/search_policies.py` — in the `else` (non-BM25) branch, change the import and the call:

```python
        from rag.vector_store import search_chunks
```
```python
            raw = retry_transient(lambda: search_chunks(query, query_vector, top_k=retrieve_k))
```

`scripts/run_eval.py` — change the module-level import in `run_retrieval_eval` from `search_vectors` to `search_chunks`, and the call:

```python
                raw_results = search_chunks(tc["question"], vector, top_k=top_k)
```

`eval/run_experiment.py` — change the import from `search_vectors` to `search_chunks`, and the call:

```python
            raw = search_chunks(input["question"], vector, top_k=retrieve_k)
```

- [ ] **Step 5: Update the existing span tests**

In `tests/unit/test_retrieval_spans.py`, replace every `vector_store_mod.search_vectors(` with `vector_store_mod.search_chunks(` and give each a query-text first argument. There are two direct call sites:

```python
    result = vector_store_mod.search_chunks("what is the retention period?", [0.1, 0.2, 0.3], top_k=5)
```
```python
    result = vector_store_mod.search_chunks("what is the retention period?", [0.1, 0.2, 0.3])
```

Also update the `monkeypatch.setattr("rag.vector_store.search_vectors", ...)` in `tests/unit/test_search_floor.py` to:

```python
    monkeypatch.setattr("rag.vector_store.search_chunks", lambda q, v, top_k: [_Hit()])
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_fused_retrieval.py tests/unit/test_retrieval_spans.py tests/unit/test_search_floor.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add rag/vector_store.py rag/tools/search_policies.py scripts/run_eval.py eval/run_experiment.py tests/
git commit -m "feat(retrieval): fuse dense and sparse server-side via the Query API"
```

---

### Task 5: Collapse `search_policies` to one path, and fix the cosine guard

**Files:**
- Modify: `rag/tools/search_policies.py:29-104` (Step 1 of the function)
- Test: `tests/unit/test_search_one_path.py` (create)

**Interfaces:**
- Consumes: `search_chunks(query_text, query_vector, top_k)` from Task 4.
- Produces: nothing new. `search_policies(query, top_k)` keeps its signature and its return contract (`NO_MATCH` / `UNAVAILABLE` / formatted sources).

**This task carries the sharpest correctness risk in the plan.** `min_confidence_score` is `0.45` and is a **cosine** threshold. Today the BM25 branch returns before reaching it. Once both modes share one path, a guard on `not reranker_enabled` alone would compare `0.45` against an RRF score of roughly `0.016` and return `NO_RELEVANT_POLICY_FOUND` for every question — escalating the entire corpus with no error anywhere.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_search_one_path.py`:

```python
"""search_policies runs one retrieval path for both modes.

The guard test below is the important one. min_confidence_score is a COSINE
threshold of 0.45. An RRF score is around 0.016. Guarding only on
`not reranker_enabled` — which is what the code said before the branches merged —
would compare the two and escalate every question in the corpus, silently.
"""

import pytest

import rag.tools.search_policies as sp


class _Hit:
    def __init__(self, score, doc_title="Backup Policy [Internal]"):
        self.score = score
        self.id = "c1"
        self.payload = {
            "doc_title": doc_title,
            "doc_id": "backup-policy-internal",
            "section": "Data Retention",
            "clause": "Retention Period",
            "clause_number": "4.7",
            "text": "Backups are retained for ninety days.",
        }


@pytest.fixture
def retrieval(monkeypatch):
    """embed_query and search_chunks mocked; nothing touches the network.

    Patched on `sp`, NOT on rag.embeddings / rag.vector_store: search_policies
    binds both names at module import (imports-at-top), so patching the source
    module would leave the already-bound name untouched and the test would hit
    the real network. This is the CLAUDE.md stale-import gotcha.
    """
    monkeypatch.setattr(sp, "embed_query", lambda q: [0.0] * 768)
    monkeypatch.setattr(sp.settings, "reranker_enabled", False)
    monkeypatch.setattr(sp.settings, "min_confidence_score", 0.45)

    def _install(points):
        monkeypatch.setattr(sp, "search_chunks", lambda q, v, top_k: points)

    return _install


def test_an_rrf_score_is_not_measured_against_the_cosine_floor(monkeypatch, retrieval):
    """The regression guard. 0.016 is a perfectly normal RRF score."""
    monkeypatch.setattr(sp.settings, "bm25_enabled", True)
    retrieval([_Hit(0.016)])

    result = sp.search_policies("how long are backups kept?")

    assert result != sp.NO_MATCH
    assert "Backup Policy [Internal]" in result


def test_the_cosine_floor_still_applies_on_the_plain_dense_path(monkeypatch, retrieval):
    """Reranker off AND bm25 off is the only configuration where the score is
    actually a cosine similarity, so it is the only one the floor may judge."""
    monkeypatch.setattr(sp.settings, "bm25_enabled", False)
    retrieval([_Hit(0.20)])

    assert sp.search_policies("how long are backups kept?") == sp.NO_MATCH


def test_the_cosine_floor_passes_a_high_enough_dense_score(monkeypatch, retrieval):
    monkeypatch.setattr(sp.settings, "bm25_enabled", False)
    retrieval([_Hit(0.80)])

    assert sp.search_policies("how long are backups kept?") != sp.NO_MATCH


def test_both_modes_format_sources_identically(monkeypatch, retrieval):
    """One code path means one output shape — no drift between the two modes."""
    retrieval([_Hit(0.90)])

    monkeypatch.setattr(sp.settings, "bm25_enabled", True)
    fused = sp.search_policies("q")
    monkeypatch.setattr(sp.settings, "bm25_enabled", False)
    dense = sp.search_policies("q")

    assert fused == dense
    assert "[Source 1] Backup Policy [Internal]" in fused


def test_the_retrieval_score_is_captured_for_eval_logging(monkeypatch, retrieval):
    monkeypatch.setattr(sp.settings, "bm25_enabled", True)
    retrieval([_Hit(0.016)])

    sp.search_policies("q")

    assert sp._last_search_results[0]["retrieval_score"] == 0.016


def test_empty_results_still_report_no_match(monkeypatch, retrieval):
    monkeypatch.setattr(sp.settings, "bm25_enabled", True)
    retrieval([])

    assert sp.search_policies("q") == sp.NO_MATCH
    assert sp._last_search_results == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_search_one_path.py -v`
Expected: FAIL — with `bm25_enabled=True` the old branch calls `hybrid_search`, which is not mocked, so the first test errors rather than returning sources.

- [ ] **Step 3: Rewrite Step 1 of `search_policies`**

In `rag/tools/search_policies.py`, move the imports to the top of the module (per the project's imports-at-top rule — the function-local ones existed only to keep the two branches apart):

```python
import asyncio

from llama_index.core.tools import FunctionTool

from config import settings
from rag.embeddings import embed_query
from rag.observability import record_floor_rejection, record_infra_unavailable
from rag.reranker import rerank
from rag.resilience import RETRY_BACKOFFS, is_transient, retry_transient
from rag.vector_store import search_chunks
```

Then replace everything from `retrieve_k = ...` down to the end of the `results = [...]` assignment with the single path:

```python
    # How many candidates to retrieve (more when the reranker will rescore)
    retrieve_k = settings.reranker_candidates if settings.reranker_enabled else top_k

    # Step 1: Retrieve candidates — dense only, or dense + sparse fused by Qdrant.
    # One path for both: fusion is server-side, so both return ScoredPoints.
    try:
        query_vector = retry_transient(lambda: embed_query(query))
    except Exception as exc:
        if is_transient(exc):
            _last_search_results = []
            _retrieval_unavailable = True
            record_infra_unavailable("embeddings", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    try:
        raw = retry_transient(lambda: search_chunks(query, query_vector, top_k=retrieve_k))
    except Exception as exc:
        if is_transient(exc):
            _last_search_results = []
            _retrieval_unavailable = True
            record_infra_unavailable("qdrant", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    if not raw:
        _last_search_results = []
        return NO_MATCH

    # min_confidence_score is a COSINE threshold (0.45). It may only judge a score
    # that IS a cosine similarity — reranker off AND no RRF fusion. An RRF score is
    # ~0.016, so dropping the bm25 half of this condition would return NO_MATCH for
    # every question in the corpus, with no error raised anywhere. Same class of bug
    # as the rerank_score-presence guard in Step 3b: the guard is on what the number
    # MEANS, not on which component produced it.
    if (
        not settings.reranker_enabled
        and not settings.bm25_enabled
        and raw[0].score < settings.min_confidence_score
    ):
        _last_search_results = []
        return NO_MATCH

    results = [
        {
            "doc_title": r.payload["doc_title"],
            "doc_id": r.payload["doc_id"],
            "section": r.payload.get("section", ""),
            "clause": r.payload.get("clause", ""),
            "clause_number": r.payload.get("clause_number", ""),
            "text": r.payload["text"],
            "retrieval_score": r.score,
            "score_type": "rrf" if settings.bm25_enabled else "cosine",
        }
        for r in raw
    ]
```

Steps 2, 3, 3b and 4 of the function (rerank, capture, floor, format) are unchanged. Delete the now-unused `from rag.reranker import rerank` line inside Step 2, since it moved to the top.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_search_one_path.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Re-point the floor test's patches**

Task 4 left `tests/unit/test_search_floor.py` patching the source modules. That
worked while `search_policies` imported them inside the function; now that the
imports are at module top, the bound names must be patched on `sp` instead — a
patch on the source module would silently miss and let the test hit the network:

```python
    monkeypatch.setattr(sp, "embed_query", lambda q: [0.0] * 768)
    monkeypatch.setattr(sp, "search_chunks", lambda q, v, top_k: [_Hit()])
```

- [ ] **Step 6: Run the retrieval and floor suites**

Run: `python -m pytest tests/unit/test_search_floor.py tests/unit/test_retrieval_spans.py tests/unit/test_request_span.py -v`
Expected: PASS — these cover the resilience, span-nesting and floor behaviour that must survive the collapse.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add rag/tools/search_policies.py tests/unit/test_search_floor.py tests/unit/test_search_one_path.py
git commit -m "refactor(search): one retrieval path, and guard the cosine floor on scale"
```

---

### Task 6: Startup preflight

**Files:**
- Modify: `rag/vector_store.py` (add `preflight_sparse_config`)
- Modify: `scripts/start_teams_bot.py:20-26`
- Test: `tests/unit/test_sparse_preflight.py` (create)

**Interfaces:**
- Consumes: `SPARSE_VECTOR_NAME` from Task 2.
- Produces: `preflight_sparse_config() -> None`, raising `RuntimeError` on a mismatch.

Without this, running `BM25_ENABLED=true` against a collection that has no sparse vector makes Qdrant return `Not existing vector name error` — which `is_transient` classifies as non-transient, so it becomes a **content escalation on every question**. The bot would look healthy while finding nothing.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sparse_preflight.py`:

```python
"""Fail fast when BM25 is on but the collection has no sparse vector.

Qdrant answers that query with "Not existing vector name error", which is not
transient — so without this check every question comes back as a content
escalation and the bot looks healthy while finding nothing. It raises rather
than auto-disabling BM25 on purpose: quiet degradation is the failure class
this migration exists to remove.
"""

import pytest

import rag.vector_store as vs


class _Params:
    def __init__(self, sparse):
        self.sparse_vectors = sparse


class _Config:
    def __init__(self, sparse):
        self.params = _Params(sparse)


class _Info:
    def __init__(self, sparse):
        self.config = _Config(sparse)


class _FakeClient:
    def __init__(self, sparse):
        self._sparse = sparse
        self.calls = 0

    def get_collection(self, name):
        self.calls += 1
        return _Info(self._sparse)


def test_preflight_passes_when_the_sparse_vector_exists(monkeypatch):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient({"bm25": object()}))

    vs.preflight_sparse_config()  # must not raise


def test_preflight_refuses_a_collection_without_the_sparse_vector(monkeypatch):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs.settings, "qdrant_collection", "compliance_policies")
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient({}))

    with pytest.raises(RuntimeError) as exc:
        vs.preflight_sparse_config()

    message = str(exc.value)
    assert "compliance_policies" in message
    assert "bm25" in message
    assert "migrate_collection.py" in message  # tells the operator what to do


def test_preflight_refuses_when_sparse_vectors_is_none(monkeypatch):
    """A collection created before this migration reports None, not {}."""
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient(None))

    with pytest.raises(RuntimeError):
        vs.preflight_sparse_config()


def test_preflight_does_not_call_qdrant_when_bm25_is_disabled(monkeypatch):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)
    client = _FakeClient({})
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.preflight_sparse_config()

    assert client.calls == 0


def test_the_bot_entry_point_runs_the_preflight():
    from pathlib import Path

    source = Path("scripts/start_teams_bot.py").read_text(encoding="utf-8")
    assert "preflight_sparse_config()" in source
    # It must run before the bot is constructed, not after.
    assert source.index("preflight_sparse_config()") < source.index("TeamsBot(")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_sparse_preflight.py -v`
Expected: FAIL — `AttributeError: module 'rag.vector_store' has no attribute 'preflight_sparse_config'`.

- [ ] **Step 3: Implement the preflight**

In `rag/vector_store.py`, after `init_collection`:

```python
def preflight_sparse_config() -> None:
    """Refuse to start if BM25 is on but the collection has no sparse vector.

    Qdrant answers a sparse query against a collection lacking that vector with
    "Not existing vector name error". That is NOT transient, so without this
    check it would surface as a content escalation on every single question --
    the bot would appear healthy while finding nothing.

    Deliberately raises instead of auto-disabling BM25. A bot that quietly drops
    to dense-only looks fine and answers worse, which is exactly the failure mode
    this migration exists to remove.
    """
    if not settings.bm25_enabled:
        return
    client = get_qdrant_client()
    info = client.get_collection(settings.qdrant_collection)
    sparse = info.config.params.sparse_vectors or {}
    if SPARSE_VECTOR_NAME not in sparse:
        raise RuntimeError(
            f"BM25_ENABLED=true but collection '{settings.qdrant_collection}' has no "
            f"'{SPARSE_VECTOR_NAME}' sparse vector (found: {sorted(sparse) or 'none'}). "
            "Build a sparse-enabled collection with "
            "`PYTHONPATH=. python scripts/migrate_collection.py --target <name>`, "
            "or set BM25_ENABLED=false."
        )
```

- [ ] **Step 4: Call it from the bot entry point**

In `scripts/start_teams_bot.py`, after `init_observability()` and before the bot is built:

```python
from rag.observability import init_observability

init_observability()

from channels.teams.auth import TokenRefresher
from channels.teams.bot import TeamsBot
from rag.vector_store import preflight_sparse_config

if __name__ == "__main__":
    # Fails loudly here rather than turning every question into an escalation.
    preflight_sparse_config()
    token_refresher = TokenRefresher()
    bot = TeamsBot(token_refresher)
    bot.run()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_sparse_preflight.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add rag/vector_store.py scripts/start_teams_bot.py tests/unit/test_sparse_preflight.py
git commit -m "feat(qdrant): refuse to start with BM25 on and no sparse vector"
```

---

### Task 7: Migration script

**Files:**
- Create: `scripts/migrate_collection.py`
- Test: `tests/unit/test_migrate_collection.py` (create)

**Interfaces:**
- Consumes: `init_collection(collection_name)`, `bm25_document(text)`, `SPARSE_VECTOR_NAME`, `get_qdrant_client()`.
- Produces: `migrate(source, target, dry_run) -> int` (points copied) and `verify(source, target, sample) -> bool`.

Copying points rather than re-parsing the `.docx` corpus keeps the dense vectors byte-identical, so any eval delta is attributable to the sparse half alone — which is what makes the gate in the rollout meaningful.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_migrate_collection.py`:

```python
"""Point-copying migration into a sparse-enabled collection.

Copying rather than re-ingesting keeps the dense vectors byte-identical, which
is what makes the eval gate meaningful: any score delta is attributable to the
sparse half alone. Nothing here touches the network.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
migrate_mod = importlib.import_module("scripts.migrate_collection")


class _SrcPoint:
    def __init__(self, pid, vector, payload):
        self.id = pid
        self.vector = vector
        self.payload = payload


class _Count:
    def __init__(self, count):
        self.count = count


class _FakeClient:
    def __init__(self, points, exists=False):
        self._points = list(points)
        self._exists = exists
        self.upserted = []
        self.created = []

    def count(self, collection_name, exact=True):
        return _Count(len(self._points))

    def collection_exists(self, name):
        return self._exists

    def scroll(self, collection_name, limit=500, offset=None, **kwargs):
        return list(self._points), None

    def upsert(self, collection_name, points, wait=True):
        self.upserted.append((collection_name, points))


@pytest.fixture
def client(monkeypatch):
    fake = _FakeClient([
        _SrcPoint("c1", [0.1, 0.2], {"text": "backups are kept ninety days"}),
        _SrcPoint("c2", [0.3, 0.4], {"text": "laptops must be encrypted"}),
    ])
    monkeypatch.setattr(migrate_mod, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(migrate_mod, "init_collection", lambda name: fake.created.append(name))
    return fake


def test_dry_run_writes_nothing(client, capsys):
    copied = migrate_mod.migrate("src", "dst", dry_run=True)

    assert copied == 0
    assert client.upserted == []
    assert client.created == []
    assert "2 points" in capsys.readouterr().out


def test_migration_preserves_id_payload_and_dense_vector(client):
    migrate_mod.migrate("src", "dst", dry_run=False)

    _, points = client.upserted[0]
    first = points[0]
    assert first.id == "c1"
    assert first.payload == {"text": "backups are kept ninety days"}
    assert first.vector[""] == [0.1, 0.2]


def test_migration_adds_a_sparse_document_from_the_payload_text(client):
    migrate_mod.migrate("src", "dst", dry_run=False)

    _, points = client.upserted[0]
    sparse = points[0].vector["bm25"]
    assert sparse.model == "qdrant/bm25"
    assert sparse.text == "backups are kept ninety days"


def test_migration_creates_the_target_from_the_shared_schema(client):
    """init_collection is the single schema definition; duplicating it here is how
    the two would drift."""
    migrate_mod.migrate("src", "dst", dry_run=False)

    assert client.created == ["dst"]


def test_a_named_dense_vector_round_trips(monkeypatch):
    """Re-running against an already-migrated collection: qdrant-client returns a
    dict there, not a bare list."""
    fake = _FakeClient([_SrcPoint("c1", {"": [0.5], "bm25": object()}, {"text": "t"})])
    monkeypatch.setattr(migrate_mod, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(migrate_mod, "init_collection", lambda name: None)

    migrate_mod.migrate("src", "dst", dry_run=False)

    assert fake.upserted[0][1][0].vector[""] == [0.5]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_migrate_collection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.migrate_collection'`.

- [ ] **Step 3: Write the script**

Create `scripts/migrate_collection.py`:

```python
#!/usr/bin/env python
"""Copy a collection's points into a new, sparse-enabled collection.

Why copy instead of re-ingesting the .docx corpus: this reuses the dense vectors
exactly as they are, so the dense half of retrieval is provably unchanged and any
eval delta is attributable to the sparse half alone. It also needs no Ollama call
and no access to the source documents.

Usage:
    PYTHONPATH=. python scripts/migrate_collection.py --target compliance_policies_v2 --dry-run
    PYTHONPATH=. python scripts/migrate_collection.py --target compliance_policies_v2
    PYTHONPATH=. python scripts/migrate_collection.py --target compliance_policies_v2 --verify
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()  # Must precede any LlamaIndex/Ollama import

from qdrant_client.models import PointStruct

from config import settings
from rag.vector_store import (
    SPARSE_VECTOR_NAME,
    bm25_document,
    get_qdrant_client,
    init_collection,
)

SCROLL_PAGE = 500
UPSERT_BATCH = 100
VERIFY_SAMPLE = 25


def _dense_of(point) -> list[float]:
    """The source point's dense vector.

    qdrant-client returns a bare list for an unnamed vector and a dict keyed by
    name when the collection has named ones. Accept both, so re-running against
    an already-migrated collection works instead of writing a dict as a vector.
    """
    vector = point.vector
    return vector[""] if isinstance(vector, dict) else vector


def migrate(source: str, target: str, dry_run: bool) -> int:
    """Copy every point from `source` into `target`, adding a sparse vector."""
    client = get_qdrant_client()
    total = client.count(collection_name=source, exact=True).count

    if dry_run:
        print(f"[dry-run] source {source}: {total} points")
        print(f"[dry-run] target {target} exists: {client.collection_exists(target)}")
        print(f"[dry-run] would create {target} and copy {total} points")
        return 0

    init_collection(target)

    copied = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=source,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        batch = [
            PointStruct(
                id=p.id,
                vector={
                    "": _dense_of(p),
                    SPARSE_VECTOR_NAME: bm25_document(p.payload.get("text", "")),
                },
                payload=p.payload,
            )
            for p in points
        ]
        for i in range(0, len(batch), UPSERT_BATCH):
            client.upsert(
                collection_name=target, points=batch[i : i + UPSERT_BATCH], wait=True
            )
        copied += len(batch)
        print(f"  copied {copied}/{total}")

        if offset is None:
            break

    return copied


def verify(source: str, target: str) -> bool:
    """Check count parity, the sparse schema, and a sample of ids."""
    client = get_qdrant_client()

    src = client.count(collection_name=source, exact=True).count
    tgt = client.count(collection_name=target, exact=True).count
    counts_ok = src == tgt
    print(f"count:        source={src} target={tgt} {'OK' if counts_ok else 'MISMATCH'}")

    sparse = client.get_collection(target).config.params.sparse_vectors or {}
    schema_ok = SPARSE_VECTOR_NAME in sparse
    print(f"sparse config: {sorted(sparse) or 'none'} {'OK' if schema_ok else 'MISSING'}")

    sample, _ = client.scroll(
        collection_name=source, limit=VERIFY_SAMPLE, with_payload=False, with_vectors=False
    )
    ids = [p.id for p in sample]
    found = client.retrieve(collection_name=target, ids=ids, with_vectors=True)
    present = sum(
        1
        for p in found
        if isinstance(p.vector, dict) and p.vector.get(SPARSE_VECTOR_NAME)
    )
    sample_ok = present == len(ids)
    print(f"sampled ids:   {present}/{len(ids)} present with a non-empty sparse vector")

    return counts_ok and schema_ok and sample_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy points into a sparse-enabled Qdrant collection",
    )
    parser.add_argument("--source", default=settings.qdrant_collection)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="check an existing target")
    parser.add_argument(
        "--force", action="store_true", help="write into a non-empty target"
    )
    args = parser.parse_args()

    if args.source == args.target:
        print("ERROR: --source and --target must differ")
        return 1

    client = get_qdrant_client()

    if args.verify:
        return 0 if verify(args.source, args.target) else 1

    if (
        not args.dry_run
        and not args.force
        and client.collection_exists(args.target)
        and client.count(collection_name=args.target, exact=True).count
    ):
        print(f"ERROR: {args.target} already has points. Use --force to write into it.")
        return 1

    copied = migrate(args.source, args.target, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nCopied {copied} points into {args.target}")
        print(f"Verify with: --target {args.target} --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_migrate_collection.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/migrate_collection.py tests/unit/test_migrate_collection.py
git commit -m "feat(scripts): copy a collection into a sparse-enabled one"
```

---

### Task 8: Collapse the two eval call sites

**Files:**
- Modify: `eval/run_experiment.py:28-115` (`make_tier1_task`) and `:370-378` (`infra_meta`)
- Modify: `scripts/run_eval.py:103-165` (`run_retrieval_eval`)
- Test: `tests/unit/test_eval_metadata.py` (extend)

**Interfaces:**
- Consumes: `search_chunks(query_text, query_vector, top_k)` from Task 4.
- Produces: nothing new. This is the last task that imports `rag.hybrid_search`; Task 9 depends on it.

Both files keep their function-local imports — that pattern is deliberate in entry-point scripts, where `init_observability()` must run before any LlamaIndex or Ollama import.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_eval_metadata.py`:

```python
def test_experiment_metadata_identifies_the_collection_and_bm25_parameters():
    """An experiment whose retrieval parameters are only recoverable from its NAME
    cannot be compared against another six weeks later. After this migration the
    collection is a parameter too: v1 and v2 hold different indexes."""
    source = Path("eval/run_experiment.py").read_text(encoding="utf-8")

    assert '"qdrant_collection": settings.qdrant_collection' in source
    assert '"bm25_enabled": settings.bm25_enabled' in source
    assert '"bm25_avg_len": settings.bm25_avg_len' in source


def test_no_eval_entry_point_still_imports_the_deleted_hybrid_module():
    for path in ("eval/run_experiment.py", "scripts/run_eval.py"):
        source = Path(path).read_text(encoding="utf-8")
        assert "hybrid_search" not in source, f"{path} still imports hybrid_search"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_eval_metadata.py -v`
Expected: FAIL — the metadata keys are absent and both files still import `hybrid_search`.

- [ ] **Step 3: Collapse `make_tier1_task`**

In `eval/run_experiment.py`, replace the whole body of `make_tier1_task` (the `_to_result_dicts` helper and both `retrieval_task` definitions) with:

```python
def make_tier1_task(top_k: int):
    # Local imports: init_observability() must run before any LlamaIndex/Ollama
    # import, and this module is imported by the CLI entry point below.
    from config import settings
    from rag.embeddings import embed_query
    from rag.vector_store import search_chunks

    retrieve_k = settings.reranker_candidates if settings.reranker_enabled else top_k

    def retrieval_task(input):
        # One path for both modes: Qdrant fuses server-side, so dense-only and
        # dense+sparse both come back as ScoredPoints with the same payload.
        vector = embed_query(input["question"])
        raw = search_chunks(input["question"], vector, top_k=retrieve_k)
        results = [
            {
                "doc_title": r.payload.get("doc_title", ""),
                "section": r.payload.get("section", ""),
                "clause": r.payload.get("clause", ""),
                "clause_number": r.payload.get("clause_number", ""),
                "text": r.payload.get("text", ""),
                "retrieval_score": round(r.score, 4),
            }
            for r in raw
        ]

        if settings.reranker_enabled and results:
            from rag.reranker import rerank

            results = rerank(input["question"], results, top_n=settings.reranker_top_n)

        return {
            "search_results": [
                {
                    "doc_title": r["doc_title"],
                    "section": r["section"],
                    "clause": r.get("clause", ""),
                    "clause_number": r.get("clause_number", ""),
                    "retrieval_score": r.get("retrieval_score", 0),
                    "rerank_score": r.get("rerank_score"),
                    "original_rank": r.get("original_rank"),
                }
                for r in results
            ]
        }

    return retrieval_task
```

- [ ] **Step 4: Record the new metadata**

In `eval/run_experiment.py`, extend `infra_meta` (shared by every tier) with three keys:

```python
    infra_meta = {
        "infra": infra,
        "llm_backend": settings.llm_backend,
        "llm_url": settings.active_ollama_url,
        "embedding_source": settings.embedding_source,
        "embedding_url": settings.ollama_embedding_url if settings.embedding_source == "ollama" else "local",
        "qdrant_url": settings.active_qdrant_url,
        # The collection is a retrieval parameter now: v1 and v2 hold different
        # indexes, so a run that does not name it cannot be compared later.
        "qdrant_collection": settings.qdrant_collection,
        "bm25_enabled": settings.bm25_enabled,
        "bm25_avg_len": settings.bm25_avg_len,
        "reranker_backend": settings.reranker_backend if settings.reranker_enabled else "none",
        "reranker_url": settings.reranker_url if settings.reranker_enabled else "none",
    }
```

- [ ] **Step 5: Collapse `run_retrieval_eval`**

In `scripts/run_eval.py`, replace the imports at the top of `run_retrieval_eval`:

```python
def run_retrieval_eval(dataset_path: Path, tag: str) -> dict:
    from rag.embeddings import embed_query
    from rag.vector_store import search_chunks
```

and replace the whole `if settings.bm25_enabled: ... else: ...` block inside the loop with:

```python
            # One path for both modes: Qdrant fuses server-side.
            vector = embed_query(tc["question"])
            raw_results = search_chunks(tc["question"], vector, top_k=top_k)
            search_results = [
                {
                    "doc_id": r.payload.get("doc_id", ""),
                    "doc_title": r.payload.get("doc_title", ""),
                    "section": r.payload.get("section", ""),
                    "section_number": r.payload.get("section_number", ""),
                    "clause": r.payload.get("clause", ""),
                    "clause_number": r.payload.get("clause_number", ""),
                    "section_display": r.payload.get("section_display", ""),
                    "text": r.payload.get("text", ""),
                    "score": r.score,
                }
                for r in raw_results
            ]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_eval_metadata.py tests/unit/test_run_experiment.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add eval/run_experiment.py scripts/run_eval.py tests/unit/test_eval_metadata.py
git commit -m "refactor(eval): one retrieval path, and record the collection in metadata"
```

---

### Task 9: Delete the file-based BM25, and update the docs

**Files:**
- Delete: `rag/bm25_index.py`, `rag/hybrid_search.py`, `scripts/build_bm25_from_qdrant.py`
- Modify: `.gitignore:37`, `.env.example:30-33`, `CLAUDE.md`, `rag/observability.py:80`
- Test: `tests/unit/test_no_file_based_bm25.py` (create)

**Interfaces:**
- Consumes: Tasks 5 and 8 removed the last importers. Nothing else.
- Produces: nothing.

- [ ] **Step 1: Confirm nothing still imports the modules**

Run: `grep -rn --include='*.py' "hybrid_search\|bm25_index\|search_bm25" . | grep -v "^./rag/bm25_index.py:\|^./rag/hybrid_search.py:\|^./scripts/build_bm25_from_qdrant.py:"`
Expected: only the docstring example in `rag/observability.py:80`. If anything else appears, stop — an earlier task is incomplete.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_no_file_based_bm25.py`:

```python
"""The file-based BM25 index is gone, not merely unused.

It was a gitignored JSON file, built only when BM25 was already on, absent from
the Docker image, and desynchronising invisibly because chunk_id is a fresh
uuid4 per ingest -- an index from an earlier ingestion had the right chunk COUNT
and entirely wrong ids (measured: 1602 vs 1602, 0 of 25 ids present). Leaving the
modules importable is how that comes back.
"""

from pathlib import Path


def test_the_file_based_modules_are_deleted():
    for path in (
        "rag/bm25_index.py",
        "rag/hybrid_search.py",
        "scripts/build_bm25_from_qdrant.py",
    ):
        assert not Path(path).exists(), f"{path} still exists"


def test_nothing_imports_them():
    offenders = []
    for path in Path(".").rglob("*.py"):
        if ".venv" in path.parts or path.name == "test_no_file_based_bm25.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "hybrid_search" in source or "bm25_index" in source:
            offenders.append(str(path))
    assert offenders == [], f"still referencing deleted modules: {offenders}"


def test_the_stale_gitignore_entry_is_gone():
    assert ".bm25_index.json" not in Path(".gitignore").read_text(encoding="utf-8")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_no_file_based_bm25.py -v`
Expected: FAIL — all three files still exist.

- [ ] **Step 4: Delete the modules**

```bash
git rm rag/bm25_index.py rag/hybrid_search.py scripts/build_bm25_from_qdrant.py
```

- [ ] **Step 5: Clean up the references**

In `.gitignore`, delete line 37 (`.bm25_index.json`).

In `rag/observability.py`, change the `get_tracer` docstring example from the deleted module's name:

```python
    Usage:
        tracer = get_tracer()
        with tracer.start_as_current_span("search_vectors") as span:
            span.set_attribute("query", query)
            span.set_attribute("vector_top_score", 0.87)
            # ... do work ...
```

In `.env.example`, replace the hybrid block:

```bash
# Hybrid Search (BM25 sparse vectors, stored in Qdrant)
# Query-side only: sparse vectors are always written at ingest, so flipping this
# needs no re-ingest. The collection must have a `bm25` sparse vector or the bot
# refuses to start — build one with scripts/migrate_collection.py.
BM25_ENABLED=true
HYBRID_VECTOR_CANDIDATES=20
HYBRID_BM25_CANDIDATES=20
# BM25 length normalisation. Qdrant's default is 256; this corpus measures ~50.
BM25_AVG_LEN=50.0
```

- [ ] **Step 6: Update CLAUDE.md**

Replace the `BM25_ENABLED=true silently does nothing (or hurts)` gotcha row with four rows carrying what was verified:

```
| `BM25_ENABLED=true` and the bot refuses to start | Correct behaviour. Sparse vectors live in the collection now; a collection without a `bm25` sparse vector cannot answer a sparse query (Qdrant: `Not existing vector name error`), and that error is NOT transient, so without the preflight every question would become a content escalation. Build one with `scripts/migrate_collection.py --target <name>`. |
| BM25 `k`/`b`/`avg_len` have no effect | A collection-level `Bm25Config` is **accepted and silently discarded** by Qdrant 1.17.1 — the `PUT` returns `ok` and the collection reads back with only `{"modifier": "idf"}`. They work ONLY in per-document `options`, so they must be passed on every upsert and every query (`rag/vector_store.py::bm25_document`). |
| Installing `fastembed` changes retrieval with no error | `cloud_inference=False` is the qdrant-client default and means "encode `models.Document` locally via fastembed". Server-side BM25 works here only because fastembed is absent. `get_qdrant_client()` sets `cloud_inference=True` explicitly — do not remove it, and do not add fastembed. |
| Adding a vector to an existing Qdrant collection | Not possible: `Not existing vector name error`. Any vector-schema change means a new collection plus a migration (`scripts/migrate_collection.py` copies points, keeping dense vectors byte-identical so an A/B isolates the change). Self-hosted inference also works only for built-in models — `qdrant/bm25` resolves, anything else fails with `InferenceService URL not configured`. |
```

Also update the Architecture block: `rag/` no longer lists `bm25_index.py` or `hybrid_search.py`, and the **Search flow** line becomes:

```
**Search flow:** `prefetch()` (rag/search_first.py) runs `search_policies` BEFORE the agent — `embed_query → search_chunks (one Qdrant Query API call: dense, or dense+sparse `bm25` fused server-side by RRF) → [rerank → top RERANKER_TOP_N] → relevance floor (RERANKER_MIN_SCORE, 0.0 = off) → format_sources()`
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_no_file_based_bm25.py -v`
Expected: PASS (3 passed)

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest tests/unit tests/load -q`
Expected: all pass. The count should be roughly 276 plus the new tests from Tasks 1-9.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: delete the file-based BM25 index and its rebuild tool"
```

---

## Rollout (operator, after the branch merges)

Not part of the implementation. Recorded here so the plan and the runbook do not drift apart. Nothing in this sequence changes server-side Qdrant configuration.

1. **Deploy** with `QDRANT_COLLECTION=compliance_policies`, `BM25_ENABLED=false`. Dense path unchanged; preflight passes trivially. Independently revertible.
2. **Build v2:** `PYTHONPATH=. python scripts/migrate_collection.py --target compliance_policies_v2`. The live bot still reads the old collection.
3. **Verify:** `--target compliance_policies_v2 --verify` must report 1602/1602, the sparse config present, and 25/25 sampled ids with a non-empty sparse vector.
4. **Gate:** run `chatbot-test-v1` against v2 with `BM25_ENABLED=true`. Mount `config.py`, `rag/`, `eval/` and `scripts/` together in the eval container, or it runs main's pipeline against the new harness.
5. **Cut over at `hit_evaluator` ≥ 0.9180:** set `QDRANT_COLLECTION=compliance_policies_v2` and `BM25_ENABLED=true`, redeploy. Below that, nothing has been switched — implement spec D10 (the client-side encoder) and return to step 2 with a fresh target.
6. **Keep `compliance_policies`** as the rollback. Deleting it is a separate, manual, explicitly-requested action.

Rollback after step 5: restore both `.env` values and redeploy. The old collection is untouched throughout.
