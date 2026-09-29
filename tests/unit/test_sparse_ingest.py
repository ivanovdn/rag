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
