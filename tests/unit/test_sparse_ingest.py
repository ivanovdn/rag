"""Ingest writes a sparse vector alongside the dense one, unconditionally.

Unconditionally is the point. The old code built the lexical index only when
BM25 was already enabled, which is why enabling the flag later gave the bot an
empty index. Now the flag is query-side only (spec D7), so flipping it needs no
re-ingest.
"""

from pathlib import Path

import pytest

import ingest.pipeline as pipeline
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
    source = Path("ingest/pipeline.py").read_text(encoding="utf-8")
    assert "bm25" not in source.lower()


# --- the delete-then-upsert window -------------------------------------------
#
# ingest_document deletes a document's chunks and then writes them back. Since
# the sparse write became unconditional, that upsert fails against a collection
# with no sparse vector ("Wrong input: Not existing vector name error: bm25")
# AFTER the delete has committed: the policy is gone from a live index, the bot
# answers "no relevant policy found" for it, and nothing looks broken. The
# schema check therefore has to run before the delete, and ungated by
# BM25_ENABLED — the write it protects is ungated too.


class _PolicyChunk:
    """Minimal stand-in for a parsed chunk: only the fields ingest_document reads."""

    doc_id = "backup-policy-internal"
    doc_title = "Backup Policy [Internal]"
    section = "Data Retention"
    clause = "Retention Period"
    text = "Backups are retained for ninety days."


@pytest.fixture
def pipeline_calls(monkeypatch):
    """ingest_document with every collaborator recorded and nothing touching the
    network. Patched on `ingest.pipeline`, not on rag.vector_store: the pipeline
    binds these names at import (CLAUDE.md stale-import gotcha)."""
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "init_collection", lambda: calls.append("init"))
    monkeypatch.setattr(pipeline, "parse_docx", lambda path, link: [_PolicyChunk()])
    monkeypatch.setattr(pipeline, "delete_document", lambda doc_id: calls.append("delete"))
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts: [[0.1]] * len(texts))
    monkeypatch.setattr(
        pipeline, "upsert_chunks", lambda chunks, embeddings: calls.append("upsert")
    )
    return calls


def test_the_schema_check_runs_before_any_delete(pipeline_calls, monkeypatch):
    monkeypatch.setattr(
        pipeline, "assert_sparse_vector", lambda name, reason: pipeline_calls.append("assert")
    )

    pipeline.ingest_document(Path("Backup Policy [Internal].docx"), "http://x/y.docx")

    assert pipeline_calls == ["init", "assert", "delete", "upsert"]


def test_ingest_document_does_not_delete_when_the_schema_check_fails(
    pipeline_calls, monkeypatch
):
    """The one that matters. Without this ordering the delete commits and the
    upsert fails, which is indistinguishable from the policy never existing."""

    def _refuse(name, reason):
        raise RuntimeError("no sparse vector on that collection")

    monkeypatch.setattr(pipeline, "assert_sparse_vector", _refuse)

    with pytest.raises(RuntimeError):
        pipeline.ingest_document(Path("Backup Policy [Internal].docx"), "http://x/y.docx")

    assert "delete" not in pipeline_calls
    assert "upsert" not in pipeline_calls


def test_the_ingest_guard_is_not_gated_on_the_query_side_flag(pipeline_calls, monkeypatch):
    """The sparse write is unconditional, so its guard must be too."""
    checked: list[str] = []
    monkeypatch.setattr(pipeline.settings, "bm25_enabled", False)
    monkeypatch.setattr(
        pipeline, "assert_sparse_vector", lambda name, reason: checked.append(name)
    )
    # Bound to a local first: a failing assertion on `settings` itself would put
    # the whole Settings repr — real .env secrets — into the pytest output.
    collection = pipeline.settings.qdrant_collection

    pipeline.ingest_document(Path("Backup Policy [Internal].docx"), "http://x/y.docx")

    assert checked == [collection]
