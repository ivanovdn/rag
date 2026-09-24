"""
Unit tests for the retrieval-path Phoenix/OpenTelemetry spans added to the four
leaf functions: rag.embeddings.embed_query / embed_texts, rag.vector_store.search_vectors,
and rag.reranker.rerank.

Uses an in-memory OTel span exporter — never a live Phoenix instance — and mocks all
HTTP/Qdrant calls, so nothing here touches the network or 172.20.0.22.
"""

import asyncio
import json

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from openinference.semconv.trace import DocumentAttributes, SpanAttributes
from qdrant_client.http.exceptions import ResponseHandlingException

import rag.embeddings as embeddings_mod
import rag.reranker as reranker_mod
import rag.tools.search_policies as search_policies_mod
import rag.vector_store as vector_store_mod


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def span_exporter(monkeypatch):
    """Force all three modules' get_tracer() to hand out a tracer backed by an
    in-memory exporter, regardless of settings.phoenix_enabled."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    # Each module imported `get_tracer` by name, so the module-level name (not
    # rag.observability.get_tracer) must be patched — see CLAUDE.md gotcha on
    # stale imports of module attributes.
    monkeypatch.setattr(embeddings_mod, "get_tracer", lambda: tracer)
    monkeypatch.setattr(vector_store_mod, "get_tracer", lambda: tracer)
    monkeypatch.setattr(reranker_mod, "get_tracer", lambda: tracer)
    return exporter


class _FakePoint:
    def __init__(self, score):
        self.score = score


class _FakeQueryResponse:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
    def __init__(self, points):
        self._points = points
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeQueryResponse(self._points)


class _FakeQdrantClientRaises:
    def __init__(self, exc):
        self._exc = exc

    def query_points(self, **kwargs):
        raise self._exc


class _FakePointWithPayload:
    """Like _FakePoint, but with `.id`/`.payload` so it round-trips through the
    full `search_policies()` pipeline (which reads `r.payload[...]`), not just
    the bare `search_vectors` span attributes."""

    def __init__(self, score, payload, point_id="chunk-id"):
        self.score = score
        self.payload = payload
        self.id = point_id


def _doc_attr(index: int, field: str) -> str:
    """Build a flattened retrieval.documents.{i}.document.{field} attribute key."""
    return f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}.{field}"


# --- embed_query ----------------------------------------------------------------


def test_embed_query_span_name_kind_and_attributes(monkeypatch, span_exporter):
    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(embeddings_mod.settings, "embedding_model", "embeddinggemma")
    monkeypatch.setattr(
        embeddings_mod, "_ollama_embed", lambda texts, prefix="": [[0.1, 0.2, 0.3]]
    )

    result = embeddings_mod.embed_query("what is the backup retention period?")

    assert result == [0.1, 0.2, 0.3]  # unchanged return value
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embed_query"
    assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "EMBEDDING"
    assert span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == "embeddinggemma"
    assert span.attributes["embedding.backend"] == "ollama"
    assert span.attributes["embedding.text_count"] == 1
    assert span.attributes["embedding.vector_dim"] == 3
    # Never dump the raw vector or any secret onto the span.
    assert SpanAttributes.EMBEDDING_EMBEDDINGS not in span.attributes


def test_embed_query_exception_propagates_unchanged_and_span_errors(monkeypatch, span_exporter):
    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    boom = httpx.ConnectError("connection refused")

    def raise_boom(texts, prefix=""):
        raise boom

    monkeypatch.setattr(embeddings_mod, "_ollama_embed", raise_boom)

    with pytest.raises(httpx.ConnectError) as excinfo:
        embeddings_mod.embed_query("q")
    assert excinfo.value is boom  # exact same exception instance, not re-wrapped

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


# --- embed_texts ------------------------------------------------------------------


def test_embed_texts_span_is_one_per_batch_call(monkeypatch, span_exporter):
    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(embeddings_mod.settings, "embedding_model", "embeddinggemma")
    calls = []

    def fake_ollama_embed(texts, prefix=""):
        calls.append(list(texts))
        return [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    monkeypatch.setattr(embeddings_mod, "_ollama_embed", fake_ollama_embed)

    result = embeddings_mod.embed_texts(["a", "b", "c"])

    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert calls == [["a", "b", "c"]]  # exactly one batch call, not one per text

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1  # one span per batch call, not per text
    span = spans[0]
    assert span.name == "embed_texts"
    assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "EMBEDDING"
    assert span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == "embeddinggemma"
    assert span.attributes["embedding.backend"] == "ollama"
    assert span.attributes["embedding.text_count"] == 3
    assert span.attributes["embedding.vector_dim"] == 2


def test_embed_texts_exception_propagates_unchanged_and_span_errors(monkeypatch, span_exporter):
    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    boom = httpx.ReadTimeout("timed out")

    def raise_boom(texts, prefix=""):
        raise boom

    monkeypatch.setattr(embeddings_mod, "_ollama_embed", raise_boom)

    with pytest.raises(httpx.ReadTimeout) as excinfo:
        embeddings_mod.embed_texts(["a", "b"])
    assert excinfo.value is boom

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


def test_embed_functions_never_record_hf_token(monkeypatch, span_exporter):
    # Synthetic secret — never the real settings.hf_token — proving the span
    # never picks it up even if a future embedding path were to read it.
    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(embeddings_mod.settings, "hf_token", "synthetic-test-secret")
    monkeypatch.setattr(
        embeddings_mod, "_ollama_embed", lambda texts, prefix="": [[0.1, 0.2] for _ in texts]
    )

    embeddings_mod.embed_query("q")
    embeddings_mod.embed_texts(["a", "b"])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        for value in span.attributes.values():
            assert "synthetic-test-secret" not in str(value)


# --- search_vectors ------------------------------------------------------------------


def test_search_vectors_span_name_kind_and_attributes(monkeypatch, span_exporter):
    fake_points = [_FakePoint(0.91), _FakePoint(0.42)]
    fake_client = _FakeQdrantClient(fake_points)
    monkeypatch.setattr(vector_store_mod, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(vector_store_mod.settings, "qdrant_collection", "compliance_policies")

    result = vector_store_mod.search_vectors([0.1, 0.2, 0.3], top_k=5)

    assert result == fake_points  # unchanged return value
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "search_vectors"
    assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "RETRIEVER"
    assert span.attributes["qdrant.collection"] == "compliance_policies"
    assert span.attributes["qdrant.limit"] == 5
    assert span.attributes["qdrant.returned_count"] == 2
    assert span.attributes["qdrant.top_score"] == 0.91


def test_search_vectors_default_limit_falls_back_to_retrieval_top_k(monkeypatch, span_exporter):
    fake_client = _FakeQdrantClient([])
    monkeypatch.setattr(vector_store_mod, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(vector_store_mod.settings, "retrieval_top_k", 10)

    result = vector_store_mod.search_vectors([0.1, 0.2, 0.3])

    assert result == []
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["qdrant.limit"] == 10
    assert span.attributes["qdrant.returned_count"] == 0
    assert "qdrant.top_score" not in span.attributes  # no points -> no top score


def test_search_vectors_span_includes_retrieval_documents(monkeypatch, span_exporter):
    long_text = "x" * 1500
    points = [
        _FakePointWithPayload(
            0.91,
            {
                "doc_title": "Backup Policy",
                "doc_id": "doc-1",
                "section": "Retention",
                "clause_number": "3.2",
                "text": long_text,
            },
            point_id="chunk-1",
        ),
        _FakePointWithPayload(
            0.42,
            {"doc_title": "Access Policy", "doc_id": "doc-2", "text": "short text"},
            point_id="chunk-2",
        ),
    ]
    monkeypatch.setattr(vector_store_mod, "get_qdrant_client", lambda: _FakeQdrantClient(points))

    vector_store_mod.search_vectors([0.1, 0.2, 0.3], top_k=2)

    span = span_exporter.get_finished_spans()[0]
    attrs = span.attributes

    assert attrs[_doc_attr(0, DocumentAttributes.DOCUMENT_ID)] == "chunk-1"
    assert attrs[_doc_attr(0, DocumentAttributes.DOCUMENT_SCORE)] == 0.91
    # Truncated: module-level constant caps document content on the span.
    assert len(attrs[_doc_attr(0, DocumentAttributes.DOCUMENT_CONTENT)]) == 1000
    meta0 = json.loads(attrs[_doc_attr(0, DocumentAttributes.DOCUMENT_METADATA)])
    assert meta0 == {"doc_title": "Backup Policy", "section": "Retention", "clause_number": "3.2"}

    assert attrs[_doc_attr(1, DocumentAttributes.DOCUMENT_ID)] == "chunk-2"
    assert attrs[_doc_attr(1, DocumentAttributes.DOCUMENT_CONTENT)] == "short text"
    meta1 = json.loads(attrs[_doc_attr(1, DocumentAttributes.DOCUMENT_METADATA)])
    # Optional payload keys default to "" rather than being omitted or raising.
    assert meta1 == {"doc_title": "Access Policy", "section": "", "clause_number": ""}


def test_search_vectors_document_attrs_missing_payload_keys_does_not_raise(
    monkeypatch, span_exporter
):
    points = [_FakePointWithPayload(0.5, {}, point_id="chunk-x")]  # empty payload
    monkeypatch.setattr(vector_store_mod, "get_qdrant_client", lambda: _FakeQdrantClient(points))

    result = vector_store_mod.search_vectors([0.1], top_k=1)  # must not raise

    assert result == points
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes[_doc_attr(0, DocumentAttributes.DOCUMENT_CONTENT)] == ""


def test_search_vectors_no_points_means_no_document_attrs(monkeypatch, span_exporter):
    monkeypatch.setattr(vector_store_mod, "get_qdrant_client", lambda: _FakeQdrantClient([]))

    vector_store_mod.search_vectors([0.1], top_k=1)

    span = span_exporter.get_finished_spans()[0]
    assert _doc_attr(0, DocumentAttributes.DOCUMENT_ID) not in span.attributes


def test_search_vectors_exception_propagates_unchanged_and_span_errors(monkeypatch, span_exporter):
    boom = ResponseHandlingException("qdrant unreachable")
    monkeypatch.setattr(
        vector_store_mod, "get_qdrant_client", lambda: _FakeQdrantClientRaises(boom)
    )

    with pytest.raises(ResponseHandlingException) as excinfo:
        vector_store_mod.search_vectors([0.1, 0.2])
    assert excinfo.value is boom

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


# --- rerank ------------------------------------------------------------------------


def test_rerank_span_name_kind_and_attributes(monkeypatch, span_exporter):
    monkeypatch.setattr(reranker_mod.settings, "reranker_backend", "llama-server")
    monkeypatch.setattr(reranker_mod.settings, "reranker_model", "qwen3-reranker-4b")
    monkeypatch.setattr(reranker_mod.settings, "reranker_top_n", 2)
    monkeypatch.setattr(
        reranker_mod, "_call_rerank", lambda query, documents, top_n: [(1, 0.95), (0, 0.10)]
    )

    results = [
        {"text": "doc0", "doc_title": "Access Policy"},
        {"text": "doc1", "doc_title": "Backup Policy"},
    ]
    output = reranker_mod.rerank("what is the backup retention period?", results)

    assert len(output) == 2
    assert output[0]["rerank_score"] == 0.95  # unchanged behaviour/ordering

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "rerank"
    assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "RERANKER"
    assert span.attributes["reranker.query"] == "what is the backup retention period?"
    assert span.attributes["reranker.model_name"] == "qwen3-reranker-4b"
    assert span.attributes["reranker.backend"] == "llama-server"
    assert span.attributes["reranker.candidates_in"] == 2
    assert span.attributes["reranker.top_k"] == 2
    assert span.attributes["reranker.results_out"] == 2
    assert span.attributes["reranker.top_score"] == 0.95


def test_rerank_span_includes_retrieval_documents(monkeypatch, span_exporter):
    monkeypatch.setattr(reranker_mod.settings, "reranker_backend", "llama-server")
    monkeypatch.setattr(reranker_mod.settings, "reranker_top_n", 2)
    monkeypatch.setattr(
        reranker_mod, "_call_rerank", lambda query, documents, top_n: [(1, 0.95), (0, 0.10)]
    )

    long_text = "a" * 1500
    results = [
        {
            "text": long_text,
            "doc_title": "Access Policy",
            "doc_id": "doc-0",
            "section": "Intro",
            "clause_number": "1.1",
        },
        {
            "text": "short",
            "doc_title": "Backup Policy",
            "doc_id": "doc-1",
            "section": "Retention",
            "clause_number": "3.2",
        },
    ]
    reranker_mod.rerank("q", results)

    span = span_exporter.get_finished_spans()[0]
    attrs = span.attributes

    # index 1 (doc-1) scored highest (0.95) so it becomes document 0 post-rerank.
    assert attrs[_doc_attr(0, DocumentAttributes.DOCUMENT_ID)] == "doc-1"
    assert attrs[_doc_attr(0, DocumentAttributes.DOCUMENT_SCORE)] == 0.95
    assert attrs[_doc_attr(0, DocumentAttributes.DOCUMENT_CONTENT)] == "short"
    meta0 = json.loads(attrs[_doc_attr(0, DocumentAttributes.DOCUMENT_METADATA)])
    assert meta0 == {"doc_title": "Backup Policy", "section": "Retention", "clause_number": "3.2"}

    assert attrs[_doc_attr(1, DocumentAttributes.DOCUMENT_ID)] == "doc-0"
    assert attrs[_doc_attr(1, DocumentAttributes.DOCUMENT_SCORE)] == 0.10
    # Truncated: module-level constant caps document content on the span.
    assert len(attrs[_doc_attr(1, DocumentAttributes.DOCUMENT_CONTENT)]) == 1000


def test_rerank_document_attrs_missing_fields_does_not_raise(monkeypatch, span_exporter):
    monkeypatch.setattr(reranker_mod.settings, "reranker_top_n", 1)
    monkeypatch.setattr(reranker_mod, "_call_rerank", lambda q, d, n: [(0, 0.5)])

    output = reranker_mod.rerank("q", [{"text": "x"}])  # no doc_title/doc_id/etc.

    assert output[0]["rerank_score"] == 0.5  # unchanged behaviour
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes[_doc_attr(0, DocumentAttributes.DOCUMENT_ID)] == ""
    meta0 = json.loads(span.attributes[_doc_attr(0, DocumentAttributes.DOCUMENT_METADATA)])
    assert meta0 == {"doc_title": "", "section": "", "clause_number": ""}


def test_rerank_fallback_documents_have_no_misleading_score(monkeypatch, span_exporter):
    """Fallback results carry no rerank_score (see the existing top_score test above);
    the per-document score must default safely rather than raise a KeyError."""

    def raise_it(query, documents, top_n):
        raise httpx.ConnectError("reranker down")

    monkeypatch.setattr(reranker_mod, "_call_rerank", raise_it)
    monkeypatch.setattr(reranker_mod.settings, "reranker_top_n", 1)

    reranker_mod.rerank("q", [{"text": "a", "doc_title": "X", "doc_id": "d1"}])

    span = span_exporter.get_finished_spans()[0]
    assert span.attributes[_doc_attr(0, DocumentAttributes.DOCUMENT_SCORE)] == 0.0
    assert span.attributes[_doc_attr(0, DocumentAttributes.DOCUMENT_ID)] == "d1"


def test_rerank_empty_results_has_no_document_attrs(monkeypatch, span_exporter):
    reranker_mod.rerank("q", [])

    span = span_exporter.get_finished_spans()[0]
    assert _doc_attr(0, DocumentAttributes.DOCUMENT_ID) not in span.attributes


def test_rerank_swallows_transient_backend_error_without_marking_span_errored(
    monkeypatch, span_exporter
):
    """rerank() deliberately falls back to original order on backend failure and
    never blocks the pipeline — the new span wrapping must not change that."""

    def raise_connect_error(query, documents, top_n):
        raise httpx.ConnectError("reranker down")

    monkeypatch.setattr(reranker_mod, "_call_rerank", raise_connect_error)
    monkeypatch.setattr(reranker_mod.settings, "reranker_top_n", 1)

    results = [{"text": "a"}, {"text": "b"}]
    output = reranker_mod.rerank("q", results)

    assert output == results[:1]  # unchanged fallback behaviour
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code != StatusCode.ERROR


def test_rerank_exception_outside_internal_handling_propagates_and_span_errors(
    monkeypatch, span_exporter
):
    """An exception raised before rerank's own try/except (e.g. a real bug) must
    still propagate unchanged through the span wrapper and mark it errored."""
    boom = ValueError("bad query")

    def raise_boom(question):
        raise boom

    monkeypatch.setattr(reranker_mod, "_build_query", raise_boom)

    with pytest.raises(ValueError) as excinfo:
        reranker_mod.rerank("q", [{"text": "x"}])
    assert excinfo.value is boom

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


def test_rerank_empty_results_still_emits_span(monkeypatch, span_exporter):
    output = reranker_mod.rerank("q", [])

    assert output == []
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["reranker.candidates_in"] == 0
    assert span.attributes["reranker.results_out"] == 0
    assert "reranker.top_score" not in span.attributes


# --- phoenix disabled: no spans, no errors -----------------------------------------


def test_all_four_functions_work_with_phoenix_disabled(monkeypatch):
    # Deliberately NOT using the span_exporter fixture: this exercises the real
    # get_tracer(), which returns a no-op tracer whenever phoenix is disabled.
    monkeypatch.setattr(embeddings_mod.settings, "phoenix_enabled", False)
    monkeypatch.setattr(vector_store_mod.settings, "phoenix_enabled", False)
    monkeypatch.setattr(reranker_mod.settings, "phoenix_enabled", False)

    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(
        embeddings_mod, "_ollama_embed", lambda texts, prefix="": [[0.1, 0.2] for _ in texts]
    )

    fake_points = [_FakePoint(0.7)]
    monkeypatch.setattr(
        vector_store_mod, "get_qdrant_client", lambda: _FakeQdrantClient(fake_points)
    )

    monkeypatch.setattr(
        reranker_mod, "_call_rerank", lambda query, documents, top_n: [(0, 0.8)]
    )

    # None of these should raise, and there is no exporter to assert against —
    # the point is exactly that nothing breaks when Phoenix is off.
    assert embeddings_mod.embed_query("q") == [0.1, 0.2]
    assert embeddings_mod.embed_texts(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert vector_store_mod.search_vectors([0.1, 0.2]) == fake_points
    assert reranker_mod.rerank("q", [{"text": "a"}])[0]["rerank_score"] == 0.8


# --- reranker.fallback: make a silent degradation visible in Phoenix -----------
#
# rerank() swallows backend failures by design so the pipeline never blocks. That
# is right for the user and wrong for the operator: without these attributes the
# span reports OK while the results are actually unranked, so a reranker outage
# looks like a perfectly healthy trace.


def test_rerank_success_records_fallback_false(monkeypatch, span_exporter):
    """The attribute is always present, so `reranker.fallback == false` is a usable
    filter rather than 'absent means fine, probably'."""
    monkeypatch.setattr(reranker_mod.settings, "reranker_top_n", 1)
    monkeypatch.setattr(reranker_mod, "_call_rerank", lambda q, d, n: [(0, 0.9)])

    reranker_mod.rerank("q", [{"text": "a"}])

    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["reranker.fallback"] is False
    assert span.attributes["reranker.top_score"] == 0.9


@pytest.mark.parametrize(
    "exc, expected_reason",
    [
        (httpx.ConnectError("reranker down"), "connect_error"),
        (httpx.TimeoutException("too slow"), "timeout"),
        (RuntimeError("something else"), "RuntimeError"),
    ],
)
def test_rerank_fallback_is_visible_on_the_span(
    monkeypatch, span_exporter, exc, expected_reason
):
    def raise_it(query, documents, top_n):
        raise exc

    monkeypatch.setattr(reranker_mod, "_call_rerank", raise_it)
    monkeypatch.setattr(reranker_mod.settings, "reranker_top_n", 1)

    results = [{"text": "a"}, {"text": "b"}]
    output = reranker_mod.rerank("q", results)

    assert output == results[:1]  # fallback behaviour itself unchanged
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["reranker.fallback"] is True
    assert span.attributes["reranker.fallback_reason"] == expected_reason
    # Still not an errored span: the request succeeded and the user got an answer.
    assert span.status.status_code != StatusCode.ERROR


def test_rerank_fallback_does_not_report_a_misleading_top_score(
    monkeypatch, span_exporter
):
    """Fallback results carry no rerank_score. Defaulting it to 0.0 would read as
    'the reranker scored everything terribly' instead of 'it never ran'."""

    def raise_it(query, documents, top_n):
        raise httpx.ConnectError("reranker down")

    monkeypatch.setattr(reranker_mod, "_call_rerank", raise_it)
    monkeypatch.setattr(reranker_mod.settings, "reranker_top_n", 1)

    reranker_mod.rerank("q", [{"text": "a"}])

    span = span_exporter.get_finished_spans()[0]
    assert "reranker.top_score" not in span.attributes


# --- Ollama embedding keep_alive ----------------------------------------------
#
# Regression guard for a measured production finding (2026-09-22): /api/embed was
# sent without keep_alive, so the embedding model fell back to Ollama's default
# 5-minute TTL while the chat model held 30m. Every question after a >5min gap —
# the normal case for this bot — paid a full model reload: 1,589ms vs 27ms warm.


def test_ollama_embed_sends_keep_alive(monkeypatch):
    """The embedding model must be pinned as long as the chat model, or it is
    evicted between questions and each query reloads it."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"embeddings": [[0.1, 0.2]]}

        return _Resp()

    monkeypatch.setattr(embeddings_mod.httpx, "post", fake_post)
    monkeypatch.setattr(embeddings_mod.settings, "ollama_keep_alive", "30m")
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(embeddings_mod, "_embedding_model", "ollama")

    embeddings_mod.embed_query("does the VPN policy cover contractors?")

    keep_alive = captured["json"].get("keep_alive")
    assert keep_alive == "30m", "embed payload must carry keep_alive"


# --- search_policies_tool async wrapper: trace nesting -------------------------
#
# Production bug: FunctionTool.from_defaults(fn=search_policies) alone makes
# llama-index build its own async wrapper via `sync_to_async`, which schedules
# the call with bare `loop.run_in_executor` — that does NOT propagate
# contextvars, so every span opened inside the tool (embed_query,
# search_vectors, rerank) started its own brand-new root trace instead of
# nesting under the agent's tool-call span. The fix gives the tool an explicit
# `async_fn` that runs the sync function via `asyncio.to_thread`, which does
# copy the context. This test exercises `search_policies_tool.acall` end to
# end (mocked internals only) and asserts on trace id specifically — that is
# the thing that was actually wrong in production, not merely "spans exist".


def test_tool_acall_nests_retrieval_spans_under_parent_trace(monkeypatch, span_exporter):
    monkeypatch.setattr(search_policies_mod.settings, "bm25_enabled", False)
    monkeypatch.setattr(search_policies_mod.settings, "reranker_enabled", True)
    monkeypatch.setattr(search_policies_mod.settings, "reranker_candidates", 5)
    monkeypatch.setattr(search_policies_mod.settings, "reranker_top_n", 2)
    monkeypatch.setattr(search_policies_mod.settings, "min_confidence_score", 0.0)

    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(
        embeddings_mod, "_ollama_embed", lambda texts, prefix="": [[0.1, 0.2, 0.3]]
    )

    fake_points = [
        _FakePointWithPayload(
            0.9, {"doc_title": "Backup Policy", "doc_id": "d1", "text": "chunk one"}, "c1"
        ),
        _FakePointWithPayload(
            0.5, {"doc_title": "Access Policy", "doc_id": "d2", "text": "chunk two"}, "c2"
        ),
    ]
    monkeypatch.setattr(
        vector_store_mod, "get_qdrant_client", lambda: _FakeQdrantClient(fake_points)
    )
    monkeypatch.setattr(reranker_mod, "_call_rerank", lambda q, d, n: [(0, 0.95), (1, 0.2)])

    # Same tracer instance the span_exporter fixture patched every module's
    # get_tracer() to return, so starting the "agent" parent span here shares
    # the exact OTel Context that the retrieval spans below must inherit.
    tracer = vector_store_mod.get_tracer()
    with tracer.start_as_current_span("agent.tool_call") as parent_span:
        parent_trace_id = parent_span.get_span_context().trace_id
        result = asyncio.run(
            search_policies_mod.search_policies_tool.acall(query="backup retention", top_k=2)
        )

    assert "RETRIEVED POLICY SOURCES" in result.raw_output  # pipeline still works

    spans = span_exporter.get_finished_spans()
    retrieval_spans = {s.name: s for s in spans if s.name in {"embed_query", "search_vectors", "rerank"}}
    assert set(retrieval_spans) == {"embed_query", "search_vectors", "rerank"}
    for name, s in retrieval_spans.items():
        assert s.context.trace_id == parent_trace_id, (
            f"{name} span started its own root trace instead of nesting under "
            "the agent's tool-call span"
        )
