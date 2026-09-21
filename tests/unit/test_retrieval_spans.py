"""
Unit tests for the retrieval-path Phoenix/OpenTelemetry spans added to the four
leaf functions: rag.embeddings.embed_query / embed_texts, rag.vector_store.search_vectors,
and rag.reranker.rerank.

Uses an in-memory OTel span exporter — never a live Phoenix instance — and mocks all
HTTP/Qdrant calls, so nothing here touches the network or 172.20.0.22.
"""

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from openinference.semconv.trace import SpanAttributes
from qdrant_client.http.exceptions import ResponseHandlingException

import rag.embeddings as embeddings_mod
import rag.reranker as reranker_mod
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
