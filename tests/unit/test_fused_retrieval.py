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


def test_bm25_on_sends_two_prefetch_branches_fused_by_rrf_at_k_60(
    monkeypatch, client, span_exporter
):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs.settings, "hybrid_vector_candidates", 20)
    monkeypatch.setattr(vs.settings, "hybrid_bm25_candidates", 15)

    vs.search_chunks("retention period", [0.1, 0.2], top_k=6)

    call = client.calls[0]
    dense, sparse = call["prefetch"]
    assert dense.query == [0.1, 0.2]
    assert dense.limit == 20
    assert dense.using is None
    assert sparse.using == "bm25"
    assert sparse.limit == 15
    assert sparse.query.model == "qdrant/bm25"
    assert sparse.query.text == "retention period"
    # k is the point, not just the fusion method. A bare FusionQuery(Fusion.RRF)
    # takes Qdrant's default of 2; the deleted client-side hybrid_search.py used
    # 60, and the eval gate's premise is that the encoder is the only variable
    # that changed. Measured on 1.17.1: k=2 gives 0.5/0.333/0.25 for ranks 1-3,
    # k=60 gives 0.016667/0.016393/0.016129 — exactly 1/60, 1/61, 1/62.
    assert call["query"].rrf.k == 60
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
    assert span.attributes["qdrant.rrf_k"] == 60
    assert span.attributes["qdrant.returned_count"] == 2
    assert span.attributes["qdrant.top_score"] == 0.91


def test_the_fusion_constant_is_pinned_not_left_to_the_server_default(
    monkeypatch, client
):
    """Qdrant's default k is 2. At k=2 an RRF score sits in the same numeric
    range as a cosine similarity, which would turn the cosine-floor guard in
    search_policies from "wrong for every question" into "wrong for some
    questions" — the harder failure to diagnose. It would also move the ranking
    the eval gate and spec D10's fallback are measured against.
    """
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)

    vs.search_chunks("q", [0.1], top_k=6)

    assert vs.RRF_K == 60
    assert client.calls[0]["query"].model_dump() == {"rrf": {"k": 60, "weights": None}}


def test_default_limit_falls_back_to_retrieval_top_k(monkeypatch, client, span_exporter):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)
    monkeypatch.setattr(vs.settings, "retrieval_top_k", 10)

    vs.search_chunks("q", [0.1])

    assert client.calls[0]["limit"] == 10
