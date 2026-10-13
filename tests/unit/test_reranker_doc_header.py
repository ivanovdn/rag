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
