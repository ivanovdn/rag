"""Tier1 measures the retrieval that ships: it calls search_policies."""

import inspect

import pytest

import eval.run_experiment as rx
import rag.tools.search_policies as sp


def test_tier1_has_no_private_copy_of_retrieval():
    src = inspect.getsource(rx.make_tier1_task)
    assert "search_policies(" in src
    for name in ("embed_query", "search_chunks", "rerank("):
        assert name not in src, name


def test_tier1_returns_what_search_policies_found_and_the_rewrite(monkeypatch):
    def _fake(query, top_k=6):
        sp._retrieval_unavailable = False
        sp._last_search_results = [
            {"doc_title": "D", "section": "S", "clause": "C", "clause_number": "1.1",
             "rerank_score": 0.9, "retrieval_score": 0.016, "score_type": "rrf"}
        ]
        sp._last_rewrite = {"mode": "multi", "queries": ["a", "b"], "fallback": False, "error": "", "latency_ms": 7}
        return "formatted"

    monkeypatch.setattr(sp, "search_policies", _fake)
    out = rx.make_tier1_task(top_k=25)({"question": "q"})

    assert out["search_results"][0]["doc_title"] == "D"
    assert out["rewrite"]["queries"] == ["a", "b"]


def test_tier1_raises_when_retrieval_is_unavailable(monkeypatch):
    """A swallowed outage would score 0.0 and read as bad retrieval."""

    def _down(query, top_k=6):
        sp._retrieval_unavailable = True
        sp._last_search_results = []
        return sp.UNAVAILABLE

    monkeypatch.setattr(sp, "search_policies", _down)
    with pytest.raises(RuntimeError, match="unavailable"):
        rx.make_tier1_task(top_k=25)({"question": "q"})


def test_agent_tiers_log_the_rewrite_with_each_search():
    import eval.agent_wrapper as aw

    assert '"rewrite": dict(sp._last_rewrite)' in inspect.getsource(aw.prefetch_logged)
    assert '"rewrite":' in inspect.getsource(rx.make_agent_task)
