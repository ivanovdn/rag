"""search_policies with the rewrite on: one embed call, extras fused, original reranked.

Patched on `sp` (names bound at import). Nothing touches the network.
"""

import httpx
import pytest

import rag.tools.search_policies as sp
from rag.query_rewrite import RewriteResult


class _Hit:
    score = 0.016
    id = "c1"
    payload = {
        "doc_title": "Access Management Policy [Internal]",
        "doc_id": "access-management-policy-internal",
        "section": "Access Management",
        "clause": "Inactivity Logoff/Lockout",
        "clause_number": "4.14",
        "text": "Users are prohibited from disabling automatic lock settings.",
    }


@pytest.fixture
def wired(monkeypatch):
    calls = {}
    monkeypatch.setattr(sp.settings, "reranker_enabled", True)
    monkeypatch.setattr(sp.settings, "bm25_enabled", False)
    monkeypatch.setattr(sp.settings, "reranker_min_score", 0.0)
    monkeypatch.setattr(sp, "rewrite_query", lambda q: RewriteResult("multi", ("alt one", "alt two")))

    def _embed_queries(qs):
        calls["embed"] = list(qs)
        return [[float(i)] for i, _ in enumerate(qs)]

    def _search(q, v, top_k, extra_queries=()):
        calls["search"] = (q, v, top_k, list(extra_queries))
        return [_Hit()]

    def _rerank(query, results, top_n):
        calls["rerank_query"] = query
        return [dict(results[0], rerank_score=0.9)]

    monkeypatch.setattr(sp, "embed_queries", _embed_queries)
    monkeypatch.setattr(sp, "search_chunks", _search)
    monkeypatch.setattr(sp, "rerank", _rerank)
    return calls


def test_original_and_rephrasings_are_embedded_in_one_call(wired):
    sp.search_policies("orig question")
    assert wired["embed"] == ["orig question", "alt one", "alt two"]


def test_rephrasings_go_to_qdrant_as_extras_beside_the_original(wired):
    sp.search_policies("orig question")
    q, v, _top_k, extras = wired["search"]
    assert (q, v) == ("orig question", [0.0])
    assert extras == [("alt one", [1.0]), ("alt two", [2.0])]


def test_the_reranker_judges_against_the_original_question(wired):
    sp.search_policies("orig question")
    assert wired["rerank_query"] == "orig question"


def test_the_rewrite_is_kept_for_eval(wired):
    sp.search_policies("orig question")
    assert sp._last_rewrite["mode"] == "multi"
    assert sp._last_rewrite["queries"] == ["alt one", "alt two"]


def test_rewritten_search_reports_rrf_scores_even_with_bm25_off(wired):
    sp.search_policies("orig question")
    assert sp._last_search_results[0]["score_type"] == "rrf"


def test_a_fallback_rewrite_searches_exactly_as_rewrite_off(monkeypatch, wired):
    """No extras -> the pre-rewrite call shape: embed_query + 3-arg search_chunks."""
    monkeypatch.setattr(sp, "rewrite_query", lambda q: RewriteResult("multi", fallback=True, error="x"))
    monkeypatch.setattr(sp, "embed_queries", lambda qs: pytest.fail("batch path used on fallback"))
    monkeypatch.setattr(sp, "embed_query", lambda q: [9.0])
    monkeypatch.setattr(sp, "search_chunks", lambda q, v, top_k: [_Hit()])

    out = sp.search_policies("orig question")

    assert "[Source 1]" in out
    assert sp._last_rewrite["fallback"] is True
    assert sp._last_search_results[0]["score_type"] == "cosine"


def test_embedding_outage_with_rewrite_on_is_still_unavailable(monkeypatch, wired):
    recorded = []

    def _down(qs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(sp, "embed_queries", _down)
    monkeypatch.setattr(sp, "RETRY_BACKOFFS", ())
    monkeypatch.setattr(sp, "retry_transient", lambda fn: fn())
    monkeypatch.setattr(sp, "record_infra_unavailable", lambda *a: recorded.append(a))

    assert sp.search_policies("orig question") == sp.UNAVAILABLE
    assert sp._retrieval_unavailable is True
    assert recorded[0][0] == "embeddings"


def test_the_pre_rerank_rank_is_kept_for_eval(monkeypatch, wired):
    """original_rank is how a run shows that a rewrite widened the reranker's
    pool: the clause it promoted came from deep in the fused list."""
    monkeypatch.setattr(sp, "rerank", lambda q, results, top_n: [dict(results[0], rerank_score=0.9, original_rank=7)])
    sp.search_policies("orig question")
    assert sp._last_search_results[0]["original_rank"] == 7
