"""The relevance floor (spec D8).

min_confidence_score has never run in production: search_policies applies it only
when the reranker is OFF, and production runs it ON. These tests pin the replacement,
including the two ways it must NOT fire — disabled by default, and never on the
reranker's degraded fallback path, where a missing rerank_score means "the reranker
never ran", not "everything scored zero".
"""

import pytest

import rag.tools.search_policies as sp


@pytest.fixture
def reranked(monkeypatch):
    """search_policies with retrieval mocked, reranker ON, returning one scored hit."""
    monkeypatch.setattr(sp.settings, "bm25_enabled", False)
    monkeypatch.setattr(sp.settings, "reranker_enabled", True)

    class _Hit:
        score = 0.9
        payload = {
            "doc_title": "Acceptable Use Policy [Internal]",
            "doc_id": "acceptable-use-policy-internal",
            "section": "Corporate Workstation and Software Use",
            "clause": "Software Installation",
            "clause_number": "4.7",
            "text": "Team Members are forbidden to install any unlicensed software.",
        }

    monkeypatch.setattr("rag.embeddings.embed_query", lambda q: [0.0] * 768)
    monkeypatch.setattr("rag.vector_store.search_vectors", lambda v, top_k: [_Hit()])
    return sp


def _rerank_returning(score):
    def _fake(query, results, top_n):
        out = dict(results[0])
        if score is not None:
            out["rerank_score"] = score
        return [out]
    return _fake


def test_floor_rejects_a_low_scoring_top_result(reranked, monkeypatch):
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.5)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(0.10))

    assert reranked.search_policies("anything") == "NO_RELEVANT_POLICY_FOUND"


def test_a_rejected_search_still_reports_what_it_found(reranked, monkeypatch):
    """Deliberately unlike the other sentinel paths, which clear _last_search_results.

    Retrieval DID return candidates. The tier-1 retrieval evaluators and the
    threshold tuning both need to see what they were and how they scored, so
    clearing this list here would destroy the only evidence for choosing the
    threshold. Do not "fix" this into matching the other paths.
    """
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.5)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(0.10))

    reranked.search_policies("anything")

    assert len(reranked._last_search_results) == 1
    assert reranked._last_search_results[0]["rerank_score"] == 0.1


def test_floor_passes_a_high_scoring_top_result(reranked, monkeypatch):
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.5)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(0.80))

    assert "[Source 1]" in reranked.search_policies("anything")


def test_a_zero_threshold_disables_the_floor(reranked, monkeypatch):
    """Ships at 0.0; the live value is measured on the VM, not guessed here."""
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.0)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(0.0))

    assert "[Source 1]" in reranked.search_policies("anything")


def test_the_reranker_fallback_path_is_never_floored(reranked, monkeypatch):
    """A reranker outage must not become "no policy exists" for every question.

    rag/reranker.py falls back to the original ordering on error, and those results
    carry NO rerank_score (see its comment at the top_score span attribute). Reading
    a missing score as 0.0 would reject every question the moment the reranker went
    down — turning a degraded-but-working pipeline into a total outage.
    """
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.9)
    monkeypatch.setattr("rag.reranker.rerank", _rerank_returning(None))

    assert "[Source 1]" in reranked.search_policies("anything")


def test_the_default_threshold_is_the_measured_one():
    """Pins what the floor was measured AS, so it cannot drift back to a guess.

    2026-09-25, chatbot-test-v1, 61 distinct questions through the real remote
    stack: correct retrievals scored 0.8478-0.9998 (n=59); the two document
    misses scored 0.5519 and 0.9886. 0.2 therefore sits 4.2x below the lowest
    correct retrieval and fired on none of the 61 — deliberately inert, there to
    catch obviously-irrelevant questions rather than borderline ones.

    Change it only after re-measuring, and update config.py's comment with the
    new sample.
    """
    # Bound to a local first: a failing `assert settings.x == y` prints the whole
    # Settings repr, which carries live .env secrets.
    value = sp.settings.reranker_min_score
    assert value == 0.2
    # Must stay clear of the lowest top_score that produced a CORRECT retrieval.
    assert value < 0.8478
