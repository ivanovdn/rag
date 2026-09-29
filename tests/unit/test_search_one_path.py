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
