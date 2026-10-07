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

    monkeypatch.setattr(sp, "embed_query", lambda q: [0.0] * 768)
    monkeypatch.setattr(sp, "search_chunks", lambda q, v, top_k: [_Hit()])
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
    monkeypatch.setattr(sp, "rerank", _rerank_returning(0.10))

    assert reranked.search_policies("anything") == "NO_RELEVANT_POLICY_FOUND"


def test_a_rejected_search_still_reports_what_it_found(reranked, monkeypatch):
    """Deliberately unlike the other sentinel paths, which clear _last_search_results.

    Retrieval DID return candidates. The tier-1 retrieval evaluators and the
    threshold tuning both need to see what they were and how they scored, so
    clearing this list here would destroy the only evidence for choosing the
    threshold. Do not "fix" this into matching the other paths.
    """
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.5)
    monkeypatch.setattr(sp, "rerank", _rerank_returning(0.10))

    reranked.search_policies("anything")

    assert len(reranked._last_search_results) == 1
    assert reranked._last_search_results[0]["rerank_score"] == 0.1


def test_floor_passes_a_high_scoring_top_result(reranked, monkeypatch):
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.5)
    monkeypatch.setattr(sp, "rerank", _rerank_returning(0.80))

    assert "[Source 1]" in reranked.search_policies("anything")


def test_a_zero_threshold_disables_the_floor(reranked, monkeypatch):
    """Ships at 0.0; the live value is measured on the VM, not guessed here."""
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.0)
    monkeypatch.setattr(sp, "rerank", _rerank_returning(0.0))

    assert "[Source 1]" in reranked.search_policies("anything")


def test_the_reranker_fallback_path_is_never_floored(reranked, monkeypatch):
    """A reranker outage must not become "no policy exists" for every question.

    rag/reranker.py falls back to the original ordering on error, and those results
    carry NO rerank_score (see its comment at the top_score span attribute). Reading
    a missing score as 0.0 would reject every question the moment the reranker went
    down — turning a degraded-but-working pipeline into a total outage.
    """
    monkeypatch.setattr(reranked.settings, "reranker_min_score", 0.9)
    monkeypatch.setattr(sp, "rerank", _rerank_returning(None))

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


# --- the predicate itself ----------------------------------------------------


def test_the_cosine_floor_applies_only_with_reranker_and_bm25_both_off():
    """min_confidence_score is a COSINE threshold, so it may judge the top score
    only when that score IS a cosine similarity.

    Both terms are load-bearing and neither is redundant: a reranked result
    carries a 0.0-1.0 rerank score, and a fused one an RRF score of about
    1/(60 + rank) ~ 0.016, which against 0.45 returns NO_MATCH for every
    question in the corpus with no error raised anywhere.
    """
    from config import Settings

    def applies(rerank, bm25):
        # _env_file=None so a developer's own .env cannot decide the answer.
        cfg = Settings(_env_file=None, reranker_enabled=rerank, bm25_enabled=bm25)
        return cfg.cosine_floor_applies

    assert applies(False, False) is True
    assert applies(True, False) is False
    assert applies(False, True) is False
    assert applies(True, True) is False


def test_nothing_rederives_the_cosine_floor_predicate_inline():
    """One definition, because a drifting copy fails silently in both directions.

    The condition was written out longhand in four places — the guard in
    search_policies and three eval metadata mirrors — and it is exactly the
    kind that rots: dropping the bm25 half compares an RRF score against a
    cosine threshold and returns NO_MATCH for every question, while dropping
    the reranker half re-enables a floor that production has never run. Both
    are silent. Shaped after test_no_undefined_names.py: a source check,
    because there is no behaviour to assert on a copy that merely exists.
    """
    import re
    from pathlib import Path

    # Whitespace-collapsed so the multi-line `if (...)` form matches too.
    pattern = re.compile(r"not\s+\S*\.?reranker_enabled\s+and\s+not\s+\S*\.?bm25_enabled")

    offenders = []
    for path in sorted(Path(".").rglob("*.py")):
        if any(part.startswith(".") or part in {"__pycache__", "tests"} for part in path.parts):
            continue
        # config.py holds the one definition.
        if path.name == "config.py":
            continue
        flat = " ".join(path.read_text(encoding="utf-8").split())
        if pattern.search(flat):
            offenders.append(f"{path} re-derives the predicate; use settings.cosine_floor_applies")

    assert offenders == [], "\n" + "\n".join(offenders)
