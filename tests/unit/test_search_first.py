"""Retrieval runs before the agent (spec D1).

The agent used to decide whether to search. Measured in production: 21 searches
across 25 runs. These tests pin the front-end that removes the decision, and the
two outcomes that must never reach an LLM at all.
"""

import pytest

import rag.search_first as sf
import rag.tools.search_policies as sp


def test_the_question_reaches_search_verbatim(monkeypatch):
    """No rewriting, no keyword extraction — the retrieval stack is tuned for
    natural-language questions, and the old prompt spent tokens saying so."""
    seen = {}
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: seen.setdefault("q", q) or "[Source 1] x")

    sf.prefetch("If it's just for internal tools, can I skip approvals?")

    assert seen["q"] == "If it's just for internal tools, can I skip approvals?"


def test_sources_are_returned_on_the_ok_path(monkeypatch):
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: "=== RETRIEVED POLICY SOURCES ===\n\n[Source 1] AUP")

    result = sf.prefetch("anything")

    assert result.status == "ok"
    assert "[Source 1] AUP" in result.sources


def test_the_bare_no_match_sentinel_is_classified(monkeypatch):
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.NO_MATCH)

    result = sf.prefetch("anything")

    assert result.status == "no_match"
    assert result.sources == ""


def test_the_formatted_no_match_sentinel_is_classified(monkeypatch):
    """format_sources([]) wraps the sentinel in a header, so it arrives in a
    second shape. Both must classify the same way."""
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.format_sources([]))

    assert sf.prefetch("anything").status == "no_match"


def test_the_unavailable_sentinel_is_classified(monkeypatch):
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.UNAVAILABLE)

    assert sf.prefetch("anything").status == "unavailable"


def test_the_unavailable_flag_is_honoured_even_without_the_sentinel(monkeypatch):
    """Belt and braces: the flag is the authoritative signal, the string is a
    convenience. A future return-shape change must not silently downgrade an
    infra outage into a content escalation."""
    def _flagged(q, *a, **k):
        sp._retrieval_unavailable = True
        return "whatever"

    monkeypatch.setattr(sp, "search_policies", _flagged)

    assert sf.prefetch("anything").status == "unavailable"


def test_prefetch_resets_the_unavailable_flag_before_searching(monkeypatch):
    """The flag is a module global read after the call; a stale True from a
    previous request would report a false outage."""
    sp._retrieval_unavailable = True
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: "[Source 1] x")

    assert sf.prefetch("anything").status == "ok"


def test_an_unavailable_prefetch_logs_which_component_failed(monkeypatch, capsys):
    """search_policies itself never logs — it only emits the Phoenix span. This
    print is the only container-log record that retrieval, not the LLM, failed.
    It used to live in _run_rag's post-run check, which Task 5 deletes."""
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.UNAVAILABLE)

    sf.prefetch("anything")

    assert "Unavailable (retrieval)" in capsys.readouterr().out


def test_compose_puts_the_question_first_then_the_sources():
    composed = sf.compose_agent_input("Can I install software?", "=== RETRIEVED POLICY SOURCES ===\n\n[Source 1] AUP")

    assert composed.startswith("Can I install software?")
    assert "[Source 1] AUP" in composed
