"""_run_rag after retrieval moved in front of the agent.

The point of the short-circuits is that they cost no LLM call: an unavailable
backend or a no-match search used to spend a full ~16s agent run to reach the
same conclusion.
"""

import pytest

import channels.teams.bot as bot
import rag.search_first as sf
import rag.tools.search_policies as sp


@pytest.fixture
def no_agent(monkeypatch):
    """Fails loudly if anything builds an agent — that is what we are asserting."""
    def _boom():
        raise AssertionError("_run_rag built an agent on a short-circuit path")

    monkeypatch.setattr("rag.agent.build_agent", _boom)


def test_an_unavailable_search_short_circuits_without_an_llm_call(monkeypatch, no_agent):
    monkeypatch.setattr(sf, "prefetch", lambda q: sf.PrefetchResult("unavailable"))

    assert bot._run_rag("anything") == {"status": "unavailable"}


def test_an_unavailable_retrieval_short_circuits_end_to_end(monkeypatch, no_agent, capsys):
    """Replaces tests/unit/test_bot_queue.py::test_run_rag_logs_retrieval_as_the_failing_component,
    deleted in the same change that added this file. That test's trigger — a tool
    setting sp._retrieval_unavailable during agent.run() — died along with the tools
    (Task 4 left zero of them), but what it actually pinned was bot behaviour on an
    unavailable retrieval end to end, and that seam is still real and still worth
    guarding.

    Every other unavailable-retrieval test mocks one side of the seam in isolation:
    this file's own short-circuit test above stubs sf.prefetch wholesale (proves
    _run_rag's branching, not prefetch's classification), and
    test_search_first.py's tests call prefetch() directly (proves prefetch's
    classification and logging, never through _run_rag). None of them drives the
    real bot._run_rag() -> the real rag.search_first.prefetch() -> a faked
    rag.tools.search_policies.search_policies() and checks the composition — the
    exact thing the deleted test proved. This one patches search_policies, not
    prefetch, so it would fail if either half of that chain broke: _run_rag's
    handling of an "unavailable" PrefetchResult, or prefetch's own classification
    and log line. Do not delete this again without replacing what it covers.
    """
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.UNAVAILABLE)

    result = bot._run_rag("anything")

    assert result == {"status": "unavailable"}
    assert "Unavailable (retrieval)" in capsys.readouterr().out


def test_a_no_match_search_escalates_without_an_llm_call(monkeypatch, no_agent):
    """CLAUDE.md requires this ('If search_policies returns NO_RELEVANT_POLICY_FOUND
    → escalate'). It used to be a prompt instruction the model could ignore; here
    it becomes a guarantee."""
    monkeypatch.setattr(sf, "prefetch", lambda q: sf.PrefetchResult("no_match"))

    result = bot._run_rag("anything")

    assert result["escalation"]["needed"] is True
    assert result["citations"] == []
    assert result["parse_success"] is True


def test_the_sources_reach_the_agent(monkeypatch):
    """The whole point: the agent answers from sources it did not have to ask for."""
    monkeypatch.setattr(
        sf, "prefetch", lambda q: sf.PrefetchResult("ok", "=== RETRIEVED POLICY SOURCES ===\n\n[Source 1] AUP")
    )
    seen = {}

    class _Agent:
        async def run(self, user_msg):
            seen["msg"] = user_msg
            return '{"answer": "a", "citations": [], "escalation": {"needed": false, "reason": ""}}'

    monkeypatch.setattr("rag.agent.build_agent", lambda: _Agent())

    bot._run_rag("Can I install software?")

    assert seen["msg"].startswith("Can I install software?")
    assert "[Source 1] AUP" in seen["msg"]
