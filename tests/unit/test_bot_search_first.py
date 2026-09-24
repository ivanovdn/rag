"""_run_rag after retrieval moved in front of the agent.

The point of the short-circuits is that they cost no LLM call: an unavailable
backend or a no-match search used to spend a full ~16s agent run to reach the
same conclusion.
"""

import pytest

import channels.teams.bot as bot
import rag.search_first as sf


@pytest.fixture
def no_agent(monkeypatch):
    """Fails loudly if anything builds an agent — that is what we are asserting."""
    def _boom():
        raise AssertionError("_run_rag built an agent on a short-circuit path")

    monkeypatch.setattr("rag.agent.build_agent", _boom)


def test_an_unavailable_search_short_circuits_without_an_llm_call(monkeypatch, no_agent):
    monkeypatch.setattr(sf, "prefetch", lambda q: sf.PrefetchResult("unavailable"))

    assert bot._run_rag("anything") == {"status": "unavailable"}


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
