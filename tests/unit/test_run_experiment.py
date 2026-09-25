"""eval's e2e task must short-circuit exactly where channels/teams/bot.py::_run_rag
does (Task 7 follow-up).

The VM eval run this branch's merge decision is based on. An infra failure
(embeddings/Qdrant down) must never reach the agent and must never be recorded
as a content escalation — reproducing that bug inside eval would hand a
"the bot could not ground this answer" result for what was really a transient
backend blip, corrupting the citation-accuracy numbers the decision relies on.
"""

import pytest

import rag.tools.search_policies as sp
from eval.run_experiment import make_agent_task


@pytest.fixture
def no_agent(monkeypatch):
    """Fails loudly if anything builds an agent — that is what we are asserting."""
    def _boom(*args, **kwargs):
        raise AssertionError("e2e_task built an agent on a short-circuit path")

    monkeypatch.setattr("eval.agent_wrapper.build_instrumented_agent", _boom)


def test_an_unavailable_retrieval_short_circuits_without_a_content_escalation(monkeypatch, no_agent):
    """Patches search_policies, not prefetch (the Task 5 lesson: patching prefetch
    would hide a break in prefetch's own UNAVAILABLE classification), so this
    fails if either half of the chain breaks: prefetch's classification of the
    UNAVAILABLE sentinel, or e2e_task's handling of an "unavailable" PrefetchResult.
    """
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.UNAVAILABLE)

    task = make_agent_task()
    result = task({"question": "anything"})

    assert result["status"] == "unavailable"
    assert result["escalation"]["needed"] is False
    assert result["agent_metadata"]["escalated"] is False
    assert result["agent_metadata"]["num_searches"] == 1


def test_a_no_match_search_escalates_without_building_the_agent(monkeypatch, no_agent):
    """Sibling of the unavailable test above, same reasoning: patches
    search_policies, not prefetch, so this fails if either half of the chain
    breaks: prefetch's classification of the NO_MATCH sentinel, or e2e_task's
    handling of a "no_match" PrefetchResult. Without this branch, an
    out-of-corpus question would fall through to compose_agent_input(question,
    "") with an agent actually built and run on an empty, markerless prompt.
    """
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: sp.NO_MATCH)

    task = make_agent_task()
    result = task({"question": "anything"})

    assert result["status"] == "no_match"
    assert result["escalation"]["needed"] is True
    # Fixed text, not model-derived — no agent ran, so nothing else could have
    # produced this string.
    assert result["escalation"]["reason"] == "No relevant policy was found for this question."
    assert result["agent_metadata"]["num_searches"] == 1
