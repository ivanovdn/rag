"""eval's e2e task must short-circuit exactly where channels/teams/bot.py::_run_rag
does (Task 7 follow-up).

The VM eval run this branch's merge decision is based on. An infra failure
(embeddings/Qdrant down) must never reach the agent and must never be recorded
as a content escalation — reproducing that bug inside eval would hand a
"the bot could not ground this answer" result for what was really a transient
backend blip, corrupting the citation-accuracy numbers the decision relies on.
"""

import asyncio

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


def test_the_candidates_flag_is_written_into_settings_not_threaded(monkeypatch):
    """tier2 and chatbot never see a threaded parameter.

    Only make_tier1_task() builds its own retrieval call. The agent tiers go
    through rag.tools.search_policies, which reads settings.reranker_candidates
    directly — so --top-k used to apply to tier1 alone while advertising itself
    generically, and a sweep run on the chatbot tier silently measured the .env
    value instead of the requested one. The resolved value has to land where
    every reader looks.
    """
    import ast
    from pathlib import Path

    source = Path("eval/run_experiment.py").read_text(encoding="utf-8")
    main_fn = next(
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    assigns_setting = [
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "reranker_candidates"
            for t in n.targets
        )
    ]
    assert assigns_setting, (
        "--candidates must be written into settings.reranker_candidates; "
        "a threaded parameter reaches tier1 only"
    )


def test_the_output_says_how_many_results_there_were_before_the_dedup(monkeypatch):
    """eval's search_results is NOT what the agent saw, and nothing said so.

    run_experiment dedupes by (doc_title, section, clause) before writing the
    output. On chatbot-test-v1 that turned 6 reranked sources into 5 for 12 of 61
    questions -- and because reranker.results_out said 6 on every span, reading
    the exported JSON suggested the reranker was dropping one. It was not. The
    pre-dedup count makes the output self-describing, so the next reader does not
    spend an afternoon on /v1/score.
    """
    import eval.run_experiment as re_mod

    def chunk(clause_number):
        return {
            "doc_title": "Acceptable Use Policy [Internal]",
            "section": "Corporate Workstation and Software Use",
            "clause": "Malware Protection",
            "clause_number": clause_number,
            "rerank_score": 0.9,
            "retrieval_score": 0.03,
            "score_type": "rrf",
        }

    # Two chunks of ONE clause, which every retrieval evaluator reads as one hit,
    # plus an unrelated third.
    results = [chunk("5.1"), chunk("5.2"), {**chunk("7.0"), "clause": "Unacceptable Use"}]
    monkeypatch.setattr(sp, "search_policies", lambda q, *a, **k: "[Source 1] text")
    monkeypatch.setattr(sp, "_last_search_results", results)

    class _FakeAgent:
        async def run(self, agent_input):
            return '{"answer": "a", "citations": [], "escalation": {"needed": false, "reason": ""}}'

    # _run_fresh_agent is nested inside make_agent_task, so patch what it calls.
    monkeypatch.setattr(
        "eval.agent_wrapper.build_instrumented_agent", lambda verbose=False: _FakeAgent()
    )

    # e2e_task calls asyncio.get_event_loop(), which the real harness satisfies
    # via setup_async(). Running inside the full suite there may be none left, so
    # supply one rather than depend on test ordering.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        out = re_mod.make_agent_task()({"question": "q"})
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    assert len(out["search_results"]) == 2
    assert out["search_results_before_dedup"] == 3


def test_the_dedup_key_is_exactly_what_the_evaluators_can_distinguish(monkeypatch):
    """Not an arbitrary triple. evaluators._match_result reads doc_title, section
    and clause and never clause_number, so two chunks differing only in
    clause_number are indistinguishable to every retrieval metric -- collapsing
    them cannot change a score. Widening the key would start counting one clause
    twice; narrowing it would merge genuinely different clauses.
    """
    import inspect

    from eval.evaluators import _match_result

    source = inspect.getsource(_match_result)
    for field in ("doc_title", "section", "clause"):
        assert field in source, f"_match_result no longer reads {field}"
    assert "clause_number" not in source, (
        "_match_result now reads clause_number — the dedup key in "
        "run_experiment.py must widen to match, or it merges distinguishable chunks"
    )
