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


def test_a_non_transient_retrieval_error_escalates_without_an_llm_call(monkeypatch, no_agent, capsys):
    """FIX (final branch review): prefetch() used to be called before _run_rag's
    try/except, so a non-transient retrieval error (a real bug, not backend
    flakiness) propagated straight out of _run_rag, past _answer's
    compliance_request span (which never got an "outcome" attribute — invisible
    in the Phoenix outcome breakdown) and into _worker_loop's catch-all, which
    sends a generic "something went wrong" reply instead of the escalation the
    spec promises for non-transient failures.

    search_policies only raises when is_transient(exc) is False — a transient
    one is caught internally and turned into sp.UNAVAILABLE, never raised (see
    rag/tools/search_policies.py) — so any exception reaching prefetch() is
    non-transient by construction. Same shape check as
    test_an_unavailable_retrieval_short_circuits_end_to_end above: patch
    search_policies itself, drive the real bot._run_rag -> real prefetch, and
    assert on the composed result, not on a stubbed prefetch.
    """
    def _boom(q, *a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(sp, "search_policies", _boom)

    result = bot._run_rag("anything")

    assert result == {
        "answer": "",
        "citations": [],
        "escalation": {"needed": True, "reason": "boom"},
    }
    assert "RAG pipeline error (retrieval)" in capsys.readouterr().out


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


# --- the grounding backstop -------------------------------------------------
#
# CLAUDE.md: "Agent must never answer without citing a retrieved chunk."
# Two ways that was violated in production, both demonstrated:
#   - a failed parse falls back to escalation.needed=False with the raw model
#     text as `answer`, so bot.py renders it and logs outcome="answered"
#   - an answer with zero citations renders as bare prose

def _sent(monkeypatch, tmp_path, result):
    """Run _answer with _run_rag stubbed; return (html, outcome)."""
    import contextlib

    import rag.observability as obs

    # Hermetic: never read or write the developer's real bot_state.json / bot.pid.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    monkeypatch.setattr(bot, "PID_FILE", tmp_path / "bot.pid")

    b = bot.TeamsBot(token_refresher=object())
    captured = {}
    monkeypatch.setattr(bot, "_run_rag", lambda q: result)
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(
        b, "_send_message",
        lambda chat_id, text, content_type="html", retry=False: captured.setdefault("html", text) or True,
    )

    class _Span:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

    span = _Span()

    class _Tracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name, **kwargs):
            yield span

    # _answer imports get_tracer INSIDE the function (the observability-first rule),
    # so the module attribute is what the call resolves — patch it, not bot's.
    monkeypatch.setattr(obs, "get_tracer", lambda: _Tracer())

    b._answer("chat1", "Can I install software?", "Ann")
    return captured.get("html", ""), span.attrs.get("compliance_request.outcome")


def test_a_failed_parse_is_escalated_not_answered(monkeypatch, tmp_path):
    html, outcome = _sent(monkeypatch, tmp_path, {
        "answer": "ESCALATED: Ticket #ESC-2026-0001. They will respond within 2 business days.",
        "citations": [],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": False,
    })

    assert outcome == "escalated_parse_failure"
    assert "ESC-2026-0001" not in html


def test_the_parse_failure_reason_never_carries_model_text(monkeypatch, tmp_path):
    """The renderer does not HTML-escape. Putting the raw response into `reason`
    would interpolate arbitrary model output straight into a Teams message."""
    html, _ = _sent(monkeypatch, tmp_path, {
        "answer": "<script>alert(1)</script> and <b>markup</b>",
        "citations": [],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": False,
    })

    assert "<script>" not in html


def test_an_answer_without_citations_is_escalated(monkeypatch, tmp_path):
    html, outcome = _sent(monkeypatch, tmp_path, {
        "answer": "You should ask IT before installing anything.",
        "citations": [],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": True,
    })

    assert outcome == "escalated_ungrounded"
    assert "You should ask IT" not in html


def test_a_cited_answer_is_still_answered(monkeypatch, tmp_path):
    _, outcome = _sent(monkeypatch, tmp_path, {
        "answer": "According to the AUP ...",
        "citations": [{"doc_title": "AUP", "section": "Use", "clause": "", "clause_number": "4.7", "quote": "q"}],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": True,
    })

    assert outcome == "answered"
