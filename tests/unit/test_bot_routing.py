import pytest

import channels.teams.bot as bot
import rag.router as router
from rag.router import RouterDecision, Category


@pytest.fixture
def teams_bot(monkeypatch, tmp_path):
    """A TeamsBot with network + RAG mocked; records every HTML it 'sends'."""
    # Hermetic: never read or write the developer's real bot_state.json.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    b = bot.TeamsBot(token_refresher=object())
    sent = []
    monkeypatch.setattr(b, "_send_message", lambda chat_id, text, content_type="html", retry=False: sent.append(text) or True)
    bot._pending_ratings.clear()
    b._sent = sent
    return b


def _force(monkeypatch, category, confidence=0.95):
    monkeypatch.setattr(router, "classify_message",
                        lambda text: RouterDecision(category=category, confidence=confidence))


def test_greeting_replies_welcome_no_search(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.GREETING)
    called = {"rag": False}
    monkeypatch.setattr(bot, "_run_rag", lambda q: called.__setitem__("rag", True) or {})
    teams_bot._answer("chat1", "hello")
    assert called["rag"] is False
    assert any("Trinetix Compliance" in h for h in teams_bot._sent)  # WELCOME_HTML
    assert not any("Got your message" in h for h in teams_bot._sent)  # ack is the poll thread's job
    assert "chat1" not in bot._pending_ratings


def test_out_of_scope_replies_redirect_no_search(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.OUT_OF_SCOPE)
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("must not search"))
    teams_bot._answer("chat1", "order me a pizza")
    assert any("only answer questions about company policies" in h for h in teams_bot._sent)
    assert not any("Got your message" in h for h in teams_bot._sent)  # ack is the poll thread's job
    assert "chat1" not in bot._pending_ratings


def test_unintelligible_replies_retype_no_search(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.UNINTELLIGIBLE)
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("must not search"))
    teams_bot._answer("chat1", "църфе ші")
    assert any("retype" in h.lower() for h in teams_bot._sent)
    assert not any("Got your message" in h for h in teams_bot._sent)  # ack is the poll thread's job
    assert "chat1" not in bot._pending_ratings


def test_in_scope_runs_rag_and_prompts_rating(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.IN_SCOPE)
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: {"answer": "See AUP.", "citations": [], "escalation": {"needed": False}})
    teams_bot._answer("chat1", "Can I install software?")
    assert "chat1" in bot._pending_ratings  # rating prompt stored
    # Combined reply send: answer and rating prompt are one Graph call, not two.
    assert len(teams_bot._sent) == 1
    assert "See AUP." in teams_bot._sent[0]
    assert "Was this helpful" in teams_bot._sent[0]


def test_unavailable_path_sends_once_with_no_rating_prompt(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(bot, "_run_rag", lambda q: {"status": "unavailable"})

    assert teams_bot._answer("chat1", "Can I install software?") is True

    assert len(teams_bot._sent) == 1
    assert "Was this helpful" not in teams_bot._sent[0]
    assert "chat1" not in bot._pending_ratings


def test_escalation_path_still_carries_rating_prompt_in_one_send(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(
        bot, "_run_rag",
        lambda q: {"answer": "", "citations": [], "escalation": {"needed": True, "reason": "No policy found."}},
    )

    teams_bot._answer("chat1", "Can I install software?")

    assert len(teams_bot._sent) == 1
    assert "Escalated to Compliance Team" in teams_bot._sent[0]
    assert "Was this helpful" in teams_bot._sent[0]
    assert "chat1" in bot._pending_ratings


def test_error_path_still_carries_rating_prompt_in_one_send(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(
        bot, "_run_rag",
        lambda q: {"answer": "", "citations": [], "escalation": {"needed": False}},
    )

    teams_bot._answer("chat1", "Can I install software?")

    assert len(teams_bot._sent) == 1
    assert "Compliance lookup failed" in teams_bot._sent[0]
    assert "Was this helpful" in teams_bot._sent[0]
    assert "chat1" in bot._pending_ratings


def test_low_confidence_safe_default_searches(monkeypatch, teams_bot):
    # OUT_OF_SCOPE but below floor -> resolve() forces IN_SCOPE -> search runs.
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    monkeypatch.setattr(bot.settings, "router_confidence_floor", 0.6)
    _force(monkeypatch, Category.OUT_OF_SCOPE, confidence=0.3)
    called = {"rag": False}
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: called.__setitem__("rag", True) or {"answer": "x", "citations": [], "escalation": {"needed": False}})
    teams_bot._answer("chat1", "ambiguous thing")
    assert called["rag"] is True
    assert not any("only answer questions about company policies" in h for h in teams_bot._sent)
    assert not any("Trinetix Compliance" in h for h in teams_bot._sent)
    assert not any("retype" in h.lower() for h in teams_bot._sent)


def test_router_disabled_bypasses_classifier(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(router, "classify_message", lambda text: pytest.fail("classifier must not run"))
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: {"answer": "x", "citations": [], "escalation": {"needed": False}})
    teams_bot._answer("chat1", "hello")  # would be a greeting, but router off -> search
    assert "chat1" in bot._pending_ratings


def test_fallback_decision_sets_fallback_flag_and_searches(monkeypatch, teams_bot):
    """decision.fallback=True makes record_classification(fallback=True) even when category matches resolved."""
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    # Force classifier to return a failure-fallback decision: category==IN_SCOPE, fallback=True.
    # resolve() will return IN_SCOPE (because decision.fallback is True), so category == decision.category;
    # the OR-term `decision.fallback` is what drives fallback=True in record_classification.
    monkeypatch.setattr(
        router,
        "classify_message",
        lambda text: RouterDecision(category=Category.IN_SCOPE, confidence=0.0, fallback=True),
    )

    rag_called = {"called": False}
    monkeypatch.setattr(
        bot,
        "_run_rag",
        lambda q: rag_called.__setitem__("called", True) or {"answer": "x", "citations": [], "escalation": {"needed": False}},
    )

    recorded = {}

    def _record(category, confidence, fallback, message):
        recorded["category"] = category
        recorded["confidence"] = confidence
        recorded["fallback"] = fallback
        recorded["message"] = message

    monkeypatch.setattr("rag.observability.record_classification", _record)

    teams_bot._answer("chat1", "Can I install software?")

    assert rag_called["called"] is True, "in_scope path must run RAG"
    assert recorded.get("fallback") is True, "record_classification must be called with fallback=True"
    assert recorded.get("message") == "Can I install software?", "message must be recorded for audit"


def test_router_branches_report_a_failed_send(monkeypatch, teams_bot):
    """Every branch out of _answer returns its send result, so the worker's
    undelivered-answer ERROR log covers greetings and redirects too."""
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    monkeypatch.setattr(teams_bot, "_send_message",
                        lambda chat_id, text, content_type="html", retry=False: None)

    for category in (Category.GREETING, Category.OUT_OF_SCOPE, Category.UNINTELLIGIBLE):
        _force(monkeypatch, category)
        assert not teams_bot._answer("chat1", "hello"), f"{category} must report the failed send"


# --- router short-circuit logging -------------------------------------------
#
# The three branches above return bool(self._send_message(...)) and used to print
# nothing on success, while _worker_loop logs only the falsy-return case: a FAILED
# send was reported and a SUCCESSFUL one was silent, so in the container log a
# working greeting looked exactly like a dropped message. capsys as in
# tests/unit/test_bot_queue.py's ack/worker log tests.

@pytest.mark.parametrize(
    "category, message, expected_line",
    [
        (Category.GREETING, "hi", "[worker] Greeting reply sent"),
        (Category.OUT_OF_SCOPE, "order me a pizza", "[worker] Out-of-scope reply sent"),
        (Category.UNINTELLIGIBLE, "църфе ші", "[worker] Unintelligible reply sent"),
    ],
)
def test_router_short_circuit_logs_its_success_line(
    monkeypatch, teams_bot, capsys, category, message, expected_line
):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, category)
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("must not search"))

    assert teams_bot._answer("chat1", message) is True  # unchanged return value

    logged = capsys.readouterr().out
    assert expected_line in logged


@pytest.mark.parametrize(
    "category, message",
    [
        (Category.GREETING, "hi"),
        (Category.OUT_OF_SCOPE, "order me a pizza"),
        (Category.UNINTELLIGIBLE, "църфе ші"),
    ],
)
def test_router_short_circuit_logs_nothing_when_the_send_fails(
    monkeypatch, teams_bot, capsys, category, message
):
    """A failed send must stay the worker's ERROR to report, not a false success."""
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, category)
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("must not search"))
    monkeypatch.setattr(teams_bot, "_send_message",
                        lambda chat_id, text, content_type="html", retry=False: None)

    assert teams_bot._answer("chat1", message) is False  # unchanged return value

    assert "reply sent" not in capsys.readouterr().out.lower()
