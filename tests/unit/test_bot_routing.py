import pytest

import channels.teams.bot as bot
import rag.router as router
from rag.router import RouterDecision, Category


@pytest.fixture
def teams_bot(monkeypatch):
    """A TeamsBot with network + RAG mocked; records every HTML it 'sends'."""
    b = bot.TeamsBot(token_refresher=object())
    sent = []
    monkeypatch.setattr(b, "_send_message", lambda chat_id, text, content_type="html": sent.append(text) or True)
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
    teams_bot._send_reply("chat1", "hello")
    assert called["rag"] is False
    assert any("Compliance Policy Assistant" in h for h in teams_bot._sent)  # WELCOME_HTML
    assert not any("Searching compliance policies" in h for h in teams_bot._sent)
    assert "chat1" not in bot._pending_ratings


def test_out_of_scope_replies_redirect_no_search(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.OUT_OF_SCOPE)
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("must not search"))
    teams_bot._send_reply("chat1", "order me a pizza")
    assert any("only answer questions about company policies" in h for h in teams_bot._sent)
    assert not any("Searching compliance policies" in h for h in teams_bot._sent)
    assert "chat1" not in bot._pending_ratings


def test_unintelligible_replies_retype_no_search(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.UNINTELLIGIBLE)
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("must not search"))
    teams_bot._send_reply("chat1", "църфе ші")
    assert any("retype" in h.lower() for h in teams_bot._sent)
    assert not any("Searching compliance policies" in h for h in teams_bot._sent)
    assert "chat1" not in bot._pending_ratings


def test_in_scope_runs_rag_and_prompts_rating(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    _force(monkeypatch, Category.IN_SCOPE)
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: {"answer": "See AUP.", "citations": [], "escalation": {"needed": False}})
    teams_bot._send_reply("chat1", "Can I install software?")
    assert any("Searching compliance policies" in h for h in teams_bot._sent)  # LOADING_HTML
    assert "chat1" in bot._pending_ratings  # rating prompt stored


def test_low_confidence_safe_default_searches(monkeypatch, teams_bot):
    # OUT_OF_SCOPE but below floor -> resolve() forces IN_SCOPE -> search runs.
    monkeypatch.setattr(bot.settings, "router_enabled", True)
    monkeypatch.setattr(bot.settings, "router_confidence_floor", 0.6)
    _force(monkeypatch, Category.OUT_OF_SCOPE, confidence=0.3)
    called = {"rag": False}
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: called.__setitem__("rag", True) or {"answer": "x", "citations": [], "escalation": {"needed": False}})
    teams_bot._send_reply("chat1", "ambiguous thing")
    assert called["rag"] is True
    assert not any("only answer questions about company policies" in h for h in teams_bot._sent)
    assert not any("Compliance Policy Assistant" in h for h in teams_bot._sent)
    assert not any("retype" in h.lower() for h in teams_bot._sent)


def test_router_disabled_bypasses_classifier(monkeypatch, teams_bot):
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(router, "classify_message", lambda text: pytest.fail("classifier must not run"))
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: {"answer": "x", "citations": [], "escalation": {"needed": False}})
    teams_bot._send_reply("chat1", "hello")  # would be a greeting, but router off -> search
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

    def _record(category, confidence, fallback):
        recorded["category"] = category
        recorded["confidence"] = confidence
        recorded["fallback"] = fallback

    monkeypatch.setattr("rag.observability.record_classification", _record)

    teams_bot._send_reply("chat1", "Can I install software?")

    assert rag_called["called"] is True, "in_scope path must run RAG"
    assert recorded.get("fallback") is True, "record_classification must be called with fallback=True"
