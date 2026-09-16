import threading
from datetime import datetime, timedelta, timezone

import pytest

import channels.teams.bot as bot


@pytest.fixture
def qbot(monkeypatch):
    """A TeamsBot with network mocked; records every HTML it 'sends'."""
    b = bot.TeamsBot(token_refresher=object())
    sent = []
    monkeypatch.setattr(b, "_send_message",
                        lambda chat_id, text, content_type="html": sent.append(text) or True)
    bot._pending_ratings.clear()
    while not b._work_q.empty():
        b._work_q.get_nowait()
    b._inflight.clear()
    b._sent = sent
    return b


def test_inbound_acks_immediately_and_does_not_run_rag(monkeypatch, qbot):
    monkeypatch.setattr(bot, "_run_rag", lambda q: pytest.fail("poll thread must not run RAG"))
    now = datetime.now(timezone.utc)
    qbot._handle_inbound("chat1", "Can I install software?", "Ann", "m1", now)
    assert any("Got your message" in h for h in qbot._sent)
    assert qbot._work_q.qsize() == 1


def test_inbound_marks_message_inflight(qbot):
    now = datetime.now(timezone.utc)
    qbot._handle_inbound("chat1", "Can I install software?", "Ann", "m1", now)
    assert qbot._inflight == {"m1": now}


def test_rating_is_handled_inline_and_not_enqueued(monkeypatch, qbot):
    monkeypatch.setattr(bot, "save_feedback", lambda **kw: None)
    bot._pending_ratings["chat1"] = {"question": "q", "answer": "a", "citations": [], "user": "Ann"}
    qbot._handle_inbound("chat1", "2", "Ann", "m2", datetime.now(timezone.utc))
    assert qbot._work_q.qsize() == 0
    assert "chat1" not in bot._pending_ratings


def test_worker_drains_job_and_clears_inflight(monkeypatch, qbot):
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(bot, "_run_rag",
                        lambda q: {"answer": "See AUP.", "citations": [], "escalation": {"needed": False}})
    now = datetime.now(timezone.utc)
    qbot._handle_inbound("chat1", "Can I install software?", "Ann", "m1", now)

    t = threading.Thread(target=qbot._worker_loop, daemon=True)
    t.start()
    qbot._work_q.join()

    assert qbot._inflight == {}
    assert any("See AUP." in h for h in qbot._sent)


def test_saved_watermark_is_held_before_the_oldest_inflight_message(qbot, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    old = datetime.now(timezone.utc) - timedelta(minutes=2)
    new = datetime.now(timezone.utc)
    qbot.last_check = new
    qbot.processed_messages = {"m_old", "m_done"}
    qbot._inflight = {"m_old": old}

    qbot._save_state()
    saved = json.loads((tmp_path / "bot_state.json").read_text())

    # Watermark rewound behind the queued message, so a restart re-delivers it.
    assert datetime.fromisoformat(saved["last_check"]) < old
    # ...and it is not in the processed set, which would otherwise skip it.
    assert "m_old" not in saved["processed_messages"]
    assert "m_done" in saved["processed_messages"]


def test_saved_watermark_is_last_check_when_nothing_inflight(qbot, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    now = datetime.now(timezone.utc)
    qbot.last_check = now
    qbot.processed_messages = {"m1"}
    qbot._inflight = {}

    qbot._save_state()
    saved = json.loads((tmp_path / "bot_state.json").read_text())
    assert datetime.fromisoformat(saved["last_check"]) == now


def test_ensure_worker_restarts_a_dead_worker(qbot):
    qbot._ensure_worker()
    first = qbot._worker
    assert first is not None and first.is_alive()

    # Simulate the worker dying: swap in a thread that has already finished.
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    qbot._worker = dead

    qbot._ensure_worker()
    assert qbot._worker is not dead
    assert qbot._worker.is_alive()
