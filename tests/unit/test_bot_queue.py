import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

import channels.teams.bot as bot


def _drain(work_q, timeout=5):
    """Bounded wait for the queue to empty.

    queue.Queue.join() takes no timeout, so a worker that dies before task_done()
    would hang the whole suite. Wait on a helper thread instead: a stuck queue
    fails the test rather than blocking CI. When join() returns, _worker_loop's
    finally block has already popped the in-flight id.
    """
    joiner = threading.Thread(target=work_q.join, daemon=True)
    joiner.start()
    joiner.join(timeout)
    assert not joiner.is_alive(), f"queue did not drain within {timeout}s"


@pytest.fixture
def qbot(monkeypatch, tmp_path):
    """A TeamsBot with network mocked; records every HTML it 'sends'."""
    # Hermetic: never read or write the developer's real bot_state.json.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    b = bot.TeamsBot(token_refresher=object())
    sent = []
    monkeypatch.setattr(b, "_send_message",
                        lambda chat_id, text, content_type="html", retry=False: sent.append(text) or True)
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
    _drain(qbot._work_q)

    assert qbot._inflight == {}
    assert any("See AUP." in h for h in qbot._sent)


def test_worker_survives_an_exception_and_never_leaks_its_text(monkeypatch, qbot):
    """A failing job gets a fixed message, not the exception, and the worker lives on."""
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    secret = "Traceback-detail-that-must-not-be-rendered"
    calls = {"n": 0}

    def _rag(question):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(secret)
        return {"answer": "See AUP.", "citations": [], "escalation": {"needed": False}}

    monkeypatch.setattr(bot, "_run_rag", _rag)
    now = datetime.now(timezone.utc)
    qbot._handle_inbound("chat1", "first question", "Ann", "m1", now)
    qbot._handle_inbound("chat1", "second question", "Ann", "m2", now)

    t = threading.Thread(target=qbot._worker_loop, daemon=True)
    t.start()
    _drain(qbot._work_q)

    assert any("Something went wrong while looking this up." in h for h in qbot._sent)
    # The renderer does not HTML-escape — raw exception text must never reach a user.
    assert not any(secret in h for h in qbot._sent)
    # The worker survived the failure and drained the next job.
    assert any("See AUP." in h for h in qbot._sent)
    assert qbot._inflight == {}


def test_undelivered_answer_is_logged_and_the_worker_moves_on(monkeypatch, qbot, capsys):
    """A reply POST that failed even after retries must not leave the user in silence."""
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    answers = iter([
        {"answer": "UNDELIVERABLE", "citations": [], "escalation": {"needed": False}},
        {"answer": "second answer", "citations": [], "escalation": {"needed": False}},
    ])
    monkeypatch.setattr(bot, "_run_rag", lambda q: next(answers))

    def _send(chat_id, text, content_type="html", retry=False):
        qbot._sent.append(text)
        return None if "UNDELIVERABLE" in text else True  # transport gave up

    monkeypatch.setattr(qbot, "_send_message", _send)

    now = datetime.now(timezone.utc)
    qbot._handle_inbound("chat1", "first question", "Ann", "m1", now)
    qbot._handle_inbound("chat1", "second question", "Ann", "m2", now)

    t = threading.Thread(target=qbot._worker_loop, daemon=True)
    t.start()
    _drain(qbot._work_q)

    logged = capsys.readouterr().out
    assert "ERROR: answer not delivered" in logged
    assert "chat1" in logged and "first question" in logged
    # The worker drained the failed job and went on to the next one.
    assert any("second answer" in h for h in qbot._sent)
    assert qbot._inflight == {}


def test_saved_watermark_is_held_before_the_oldest_inflight_message(qbot, tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    old = datetime.now(timezone.utc) - timedelta(minutes=2)
    new = datetime.now(timezone.utc)
    qbot.last_check = new
    qbot.processed_messages = dict.fromkeys(["m_old", "m_done"])
    qbot._inflight = {"m_old": old}

    qbot._save_state()
    saved = json.loads((tmp_path / "bot_state.json").read_text())

    # Watermark rewound behind the queued message, so a restart re-delivers it.
    assert datetime.fromisoformat(saved["last_check"]) < old
    # ...and it is not in the processed set, which would otherwise skip it.
    assert "m_old" not in saved["processed_messages"]
    assert "m_done" in saved["processed_messages"]


def test_saved_watermark_is_last_check_when_nothing_inflight(qbot, tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    now = datetime.now(timezone.utc)
    qbot.last_check = now
    qbot.processed_messages = dict.fromkeys(["m1"])
    qbot._inflight = {}

    qbot._save_state()
    saved = json.loads((tmp_path / "bot_state.json").read_text())
    assert datetime.fromisoformat(saved["last_check"]) == now


def test_ensure_worker_is_idempotent_while_alive(qbot):
    """The headline invariant: exactly one worker, never more."""
    qbot._ensure_worker()
    first = qbot._worker
    qbot._ensure_worker()
    assert qbot._worker is first


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


def test_messages_are_enqueued_oldest_first(qbot, monkeypatch):
    """Graph returns newest-first; people must be answered in the order they asked."""
    monkeypatch.setattr(qbot, "_get_my_user_id", lambda: "me")
    future = datetime.now(timezone.utc) + timedelta(minutes=5)

    def _msg(n):
        stamp = (future + timedelta(seconds=n)).isoformat().replace("+00:00", "Z")
        return {"id": f"m{n}", "messageType": "message",
                "from": {"user": {"id": "someone", "displayName": "Ann"}},
                "createdDateTime": stamp, "body": {"content": f"question {n}"}}

    def _api(url, method="GET", json_data=None, retry=False):
        if url.endswith("/me/chats"):
            return {"value": [{"id": "chat1"}]}
        return {"value": [_msg(3), _msg(1), _msg(2)]}  # newest-first, as Graph returns

    monkeypatch.setattr(qbot, "_api_request", _api)

    qbot.process_new_messages()

    queued = [qbot._work_q.get_nowait()[1] for _ in range(qbot._work_q.qsize())]
    assert queued == ["question 1", "question 2", "question 3"]
