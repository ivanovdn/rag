"""Graceful shutdown on SIGTERM.

Without a handler, `docker compose restart` waits out Docker's full stop grace
and then SIGKILLs the bot (observed: `bot-1 exited with code 137`), so every
deploy is a hard crash. That costs real duplicate answers: _save_state runs once
per poll cycle, so an answered id stays non-durable for up to teams_poll_interval
seconds after the reply was sent, and a kill inside that window makes the next
start re-deliver an already-answered question.

These tests run the real run() loop in a thread with a hard join timeout — a
regression that spins, or a drain that stops being bounded, must fail the suite
rather than hang it (same reasoning as _drain in test_bot_queue.py).
"""

import json
import signal
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

import channels.teams.bot as bot


def _run_bounded(b, timeout=10):
    """run() in a thread; it must return on its own within `timeout`."""
    t = threading.Thread(target=b.run, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"run() did not return within {timeout}s"


@pytest.fixture(autouse=True)
def restore_sigterm():
    """run() installs a process-wide SIGTERM handler; give pytest its own back.

    signal.signal is captured here rather than looked up at teardown because one
    test monkeypatches it to simulate an off-main-thread registration failure.
    """
    install = signal.signal
    original = signal.getsignal(signal.SIGTERM)
    yield
    install(signal.SIGTERM, original)


@pytest.fixture
def sbot(monkeypatch, tmp_path):
    """A TeamsBot with network mocked, and its state/PID files inside tmp_path."""
    # Hermetic: never read or write the developer's real bot_state.json / bot.pid.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    monkeypatch.setattr(bot, "PID_FILE", tmp_path / "bot.pid")
    b = bot.TeamsBot(token_refresher=object())
    monkeypatch.setattr(b, "_send_message",
                        lambda chat_id, text, content_type="html", retry=False: True)
    yield b
    # Never leave a worker thread parked on the queue after a test.
    b._shutdown.set()
    b._work_q.put(bot._WORKER_STOP)


def test_sigterm_handler_sets_the_shutdown_event(sbot):
    assert sbot._install_signal_handler() is True
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    handler(signal.SIGTERM, None)  # what the interpreter does on delivery
    assert sbot._shutdown.is_set()


def test_shutdown_event_exits_the_main_loop(monkeypatch, sbot):
    """The loop must end on the event, not spin on `while True`."""
    cycles = {"n": 0}

    def _cycle():
        cycles["n"] += 1
        sbot._shutdown.set()  # as if SIGTERM landed during this cycle

    monkeypatch.setattr(sbot, "process_new_messages", _cycle)

    _run_bounded(sbot)

    assert cycles["n"] == 1


def test_a_shutdown_during_the_poll_wait_returns_promptly(monkeypatch, sbot):
    """The wait between cycles must be interruptible, not time.sleep().

    This is the part that actually delivers the benefit. With a plain sleep, a
    SIGTERM arriving during the idle interval sits out the whole 30s and is
    SIGKILLed anyway — the handler would be installed and do nothing. A long
    interval against a short join bound makes that regression fail loudly
    instead of passing slowly.
    """
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 30)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 30)
    cycled = threading.Event()
    monkeypatch.setattr(sbot, "process_new_messages", cycled.set)

    t = threading.Thread(target=sbot.run, daemon=True)
    t.start()
    assert cycled.wait(5), "the poll loop never ran a cycle"

    sbot._shutdown.set()  # the cycle is done: run() is in (or entering) the wait
    t.join(5)

    assert not t.is_alive(), (
        "run() did not wake from the poll wait within 5s of a 30s interval — "
        "the wait is not interruptible (regressed to time.sleep?)"
    )


def test_state_is_saved_exactly_once_on_shutdown(monkeypatch, sbot, tmp_path):
    saves = {"n": 0}
    real_save = sbot._save_state

    def _counting_save():
        saves["n"] += 1
        real_save()

    monkeypatch.setattr(sbot, "_save_state", _counting_save)
    # Stubbed so the cycle's own _save_state is not what we are counting.
    monkeypatch.setattr(sbot, "process_new_messages", sbot._shutdown.set)

    _run_bounded(sbot)

    assert saves["n"] == 1
    assert (tmp_path / "bot_state.json").exists()


def test_pid_lock_is_released_on_the_shutdown_path(monkeypatch, sbot, tmp_path):
    pid_file = tmp_path / "bot.pid"
    held = {}

    def _cycle():
        held["while_running"] = pid_file.exists()
        sbot._shutdown.set()

    monkeypatch.setattr(sbot, "process_new_messages", _cycle)

    _run_bounded(sbot)

    assert held["while_running"] is True
    assert not pid_file.exists()


def test_an_idle_worker_exits_on_the_shutdown_sentinel(sbot, capsys):
    """_worker_loop blocks forever in _work_q.get(); the sentinel is what frees it."""
    sbot._ensure_worker()
    worker = sbot._worker
    assert worker is not None and worker.is_alive()

    sbot._shutdown.set()
    sbot._graceful_shutdown("SIGTERM")

    assert not worker.is_alive()
    logged = capsys.readouterr().out
    assert "Worker drained in" in logged
    assert "State saved" in logged


def test_the_drain_is_bounded_and_unfinished_work_is_re_delivered(
    monkeypatch, sbot, tmp_path, capsys
):
    """A worker that never finishes must not hold shutdown past the grace.

    And the bound must stay safe by construction: the unfinished message is still
    in _inflight, so _save_state keeps its id out of processed_messages and holds
    the watermark behind it — the next start re-delivers it.
    """
    monkeypatch.setattr(bot.settings, "teams_shutdown_grace_seconds", 1)
    started = threading.Event()
    release = threading.Event()

    def _never_finishes(chat_id, text, sender_name="Unknown", queued_at=None):
        started.set()
        # Bounded so that an unbounded join() fails this test in ~10s rather than
        # hanging the suite forever.
        release.wait(10)
        return True

    monkeypatch.setattr(sbot, "_answer", _never_finishes)

    created = datetime.now(timezone.utc)
    sbot.last_check = created + timedelta(seconds=1)
    sbot._mark_processed("chat1:m1")  # the poll loop marks it before handing it off
    sbot._handle_inbound("chat1", "Can I install software?", "Ann", "chat1:m1", created)
    sbot._ensure_worker()
    assert started.wait(5), "the worker never picked up the job"

    sbot._shutdown.set()
    t0 = time.monotonic()
    sbot._graceful_shutdown("SIGTERM")
    elapsed = time.monotonic() - t0
    release.set()

    assert elapsed < 5, f"the drain was not bounded by the grace (took {elapsed:.1f}s)"
    logged = capsys.readouterr().out
    assert "WARNING: worker still busy" in logged

    saved = json.loads((tmp_path / "bot_state.json").read_text())
    assert "chat1:m1" not in saved["processed_messages"]
    assert datetime.fromisoformat(saved["last_check"]) < created


def test_a_signal_registration_failure_does_not_stop_the_bot(monkeypatch, sbot, capsys):
    """signal.signal raises ValueError off the main thread; the bot must still run."""
    def _off_main_thread(signum, handler):
        raise ValueError("signal only works in main thread of the main interpreter")

    monkeypatch.setattr(signal, "signal", _off_main_thread)
    monkeypatch.setattr(sbot, "process_new_messages", sbot._shutdown.set)

    _run_bounded(sbot)

    logged = capsys.readouterr().out
    assert "WARNING: could not install the SIGTERM handler" in logged
    assert "Waiting for messages" in logged  # it started anyway


# --- which exits drain -------------------------------------------------------
#
# SIGTERM and the error limit both drain: both are controlled decisions to stop,
# with an answer possibly mid-flight that can still be delivered. KeyboardInterrupt
# deliberately does not — that asymmetry is pinned below so it cannot rot into
# looking like an oversight.

def test_the_error_limit_exit_drains_and_saves_too(monkeypatch, sbot, tmp_path, capsys):
    """The failures are in the POLL loop (Graph unreachable) and say nothing about
    the worker, whose answer may well complete and reach the user. Abandoning it
    would make them wait for a restart for an answer the bot had already produced."""
    monkeypatch.setattr(bot.settings, "teams_max_consecutive_errors", 2)
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 0)  # no real backoff wait
    saves = {"n": 0}
    real_save = sbot._save_state

    def _counting_save():
        saves["n"] += 1
        real_save()

    monkeypatch.setattr(sbot, "_save_state", _counting_save)

    def _graph_is_down():
        raise RuntimeError("Graph unreachable")

    monkeypatch.setattr(sbot, "process_new_messages", _graph_is_down)

    _run_bounded(sbot)

    logged = capsys.readouterr().out
    assert "Too many consecutive errors, stopping bot" in logged
    assert "Shutdown requested (too many consecutive errors)" in logged  # names the exit
    assert "Worker drained in" in logged
    assert "State saved" in logged
    assert saves["n"] == 1
    assert (tmp_path / "bot_state.json").exists()
    assert sbot._worker is not None and not sbot._worker.is_alive()


def test_keyboard_interrupt_still_exits_without_draining(monkeypatch, sbot, tmp_path):
    """Ctrl-C is an operator asking to stop NOW; making local dev wait up to
    teams_shutdown_grace_seconds is a bad trade. Deliberately asymmetric."""
    saves = {"n": 0}
    monkeypatch.setattr(sbot, "_save_state", lambda: saves.__setitem__("n", saves["n"] + 1))

    def _ctrl_c():
        raise KeyboardInterrupt

    monkeypatch.setattr(sbot, "process_new_messages", _ctrl_c)

    _run_bounded(sbot)

    assert saves["n"] == 0
    assert not sbot._shutdown.is_set()
    assert not (tmp_path / "bot.pid").exists()  # the PID lock is still released
