import json
import pathlib
import threading
from datetime import datetime, timedelta, timezone

import channels.teams.auth as auth


class _WatchedLock:
    """A Lock that reports when a second caller actually blocks on it.

    Lets the test prove the two threads overlapped instead of inferring it from a
    sleep — a sleep-based version can only ever false-pass, so a lock regression
    could slip through on a loaded CI box.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.contended.set()  # someone holds it; we are about to block
            self._lock.acquire()
        return self

    def __exit__(self, *exc_info):
        self._lock.release()
        return False


def test_concurrent_callers_refresh_the_token_once(tmp_path, monkeypatch):
    """The poll thread and the RAG worker both call get_access_token(); an expired
    token must be refreshed once, not once per thread."""
    token_file = tmp_path / "refresh_token.json"
    token_file.write_text(json.dumps({"refresh_token": "seed"}))
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)
    refresher = auth.TokenRefresher()

    watched = _WatchedLock()
    monkeypatch.setattr(refresher, "_lock", watched)

    calls = []
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    def slow_refresh():
        calls.append(1)
        refresh_started.set()
        assert release_refresh.wait(timeout=5), "test never released the refresh"
        refresher.access_token = "tok"
        refresher.token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        return refresher.access_token

    monkeypatch.setattr(refresher, "_refresh_access_token", slow_refresh)

    first = threading.Thread(target=refresher.get_access_token)
    first.start()
    assert refresh_started.wait(timeout=5), "the first refresh never started"

    second = threading.Thread(target=refresher.get_access_token)
    second.start()
    # Deterministic: proceed only once the second caller is provably blocked on the
    # lock. If get_access_token did not take the lock at all, this fails here.
    assert watched.contended.wait(timeout=5), "second caller never contended for the lock"
    release_refresh.set()

    # Bounded: this test exercises locking, so a lock bug must fail it, not hang the suite.
    for t in (first, second):
        t.join(timeout=5)
    assert not any(t.is_alive() for t in (first, second)), "get_access_token did not return — deadlock?"

    assert calls == [1], f"token refreshed {len(calls)} times"


def test_failed_refresh_cools_off_instead_of_retrying_every_call(tmp_path, monkeypatch):
    """A dead credential must not be retried on every Graph call — _get_headers() runs
    once per request, and each attempt costs a ~10s POST under the lock."""
    token_file = tmp_path / "refresh_token.json"
    token_file.write_text(json.dumps({"refresh_token": "seed"}))
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)
    refresher = auth.TokenRefresher()

    calls = []

    def failing_refresh():
        calls.append(1)
        refresher._retry_refresh_after = (
            datetime.now(timezone.utc) + timedelta(seconds=auth._TOKEN_REFRESH_COOLDOWN)
        )
        return None

    monkeypatch.setattr(refresher, "_refresh_access_token", failing_refresh)

    for _ in range(5):
        assert refresher.get_access_token() is None
    assert calls == [1], f"refresh attempted {len(calls)} times during the cool-off"

    # Once the cool-off passes, it tries again.
    refresher._retry_refresh_after = datetime.now(timezone.utc) - timedelta(seconds=1)
    refresher.get_access_token()
    assert len(calls) == 2


def test_refresh_token_file_is_written_atomically(tmp_path, monkeypatch):
    """A container kill mid-write must not be able to leave an empty or truncated
    file: Azure invalidates the refresh token on every use, so this file is the only
    surviving copy of the live credential and losing it means an interactive
    device-code sign-in (scripts/get_refresh_token.py) to recover."""
    token_file = tmp_path / "refresh_token.json"
    token_file.write_text(json.dumps({"refresh_token": "seed"}))
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)
    refresher = auth.TokenRefresher()
    refresher.refresh_token = "fake-refresh-token"

    replaced = {}
    real_replace = auth.os.replace

    def _spy(src, dst):
        # At the moment of the rename the target must still hold the previous, valid
        # content — never a partially written file.
        replaced["src"] = str(src)
        replaced["dst"] = str(dst)
        return real_replace(src, dst)

    monkeypatch.setattr(auth.os, "replace", _spy)
    refresher._save_refresh_token()

    assert replaced["dst"] == str(token_file), "the final write must be a rename onto the target"
    assert replaced["src"] != str(token_file), "content must be staged in a separate file"
    assert pathlib.Path(replaced["src"]).parent == token_file.parent, \
        "the temp file must share a directory with the target, or the rename is not atomic"
    assert json.loads(token_file.read_text())["refresh_token"] == "fake-refresh-token"
