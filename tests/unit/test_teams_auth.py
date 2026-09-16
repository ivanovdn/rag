import json
import threading
import time
from datetime import datetime, timedelta, timezone

import channels.teams.auth as auth


def test_concurrent_callers_refresh_the_token_once(tmp_path, monkeypatch):
    """The poll thread and the RAG worker both call get_access_token(); an expired
    token must be refreshed once, not once per thread."""
    token_file = tmp_path / "refresh_token.json"
    token_file.write_text(json.dumps({"refresh_token": "seed"}))
    monkeypatch.setattr(auth, "_TOKEN_FILE", token_file)
    refresher = auth.TokenRefresher()

    calls = []

    def slow_refresh():
        calls.append(1)
        time.sleep(0.05)  # long enough for the second thread to arrive mid-refresh
        refresher.access_token = "tok"
        refresher.token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        return refresher.access_token

    monkeypatch.setattr(refresher, "_refresh_access_token", slow_refresh)

    threads = [threading.Thread(target=refresher.get_access_token) for _ in range(2)]
    for t in threads:
        t.start()
    # Bounded: this test exercises locking, so a lock bug must fail it, not hang the suite.
    for t in threads:
        t.join(timeout=5)
    assert not any(t.is_alive() for t in threads), "get_access_token did not return — deadlock?"

    assert calls == [1], f"token refreshed {len(calls)} times"


def test_failed_refresh_cools_off_instead_of_retrying_every_call(tmp_path, monkeypatch):
    """A dead credential must not be retried on every Graph call — _get_headers() runs
    once per request, and each attempt costs a ~10s POST under the lock."""
    token_file = tmp_path / "refresh_token.json"
    token_file.write_text(json.dumps({"refresh_token": "seed"}))
    monkeypatch.setattr(auth, "_TOKEN_FILE", token_file)
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
