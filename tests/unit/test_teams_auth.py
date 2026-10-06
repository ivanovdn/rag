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


# --- terminal credential failures ---------------------------------------------------
#
# 2026-10-06: the app registration's client secret expired. The one line that said so
# (AADSTS7000222) was reprinted every 30s between a 401 per poll cycle, and reading the
# log meant scrolling past dozens of identical "401 Client Error" lines to find it.
# Retrying cannot fix either of these; a human has to act, and the two need DIFFERENT
# actions, so each names its own.


class _FailingResponse:
    """A requests Response whose raise_for_status() fails the way AAD's does."""

    def __init__(self, body):
        self.text = body

    def raise_for_status(self):
        err = auth.requests.exceptions.HTTPError("401 Client Error: Unauthorized")
        err.response = self
        raise err


# Trimmed from the real 2026-10-06 response. The app id is a placeholder: baking this
# deployment's identifiers into the suite would make the test environment-specific for
# no gain.
_EXPIRED_SECRET_BODY = json.dumps(
    {
        "error": "invalid_client",
        "error_description": (
            "AADSTS7000222: The provided client secret keys for app "
            "'00000000-0000-0000-0000-000000000000' are expired. Visit the Azure portal "
            "to create new keys for your app: https://aka.ms/NewClientSecret"
        ),
        "error_codes": [7000222],
    }
)

_DEAD_REFRESH_TOKEN_BODY = json.dumps(
    {
        "error": "invalid_grant",
        "error_description": "AADSTS700082: The refresh token has expired due to inactivity.",
        "error_codes": [700082],
    }
)


def _refresher_that_fails_with(body, tmp_path, monkeypatch):
    token_file = tmp_path / "refresh_token.json"
    token_file.write_text(json.dumps({"refresh_token": "seed"}))
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)
    refresher = auth.TokenRefresher()
    monkeypatch.setattr(auth.requests, "post", lambda *a, **k: _FailingResponse(body))
    return refresher


def test_an_expired_client_secret_says_so_and_says_what_to_do(tmp_path, monkeypatch, capsys):
    refresher = _refresher_that_fails_with(_EXPIRED_SECRET_BODY, tmp_path, monkeypatch)

    refresher._refresh_access_token()

    out = capsys.readouterr().out
    assert "AADSTS7000222" in out
    assert "TEAMS_CLIENT_SECRET" in out
    # The action, not just the diagnosis: a reader who has never seen this must not
    # have to work out that the fix lives in the Azure portal.
    assert "Certificates & secrets" in out


def test_a_dead_refresh_token_points_at_the_device_code_recovery(tmp_path, monkeypatch, capsys):
    """The other terminal credential failure, and it needs the OPPOSITE action --
    a new client secret does nothing for a revoked refresh token. Naming only one
    of the two would send the next reader to the portal for the wrong thing."""
    refresher = _refresher_that_fails_with(_DEAD_REFRESH_TOKEN_BODY, tmp_path, monkeypatch)

    refresher._refresh_access_token()

    out = capsys.readouterr().out
    assert "scripts/get_refresh_token.py" in out
    assert "Certificates & secrets" not in out, "pointed at the client-secret fix instead"


def test_the_terminal_notice_is_printed_once_not_every_retry(tmp_path, monkeypatch, capsys):
    """Printing it per attempt would recreate the wall it exists to replace -- at a
    30s cooldown that is 120 copies an hour."""
    refresher = _refresher_that_fails_with(_EXPIRED_SECRET_BODY, tmp_path, monkeypatch)

    for _ in range(5):
        refresher._refresh_access_token()

    assert capsys.readouterr().out.count("AADSTS7000222 is terminal") == 1


def test_a_transient_failure_is_not_announced_as_terminal(tmp_path, monkeypatch, capsys):
    """An AAD blip or a 5xx must not tell the operator to go rotate a working
    credential. Crying wolf here is worse than silence: the recovery it names costs
    an interactive sign-in and invalidates the live token."""
    body = json.dumps({"error": "temporarily_unavailable", "error_codes": [50196]})
    refresher = _refresher_that_fails_with(body, tmp_path, monkeypatch)

    refresher._refresh_access_token()

    out = capsys.readouterr().out
    assert "terminal" not in out
    assert "Error refreshing token" in out, "the ordinary failure log must still happen"


def test_a_non_json_error_body_does_not_crash_the_refresh(tmp_path, monkeypatch, capsys):
    """A proxy or gateway in front of AAD returns HTML, not JSON. The refresh path
    runs under the auth lock held by the poll thread -- an exception escaping here
    would take down polling, which is strictly worse than the failure it describes."""
    refresher = _refresher_that_fails_with("<html>502 Bad Gateway</html>", tmp_path, monkeypatch)

    assert refresher._refresh_access_token() is None
    assert "Error refreshing token" in capsys.readouterr().out


def test_a_successful_refresh_rearms_the_notice(tmp_path, monkeypatch, capsys):
    """Once-per-process, not once-ever: a secret rotated in Azure while the bot runs
    would otherwise silence the signal for the next, different failure."""
    refresher = _refresher_that_fails_with(_EXPIRED_SECRET_BODY, tmp_path, monkeypatch)
    refresher._refresh_access_token()

    class _Ok:
        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "tok", "expires_in": 3600}

    monkeypatch.setattr(auth.requests, "post", lambda *a, **k: _Ok())
    refresher._refresh_access_token()

    monkeypatch.setattr(
        auth.requests, "post", lambda *a, **k: _FailingResponse(_EXPIRED_SECRET_BODY)
    )
    capsys.readouterr()
    refresher._refresh_access_token()

    assert "AADSTS7000222 is terminal" in capsys.readouterr().out
