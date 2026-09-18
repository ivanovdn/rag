"""Graph transport retries: a blip must not cost a user their answer, and a 4xx
must not cost a poll cycle."""
import pytest
import requests

import channels.teams.bot as bot


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = {"ok": True} if payload is None else payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._payload


@pytest.fixture
def tbot(monkeypatch, tmp_path):
    """A TeamsBot with mocked transport whose retry backoffs don't really sleep."""
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    monkeypatch.setattr(bot.time, "sleep", lambda seconds: None)
    b = bot.TeamsBot(token_refresher=object())
    monkeypatch.setattr(b, "_get_headers", lambda: {"Authorization": "Bearer test"})
    return b


def test_transient_failure_is_retried_and_then_succeeds(tbot, monkeypatch):
    calls = {"n": 0}

    def _post(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("graph blip")
        return _FakeResponse(201, {"id": "delivered"})

    monkeypatch.setattr(bot.requests, "post", _post)

    assert tbot._api_request("https://graph/test", method="POST", json_data={}, retry=True) == {"id": "delivered"}
    assert calls["n"] == 2, "a transient failure must be retried"


def test_timeout_is_retried_until_the_attempts_run_out(tbot, monkeypatch):
    calls = {"n": 0}

    def _get(url, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.Timeout("graph is not answering")

    monkeypatch.setattr(bot.requests, "get", _get)

    assert tbot._api_request("https://graph/test", retry=True) is None
    assert calls["n"] == len(bot._GRAPH_RETRY_BACKOFFS) + 1


def test_server_error_is_retried(tbot, monkeypatch):
    calls = {"n": 0}

    def _get(url, **kwargs):
        calls["n"] += 1
        return _FakeResponse(503)

    monkeypatch.setattr(bot.requests, "get", _get)

    assert tbot._api_request("https://graph/test", retry=True) is None
    assert calls["n"] == len(bot._GRAPH_RETRY_BACKOFFS) + 1, "5xx is transient — retry it"


def test_client_error_is_not_retried(tbot, monkeypatch):
    """A deleted chat or a bad payload will not come back; retrying wastes a poll cycle."""
    calls = {"n": 0}

    def _post(url, **kwargs):
        calls["n"] += 1
        return _FakeResponse(404)

    monkeypatch.setattr(bot.requests, "post", _post)

    assert tbot._api_request("https://graph/test", method="POST", json_data={}, retry=True) is None
    assert calls["n"] == 1, "a 4xx must not be retried even with retry on"


def test_poll_thread_calls_are_single_attempt(tbot, monkeypatch):
    """The poll loop must never stall: retry is opt-in, and its calls don't opt in.

    A transient miss here is self-healing — the next cycle re-reads the same chats.
    """
    calls = {"n": 0}

    def _get(url, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.Timeout("graph is slow")

    monkeypatch.setattr(bot.requests, "get", _get)

    assert tbot._api_request("https://graph/me/chats") is None
    assert calls["n"] == 1, "the poll thread's calls must not be retried"


def test_ack_is_single_attempt_but_the_answer_send_is_retried(tbot, monkeypatch):
    calls = {"n": 0}

    def _post(url, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("blip")

    monkeypatch.setattr(bot.requests, "post", _post)

    tbot._send_message("chat1", "<p>ack</p>")  # poll thread — default
    assert calls["n"] == 1, "the ack must not stall the poll loop"

    calls["n"] = 0
    tbot._send_message("chat1", "<p>answer</p>", retry=True)  # worker thread
    assert calls["n"] == len(bot._GRAPH_RETRY_BACKOFFS) + 1
