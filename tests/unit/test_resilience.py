import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.resilience import is_transient, retry_transient, RETRY_BACKOFFS


# --- is_transient ---

def test_httpx_connect_and_timeout_are_transient():
    assert is_transient(httpx.ConnectError("x"))
    assert is_transient(httpx.ConnectTimeout("x"))
    assert is_transient(httpx.ReadTimeout("x"))
    assert is_transient(httpx.PoolTimeout("x"))


def test_httpx_5xx_is_transient_4xx_is_not():
    req = httpx.Request("POST", "http://x")
    resp5 = httpx.Response(503, request=req)
    resp4 = httpx.Response(400, request=req)
    assert is_transient(httpx.HTTPStatusError("x", request=req, response=resp5))
    assert not is_transient(httpx.HTTPStatusError("x", request=req, response=resp4))


def test_qdrant_errors_classified():
    assert is_transient(ResponseHandlingException("conn refused"))
    assert is_transient(UnexpectedResponse(status_code=502, reason_phrase="", content=b"", headers=httpx.Headers()))
    assert not is_transient(UnexpectedResponse(status_code=404, reason_phrase="", content=b"", headers=httpx.Headers()))


def test_logic_errors_are_not_transient():
    assert not is_transient(ValueError("bad"))
    assert not is_transient(KeyError("missing"))
    assert not is_transient(RuntimeError("boom"))


# --- retry_transient ---

def test_returns_first_success_no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: slept.append(s))
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        return "ok"
    assert retry_transient(fn) == "ok"
    assert calls["n"] == 1
    assert slept == []


def test_retries_transient_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: slept.append(s))
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("down")
        return "recovered"
    assert retry_transient(fn) == "recovered"
    assert calls["n"] == 3
    assert slept == [0.5, 1.0]  # slept before the 2nd and 3rd attempts


def test_exhausts_and_reraises_on_persistent_transient(monkeypatch):
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: None)
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise httpx.ConnectError("still down")
    with pytest.raises(httpx.ConnectError):
        retry_transient(fn)
    assert calls["n"] == len(RETRY_BACKOFFS) + 1  # initial + one retry per backoff


def test_non_transient_reraises_immediately_without_retry(monkeypatch):
    slept = []
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: slept.append(s))
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise ValueError("logic bug")
    with pytest.raises(ValueError):
        retry_transient(fn)
    assert calls["n"] == 1
    assert slept == []
