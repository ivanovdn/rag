"""Transient-failure classification and retry for backend (httpx/Qdrant) calls.

A "transient" failure is a backend being briefly unreachable or slow (connection
errors, timeouts, HTTP 5xx). Logic errors (4xx, validation, parse, bugs) are NOT
transient and must surface immediately rather than being retried or masked.
"""

import time
from typing import Callable, TypeVar

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

# Optional LLM-client exception types — present when the agent runs; guarded so
# this module stays importable in minimal environments (e.g. unit-test-only).
try:
    from ollama import ResponseError as _OllamaResponseError
except Exception:  # pragma: no cover
    _OllamaResponseError = None
try:
    import openai as _openai
except Exception:  # pragma: no cover
    _openai = None

T = TypeVar("T")

RETRY_BACKOFFS: tuple[float, ...] = (0.5, 1.0, 2.0)

# Connection/timeout failures across the clients we use:
#   - httpx (embeddings, reranker, qdrant transport)
#   - builtins ConnectionError/TimeoutError: the `ollama` client re-wraps httpx
#     connect/timeout failures into these
_TRANSIENT_TYPES = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
    httpx.WriteTimeout,
    ConnectionError,  # builtins — ollama connect-refused
    TimeoutError,     # builtins — ollama/socket timeout
)


def _is_5xx(status) -> bool:
    return status is not None and 500 <= status < 600


def is_transient(exc: BaseException) -> bool:
    """True if exc represents a transient backend failure (retry-worthy).

    Covers connection/timeout errors and HTTP 5xx across every client in the
    pipeline (httpx, qdrant, ollama, openai). 4xx and logic errors are NOT
    transient and must surface immediately rather than being retried/masked.
    """
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return _is_5xx(exc.response.status_code)
    if isinstance(exc, ResponseHandlingException):
        return True  # qdrant wraps connection/timeout failures here
    if isinstance(exc, UnexpectedResponse):
        return _is_5xx(exc.status_code)
    if _OllamaResponseError is not None and isinstance(exc, _OllamaResponseError):
        # ollama maps HTTP errors here; status_code=-1 means unknown → not transient
        return _is_5xx(getattr(exc, "status_code", None))
    if _openai is not None:
        if isinstance(exc, _openai.APIConnectionError):  # includes APITimeoutError
            return True
        if isinstance(exc, _openai.APIStatusError):
            return _is_5xx(getattr(exc, "status_code", None))
    return False


def retry_transient(fn: Callable[[], T], *, backoffs: tuple[float, ...] = RETRY_BACKOFFS) -> T:
    """Call fn(); retry on transient errors with the given backoff schedule.

    Up to len(backoffs)+1 attempts. Non-transient exceptions re-raise immediately.
    The last transient exception re-raises after the schedule is exhausted.
    """
    for delay in backoffs:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified below
            if not is_transient(exc):
                raise
            time.sleep(delay)
    # final attempt (no sleep after); let any exception propagate
    return fn()
