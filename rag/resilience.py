"""Transient-failure classification and retry for backend (httpx/Qdrant) calls.

A "transient" failure is a backend being briefly unreachable or slow (connection
errors, timeouts, HTTP 5xx). Logic errors (4xx, validation, parse, bugs) are NOT
transient and must surface immediately rather than being retried or masked.
"""

import time
from typing import Callable, TypeVar

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

T = TypeVar("T")

RETRY_BACKOFFS: tuple[float, ...] = (0.5, 1.0, 2.0)

_TRANSIENT_HTTPX = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
    httpx.WriteTimeout,
)


def is_transient(exc: BaseException) -> bool:
    """True if exc represents a transient backend failure (retry-worthy)."""
    if isinstance(exc, _TRANSIENT_HTTPX):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, ResponseHandlingException):
        # qdrant wraps connection/timeout failures in this
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and 500 <= exc.status_code < 600
    return False


def retry_transient(fn: Callable[[], T], *, backoffs: tuple[float, ...] = RETRY_BACKOFFS) -> T:
    """Call fn(); retry on transient errors with the given backoff schedule.

    Up to len(backoffs)+1 attempts. Non-transient exceptions re-raise immediately.
    The last transient exception re-raises after the schedule is exhausted.
    """
    for i, delay in enumerate(backoffs):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified below
            if not is_transient(exc):
                raise
            time.sleep(delay)
    # final attempt (no sleep after); let any exception propagate
    return fn()
