"""Fail fast when BM25 is on but the collection has no sparse vector.

Qdrant answers that query with "Not existing vector name error", which is not
transient — so without this check every question comes back as a content
escalation and the bot looks healthy while finding nothing. It raises rather
than auto-disabling BM25 on purpose: quiet degradation is the failure class
this migration exists to remove.
"""

import pytest

import rag.vector_store as vs


class _Params:
    def __init__(self, sparse):
        self.sparse_vectors = sparse


class _Config:
    def __init__(self, sparse):
        self.params = _Params(sparse)


class _Info:
    def __init__(self, sparse):
        self.config = _Config(sparse)


class _FakeClient:
    def __init__(self, sparse):
        self._sparse = sparse
        self.calls = 0

    def get_collection(self, name):
        self.calls += 1
        return _Info(self._sparse)


def test_preflight_passes_when_the_sparse_vector_exists(monkeypatch):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient({"bm25": object()}))

    vs.preflight_sparse_config()  # must not raise


def test_preflight_refuses_a_collection_without_the_sparse_vector(monkeypatch):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs.settings, "qdrant_collection", "compliance_policies")
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient({}))

    with pytest.raises(RuntimeError) as exc:
        vs.preflight_sparse_config()

    message = str(exc.value)
    assert "compliance_policies" in message
    assert "bm25" in message
    assert "migrate_collection.py" in message  # tells the operator what to do


def test_preflight_refuses_when_sparse_vectors_is_none(monkeypatch):
    """A collection created before this migration reports None, not {}."""
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient(None))

    with pytest.raises(RuntimeError):
        vs.preflight_sparse_config()


def test_preflight_does_not_call_qdrant_when_bm25_is_disabled(monkeypatch):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)
    client = _FakeClient({})
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.preflight_sparse_config()

    assert client.calls == 0


def test_the_bot_entry_point_runs_the_preflight():
    from pathlib import Path

    source = Path("scripts/start_teams_bot.py").read_text(encoding="utf-8")
    assert "preflight_sparse_config()" in source
    # It must run before the bot is constructed, not after.
    assert source.index("preflight_sparse_config()") < source.index("TeamsBot(")
