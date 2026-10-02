"""The sparse-schema guard: the shared assertion and its query-side wrapper.

assert_sparse_vector is the single definition; preflight_sparse_config is the
query-side caller and is gated on bm25_enabled, while the write side
(ingest/pipeline.py, scripts/ingest_all.py) calls the assertion directly and is
NOT gated — upsert_chunks writes a sparse vector whatever that flag says.

Qdrant answers a sparse query against a collection that has no sparse vector
with "Not existing vector name error", which is not transient — so without the
query-side check every question comes back as a content escalation and the bot
looks healthy while finding nothing. It raises rather than auto-disabling BM25
on purpose: quiet degradation is the failure class this migration exists to
remove.
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
        self.names: list[str] = []

    def get_collection(self, name):
        self.calls += 1
        self.names.append(name)
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

# --- the shared assertion ----------------------------------------------------
#
# Ungated on purpose. The sparse WRITE is unconditional (upsert_chunks), so its
# guard has to be too: gating this on bm25_enabled is exactly what let a failed
# re-ingest wipe a document out of a live collection.


def test_the_assertion_passes_on_a_sparse_enabled_collection(monkeypatch):
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient({"bm25": object()}))

    vs.assert_sparse_vector("compliance_policies_v2", "Ingest writes a sparse vector.")


def test_the_assertion_refuses_a_pre_migration_collection(monkeypatch):
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient({}))

    with pytest.raises(RuntimeError) as exc:
        vs.assert_sparse_vector("compliance_policies", "Ingest writes a sparse vector.")

    message = str(exc.value)
    assert "Ingest writes a sparse vector." in message  # the caller's framing
    assert "compliance_policies" in message  # which collection
    assert "bm25" in message  # which vector is missing
    assert "migrate_collection.py" in message  # what to do about it


def test_the_assertion_ignores_the_query_side_flag(monkeypatch):
    """BM25_ENABLED off must NOT excuse a collection from the check: the write
    happens regardless, so the guard has to run regardless."""
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: _FakeClient({}))

    with pytest.raises(RuntimeError):
        vs.assert_sparse_vector("compliance_policies", "Ingest writes a sparse vector.")


def test_the_assertion_checks_the_collection_it_is_given(monkeypatch):
    """Not settings.qdrant_collection — migrate_collection.py and a re-ingest can
    legitimately target different collections."""
    client = _FakeClient({"bm25": object()})
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.assert_sparse_vector("some_other_collection", "why")

    assert client.names == ["some_other_collection"]


def test_the_preflight_delegates_rather_than_repeating_the_check(monkeypatch):
    """One definition of the check, two callers. Duplicating it is how the two
    messages and the two conditions drift apart."""
    seen = []
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)
    monkeypatch.setattr(vs.settings, "qdrant_collection", "compliance_policies")
    monkeypatch.setattr(
        vs, "assert_sparse_vector", lambda name, reason: seen.append((name, reason))
    )

    vs.preflight_sparse_config()

    assert [name for name, _ in seen] == ["compliance_policies"]
    assert "BM25_ENABLED=false" in seen[0][1]  # the query-side-only remedy


# --- entry points ------------------------------------------------------------


def test_the_ingest_entry_point_guards_the_schema_before_ingesting():
    from pathlib import Path

    source = Path("scripts/ingest_all.py").read_text(encoding="utf-8")
    assert "assert_sparse_vector(" in source
    # Before anything is parsed or written, not after the first document.
    assert source.index("assert_sparse_vector(") < source.index("ingest_files(paths")


def test_the_eval_entry_points_run_the_preflight():
    """A gate run with BM25 on against a collection with no sparse vector fails
    every query non-transiently, so hit_evaluator reads near zero — which looks
    exactly like the sparse half failing on its merits, and sends the operator
    to the D10 fallback over a one-line env mistake."""
    from pathlib import Path

    for path in ("eval/run_experiment.py", "scripts/run_eval.py"):
        source = Path(path).read_text(encoding="utf-8")
        assert "preflight_sparse_config()" in source, path
