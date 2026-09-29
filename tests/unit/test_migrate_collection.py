"""Point-copying migration into a sparse-enabled collection.

Copying rather than re-ingesting keeps the dense vectors byte-identical, which
is what makes the eval gate meaningful: any score delta is attributable to the
sparse half alone. Nothing here touches the network.
"""

from pathlib import Path

import pytest

import scripts.migrate_collection as migrate_mod


class _SrcPoint:
    def __init__(self, pid, vector, payload):
        self.id = pid
        self.vector = vector
        self.payload = payload


class _Count:
    def __init__(self, count):
        self.count = count


class _FakeClient:
    def __init__(self, points, exists=False):
        self._points = list(points)
        self._exists = exists
        self.upserted = []
        self.created = []

    def count(self, collection_name, exact=True):
        return _Count(len(self._points))

    def collection_exists(self, name):
        return self._exists

    def scroll(self, collection_name, limit=500, offset=None, **kwargs):
        return list(self._points), None

    def upsert(self, collection_name, points, wait=True):
        self.upserted.append((collection_name, points))


class _TwoPageClient(_FakeClient):
    """Splits its points into two scroll pages instead of one, exercising the
    while-True pagination loop that a single-page _FakeClient never reaches."""

    def __init__(self, points, split):
        super().__init__(points)
        self._split = split
        self._scroll_calls = 0

    def scroll(self, collection_name, limit=500, offset=None, **kwargs):
        self._scroll_calls += 1
        if self._scroll_calls == 1:
            return self._points[:self._split], "more"
        return self._points[self._split:], None


class _SparseParams:
    def __init__(self, sparse):
        self.sparse_vectors = sparse


class _SparseConfig:
    def __init__(self, sparse):
        self.params = _SparseParams(sparse)


class _CollectionInfo:
    def __init__(self, sparse):
        self.config = _SparseConfig(sparse)


class _VerifyClient:
    """Models source and target as two distinct named collections, unlike
    _FakeClient (which treats every collection_name as the same store) —
    verify() compares the two, so a test needs them able to actually differ."""

    def __init__(self, source_points, target_points, sparse=None):
        self._collections = {"src": list(source_points), "dst": list(target_points)}
        self._sparse = {"bm25": object()} if sparse is None else sparse

    def count(self, collection_name, exact=True):
        return _Count(len(self._collections[collection_name]))

    def get_collection(self, name):
        return _CollectionInfo(self._sparse)

    def scroll(self, collection_name, limit=500, offset=None, **kwargs):
        return list(self._collections[collection_name]), None

    def retrieve(self, collection_name, ids, **kwargs):
        by_id = {p.id: p for p in self._collections[collection_name]}
        return [by_id[i] for i in ids if i in by_id]


@pytest.fixture
def client(monkeypatch):
    fake = _FakeClient([
        _SrcPoint("c1", [0.1, 0.2], {"text": "backups are kept ninety days"}),
        _SrcPoint("c2", [0.3, 0.4], {"text": "laptops must be encrypted"}),
    ])
    monkeypatch.setattr(migrate_mod, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(migrate_mod, "init_collection", lambda name: fake.created.append(name))
    return fake


def test_dry_run_writes_nothing(client, capsys):
    copied = migrate_mod.migrate("src", "dst", dry_run=True)

    assert copied == 0
    assert client.upserted == []
    assert client.created == []
    assert "2 points" in capsys.readouterr().out


def test_migration_preserves_id_payload_and_dense_vector(client):
    migrate_mod.migrate("src", "dst", dry_run=False)

    _, points = client.upserted[0]
    first = points[0]
    assert first.id == "c1"
    assert first.payload == {"text": "backups are kept ninety days"}
    assert first.vector[""] == [0.1, 0.2]


def test_migration_adds_a_sparse_document_from_the_payload_text(client):
    migrate_mod.migrate("src", "dst", dry_run=False)

    _, points = client.upserted[0]
    sparse = points[0].vector["bm25"]
    assert sparse.model == "qdrant/bm25"
    assert sparse.text == "backups are kept ninety days"


def test_migration_creates_the_target_from_the_shared_schema(client):
    """init_collection is the single schema definition; duplicating it here is how
    the two would drift."""
    migrate_mod.migrate("src", "dst", dry_run=False)

    assert client.created == ["dst"]


def test_a_named_dense_vector_round_trips(monkeypatch):
    """Re-running against an already-migrated collection: qdrant-client returns a
    dict there, not a bare list."""
    fake = _FakeClient([_SrcPoint("c1", {"": [0.5], "bm25": object()}, {"text": "t"})])
    monkeypatch.setattr(migrate_mod, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(migrate_mod, "init_collection", lambda name: None)

    migrate_mod.migrate("src", "dst", dry_run=False)

    assert fake.upserted[0][1][0].vector[""] == [0.5]


def test_migration_copies_every_point_across_two_scroll_pages(monkeypatch):
    """migrate()'s while-True scroll loop must keep paging until offset is
    None — the shared `client` fixture returns everything in a single page
    and gives that loop zero signal. This is the actual bulk-copy mechanism
    for the ~1602 real points this script runs against."""
    points = [_SrcPoint(f"c{i}", [float(i)], {"text": f"t{i}"}) for i in range(4)]
    fake = _TwoPageClient(points, split=2)
    monkeypatch.setattr(migrate_mod, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(migrate_mod, "init_collection", lambda name: None)

    copied = migrate_mod.migrate("src", "dst", dry_run=False)

    assert copied == 4
    copied_ids = [p.id for _, batch in fake.upserted for p in batch]
    assert sorted(copied_ids) == ["c0", "c1", "c2", "c3"]
    assert len(copied_ids) == len(set(copied_ids))  # nothing duplicated


def test_migration_sub_batches_a_page_larger_than_upsert_batch(monkeypatch):
    """UPSERT_BATCH sub-batches one scroll page into multiple upsert calls.
    Shrinking the constant is cheaper than building 100+ fixture points to
    cross the real default and exercises the same inner loop either way."""
    monkeypatch.setattr(migrate_mod, "UPSERT_BATCH", 2)
    points = [_SrcPoint(f"c{i}", [float(i)], {"text": f"t{i}"}) for i in range(5)]
    fake = _FakeClient(points)
    monkeypatch.setattr(migrate_mod, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(migrate_mod, "init_collection", lambda name: None)

    copied = migrate_mod.migrate("src", "dst", dry_run=False)

    assert copied == 5
    assert len(fake.upserted) == 3  # batches of 2, 2, 1 — none dropped, none duplicated
    copied_ids = [p.id for _, batch in fake.upserted for p in batch]
    assert sorted(copied_ids) == ["c0", "c1", "c2", "c3", "c4"]
    assert len(copied_ids) == len(set(copied_ids))


def test_verify_confirms_identical_dense_vectors_and_payloads(monkeypatch, capsys):
    """Count parity, sparse-schema presence, and non-empty sparse vectors on a
    sample would all pass even if dense vectors were zeroed or scrambled —
    this is the check that actually protects the premise that copying (rather
    than re-ingesting) leaves the dense half byte-identical."""
    src_points = [
        _SrcPoint("c1", [0.1, 0.2], {"text": "backups are kept ninety days"}),
        _SrcPoint("c2", [0.3, 0.4], {"text": "laptops must be encrypted"}),
    ]
    dst_points = [
        _SrcPoint("c1", {"": [0.1, 0.2], "bm25": object()}, {"text": "backups are kept ninety days"}),
        _SrcPoint("c2", {"": [0.3, 0.4], "bm25": object()}, {"text": "laptops must be encrypted"}),
    ]
    fake = _VerifyClient(src_points, dst_points)
    monkeypatch.setattr(migrate_mod, "get_qdrant_client", lambda: fake)

    ok = migrate_mod.verify("src", "dst")

    assert ok is True
    out = capsys.readouterr().out
    assert "dense vectors:  2/2 identical" in out
    assert "payloads:       2/2 identical" in out


def test_verify_catches_a_dense_vector_mismatch(monkeypatch, capsys):
    """A migration that silently zeroed or mis-copied a dense vector must fail
    verify() and be visible in its report, even though count parity, the
    sparse schema, and sparse presence on the sample all still pass."""
    src_points = [
        _SrcPoint("c1", [0.1, 0.2], {"text": "backups are kept ninety days"}),
        _SrcPoint("c2", [0.3, 0.4], {"text": "laptops must be encrypted"}),
    ]
    dst_points = [
        _SrcPoint("c1", {"": [0.0, 0.0], "bm25": object()}, {"text": "backups are kept ninety days"}),
        _SrcPoint("c2", {"": [0.3, 0.4], "bm25": object()}, {"text": "laptops must be encrypted"}),
    ]
    fake = _VerifyClient(src_points, dst_points)
    monkeypatch.setattr(migrate_mod, "get_qdrant_client", lambda: fake)

    ok = migrate_mod.verify("src", "dst")

    assert ok is False
    assert "dense vectors:  1/2 identical" in capsys.readouterr().out


def test_init_observability_runs_after_arg_parsing_not_before():
    """argparse exits from inside parse_args() itself on --help or a usage
    error, so init_observability() must run after parse_args(), never as the
    literal first statement of main() — otherwise a bare --help would
    register Phoenix and attempt a network export before help text prints."""
    source = Path("scripts/migrate_collection.py").read_text(encoding="utf-8")
    assert source.index("args = parser.parse_args()") < source.index("init_observability()")
