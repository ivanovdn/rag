"""Point-copying migration into a sparse-enabled collection.

Copying rather than re-ingesting keeps the dense vectors byte-identical, which
is what makes the eval gate meaningful: any score delta is attributable to the
sparse half alone. Nothing here touches the network.
"""

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
