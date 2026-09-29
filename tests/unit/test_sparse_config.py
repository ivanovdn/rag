"""Config and client wiring for Qdrant-native sparse vectors.

cloud_inference is the load-bearing one. Its name is misleading: it means
"do not encode locally, send the text to the server". The client's default is
False, which means "encode locally via fastembed". Server-side BM25 works today
only because fastembed is not installed — anyone installing it for an unrelated
reason would silently move encoding from the server to the client, changing
retrieval with no error and no log line.
"""

import rag.vector_store as vs
from config import settings
from qdrant_client.models import Distance, Modifier


def test_bm25_avg_len_defaults_to_the_measured_corpus_average():
    # Bound to a local first: a failing assert on settings.<attr> would embed the
    # full Settings repr, which carries real .env secrets, into pytest output.
    avg_len = settings.bm25_avg_len
    assert avg_len == 50.0


def test_qdrant_client_pins_server_side_inference(monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vs, "QdrantClient", _FakeClient)
    monkeypatch.setattr(vs, "_client", None)  # module-level singleton

    vs.get_qdrant_client()

    assert captured["cloud_inference"] is True


class _RecordingClient:
    """Records create_collection / create_payload_index calls; creates nothing."""

    def __init__(self, exists=False):
        self._exists = exists
        self.created = None
        self.indexes = []

    def collection_exists(self, name):
        return self._exists

    def create_collection(self, **kwargs):
        self.created = kwargs

    def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs)


def test_init_collection_declares_the_sparse_bm25_vector(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.init_collection()

    sparse = client.created["sparse_vectors_config"]
    assert set(sparse) == {"bm25"}
    assert sparse["bm25"].modifier == Modifier.IDF


def test_init_collection_keeps_the_dense_vector_unnamed_and_768(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(vs.settings, "qdrant_vector_dim", 768)

    vs.init_collection()

    dense = client.created["vectors_config"]
    # A bare VectorParams (not a dict) is what makes the vector unnamed. Naming it
    # would force every existing search call to specify `using=`.
    assert dense.size == 768
    assert dense.distance == Distance.COSINE


def test_init_collection_targets_a_named_collection(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.init_collection("compliance_policies_v2")

    assert client.created["collection_name"] == "compliance_policies_v2"
    assert all(i["collection_name"] == "compliance_policies_v2" for i in client.indexes)
    assert len(client.indexes) == 6


def test_init_collection_is_a_noop_when_the_collection_exists(monkeypatch):
    client = _RecordingClient(exists=True)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)

    vs.init_collection()

    assert client.created is None
