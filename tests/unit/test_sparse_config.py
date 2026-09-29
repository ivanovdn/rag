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
