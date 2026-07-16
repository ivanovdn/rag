import types

import rag.vector_store as vs
from config import settings


class _FakeClient:
    def __init__(self):
        self.query_calls = []
        self.scroll_calls = []

    def query_points(self, collection_name, query, limit, with_payload):
        self.query_calls.append(collection_name)
        return types.SimpleNamespace(points=[])

    def scroll(self, collection_name, limit, with_payload):
        self.scroll_calls.append(collection_name)
        pt = types.SimpleNamespace(payload={"name": "Docker", "status": "allowed"})
        return ([pt], None)


def test_search_vectors_defaults_to_policy_collection(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    vs.search_vectors([0.1, 0.2], top_k=3)
    assert fake.query_calls == [settings.qdrant_collection]


def test_search_vectors_honors_collection_name(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    vs.search_vectors([0.1, 0.2], top_k=3, collection_name="software_registry")
    assert fake.query_calls == ["software_registry"]


def test_scroll_all_returns_payloads(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    rows = vs.scroll_all("software_registry")
    assert rows == [{"name": "Docker", "status": "allowed"}]
    assert fake.scroll_calls == ["software_registry"]
