"""Rewritten queries ride in the SAME Qdrant request as the original.

Asserts on the request Qdrant receives; nothing reaches the network.
"""

import pytest

import rag.vector_store as vs


class _Resp:
    points = []


class _FakeClient:
    def __init__(self, scroll_pages=()):
        self.calls, self.scrolls = [], []
        self._pages = list(scroll_pages)

    def query_points(self, **kw):
        self.calls.append(kw)
        return _Resp()

    def scroll(self, **kw):
        self.scrolls.append(kw)
        return self._pages.pop(0)


@pytest.fixture
def client(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    return fake


def test_each_extra_query_adds_a_dense_and_a_sparse_prefetch(monkeypatch, client):
    monkeypatch.setattr(vs.settings, "bm25_enabled", True)

    vs.search_chunks("orig", [1.0], top_k=25, extra_queries=[("alt one", [2.0]), ("alt two", [3.0])])

    call = client.calls[0]
    pf = call["prefetch"]
    assert len(pf) == 6
    assert [p.query for p in pf[0::2]] == [[1.0], [2.0], [3.0]]
    assert [p.query.text for p in pf[1::2]] == ["orig", "alt one", "alt two"]
    assert {p.limit for p in pf} == {25} and call["limit"] == 25  # one knob sizes all
    assert call["query"].rrf.k == 60


def test_extra_queries_are_fused_even_with_bm25_off(monkeypatch, client):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)

    vs.search_chunks("orig", [1.0], top_k=6, extra_queries=[("alt", [2.0])])

    call = client.calls[0]
    assert [p.query for p in call["prefetch"]] == [[1.0], [2.0]]
    assert all(p.using is None for p in call["prefetch"])
    assert call["query"].rrf.k == 60


def test_without_extras_the_request_is_unchanged(monkeypatch, client):
    monkeypatch.setattr(vs.settings, "bm25_enabled", False)

    vs.search_chunks("orig", [1.0], top_k=6)

    call = client.calls[0]
    assert call["query"] == [1.0] and "prefetch" not in call


def test_policy_titles_are_distinct_sorted_and_read_once(monkeypatch):
    class _P:
        def __init__(self, t):
            self.payload = {"doc_title": t}

    fake = _FakeClient(scroll_pages=[([_P("B"), _P("A")], "next"), ([_P("A")], None)])
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(vs.settings, "qdrant_collection", "titles_test_collection")
    vs._policy_titles.cache_clear()

    assert vs.policy_titles() == ["A", "B"]
    assert vs.policy_titles() == ["A", "B"]
    assert len(fake.scrolls) == 2  # two pages, once — the second call is cached
    assert fake.scrolls[0]["with_payload"] == ["doc_title"]
    assert fake.scrolls[0]["with_vectors"] is False
