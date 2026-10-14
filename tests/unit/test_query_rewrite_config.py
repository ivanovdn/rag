"""Settings and plumbing the query rewrite relies on (design 2026-10-09)."""

import pytest
from pydantic import ValidationError

import rag.embeddings as embeddings_mod
from config import Settings, inert_env_keys
from rag.agent import get_llm


def test_rewrite_is_off_unless_configured():
    assert Settings(_env_file=None).query_rewrite == "off"


def test_an_unknown_rewrite_mode_refuses_to_start():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, query_rewrite="mutli")


@pytest.mark.parametrize("mode", ["multi", "multi_titles", "hyde"])
def test_a_rewritten_search_is_never_judged_by_the_cosine_floor(mode):
    """Several queries are RRF-fused even with BM25 off: 0.016 against 0.45
    would turn every question into "no policy exists"."""
    s = Settings(_env_file=None, reranker_enabled=False, bm25_enabled=False, query_rewrite=mode)
    assert s.cosine_floor_applies is False


def test_the_cosine_floor_still_applies_to_a_plain_dense_search():
    s = Settings(_env_file=None, reranker_enabled=False, bm25_enabled=False, query_rewrite="off")
    assert s.cosine_floor_applies is True


def test_the_rewrite_timeout_is_inert_while_rewrite_is_off():
    s = Settings(_env_file=None, query_rewrite="off")
    reported = inert_env_keys(s, env_text="QUERY_REWRITE_TIMEOUT=5\n", environ={})
    assert len(reported) == 1
    assert reported[0].startswith("QUERY_REWRITE_TIMEOUT is set but does nothing here")


def test_the_rewrite_timeout_is_live_when_rewrite_is_on():
    s = Settings(_env_file=None, query_rewrite="multi")
    assert inert_env_keys(s, env_text="QUERY_REWRITE_TIMEOUT=5\n", environ={}) == []


def test_get_llm_honours_a_timeout_override(monkeypatch):
    monkeypatch.setattr("rag.agent.settings.llm_backend", "ollama")
    assert get_llm(timeout=20).request_timeout == 20.0


def test_get_llm_defaults_to_the_answer_timeout(monkeypatch):
    monkeypatch.setattr("rag.agent.settings.llm_backend", "ollama")
    from rag.agent import settings

    assert get_llm().request_timeout == float(settings.active_request_timeout)


def test_several_queries_cost_one_embedding_call(monkeypatch):
    calls = []

    def _fake(texts, prefix=""):
        calls.append((list(texts), prefix))
        return [[float(i)] for i, _ in enumerate(texts)]

    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(embeddings_mod.settings, "embedding_query_prefix", "Q: ")
    monkeypatch.setattr(embeddings_mod, "_ollama_embed", _fake)

    assert embeddings_mod.embed_queries(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]
    assert calls == [(["a", "b", "c"], "Q: ")]


def test_a_bounded_call_is_not_retried_by_the_openai_client(monkeypatch):
    """OpenAILike defaults to max_retries=3 plus a tenacity retry on timeouts: a
    20s rewrite timeout would cost ~80s on a hung vLLM before the fallback."""
    monkeypatch.setattr("rag.agent.settings.llm_backend", "openai-compatible")
    assert get_llm(timeout=20).max_retries == 0


def test_an_answer_call_keeps_the_openai_client_retries(monkeypatch):
    monkeypatch.setattr("rag.agent.settings.llm_backend", "openai-compatible")
    assert get_llm().max_retries > 0
