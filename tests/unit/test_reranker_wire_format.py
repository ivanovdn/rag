"""The reranker's two backends speak different wire formats, and nothing pinned it.

Found by mutating `settings.reranker_uses_chat_template` to `reranker_backend ==
"vllm"` while extracting that predicate into config: `vllm-score` -- the backend
production runs -- silently fell through to the llama-server path, sending an
unwrapped query and unwrapped documents to a model that requires the Qwen3 chat
template, and the whole suite stayed green.

That failure does not raise. The server accepts the request and returns scores,
they are simply worse, and `rerank()` falls back to the original order on error
rather than on nonsense -- so the symptom is a quiet drop in retrieval quality
with no error anywhere. These tests make the backend family structural.
"""

import pytest

import rag.reranker as rr


@pytest.fixture
def backend(monkeypatch):
    def _set(name):
        monkeypatch.setattr(rr.settings, "reranker_backend", name)
    return _set


@pytest.mark.parametrize("name", ["vllm", "vllm-score"])
def test_a_vllm_query_carries_the_full_chat_template(backend, name):
    backend(name)

    query = rr._build_query("may I use a personal laptop")

    assert query.startswith("<|im_start|>system")
    assert "<Instruct>: " in query
    assert "<Query>: may I use a personal laptop" in query


@pytest.mark.parametrize("name", ["vllm", "vllm-score"])
def test_a_vllm_document_is_wrapped_and_closed(backend, name):
    backend(name)

    docs = rr._build_documents(["Team Members must use corporate-issued workstations."])

    assert docs[0].startswith("<Document>: ")
    # The assistant turn has to be opened and the empty think block closed, or the
    # model continues the user turn instead of scoring.
    assert docs[0].endswith(rr._VLLM_DOC_SUFFIX)


def test_the_two_vllm_backends_are_treated_identically(backend):
    """`vllm-score` differs from `vllm` only in which endpoint reads the result
    (see the /v1/score branch in rerank), never in what goes on the wire. A
    predicate that names one and not the other is the bug this file was born
    from, and it is invisible from the outside."""
    backend("vllm")
    vllm_query, vllm_docs = rr._build_query("q"), rr._build_documents(["d"])
    backend("vllm-score")

    assert rr._build_query("q") == vllm_query
    assert rr._build_documents(["d"]) == vllm_docs


def test_llama_server_takes_the_simple_template_and_no_wrapping(backend, monkeypatch):
    backend("llama-server")
    monkeypatch.setattr(rr.settings, "reranker_query_template", "<Instruct>: {instruction}\\n<Query>: {query}")

    query = rr._build_query("q")
    docs = rr._build_documents(["d"])

    assert "<|im_start|>" not in query
    # The literal backslash-n from .env is converted, not sent through.
    assert "\\n" not in query
    assert query.endswith("<Query>: q")
    assert docs == ["d"]
