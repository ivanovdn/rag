import pytest

from config import settings
from rag.agent import get_llm


def test_ollama_llm_gets_keep_alive_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    monkeypatch.setattr(settings, "ollama_keep_alive", "30m")
    llm = get_llm()
    assert llm.keep_alive == "30m"


def test_ollama_keep_alive_default_is_longer_than_ollama_default():
    # Ollama's own default is '5m'; ours must outlast a working-day gap.
    assert settings.ollama_keep_alive != "5m"


def test_openai_like_disables_thinking(monkeypatch):
    # Qwen3 models think by default on vLLM. Measured cost: 294 reasoning
    # tokens and 34.5s for one 16-token router classification.
    monkeypatch.setattr(settings, "llm_backend", "openai-compatible")
    llm = get_llm()
    body = llm.additional_kwargs["extra_body"]
    assert body["chat_template_kwargs"]["enable_thinking"] is False


def test_get_llm_builds_a_fresh_client_per_call(monkeypatch):
    # _run_rag runs each request in a new asyncio.run() loop. llama-index's Ollama
    # caches its httpx.AsyncClient on first use, and a pooled connection from a
    # closed loop raises "RuntimeError: Event loop is closed" on the next one
    # (reproduced 2026-09-16). A fresh client per call is what keeps that safe.
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    assert get_llm() is not get_llm()
