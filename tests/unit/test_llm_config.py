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
