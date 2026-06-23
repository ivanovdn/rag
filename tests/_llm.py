"""Probe whether a classification LLM is reachable, so live tests can auto-skip
(like the docx-corpus tests) on machines/CI without the local model stack."""
import httpx

from config import settings


def llm_reachable() -> bool:
    url = settings.active_ollama_url if settings.llm_backend == "ollama" else settings.openai_api_base
    try:
        httpx.get(url, timeout=2.0)  # any HTTP response (even 404) means it's up
        return True
    except Exception:
        return False
