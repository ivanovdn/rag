import os

import httpx

from config import settings

_embedding_model = None


def _get_huggingface_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
    return HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        trust_remote_code=True,
        query_instruction=settings.embedding_query_prefix,
        text_instruction=settings.embedding_passage_prefix,
    )


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if settings.embedding_source == "ollama":
            _embedding_model = "ollama"
        else:
            _embedding_model = _get_huggingface_model()
    return _embedding_model


def _ollama_embed(texts: list[str], prefix: str = "") -> list[list[float]]:
    """Call Ollama /api/embed endpoint."""
    url = f"{settings.ollama_embedding_url}/api/embed"
    prefixed = [f"{prefix}{t}" if prefix else t for t in texts]
    resp = httpx.post(
        url,
        json={"model": settings.embedding_model, "input": prefixed},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (passage prefix applied automatically)."""
    model = get_embedding_model()
    if model == "ollama":
        return _ollama_embed(texts, prefix=settings.embedding_passage_prefix)
    return [model.get_text_embedding(t) for t in texts]


def embed_query(query: str) -> list[float]:
    """Embed a single query (query prefix applied automatically)."""
    model = get_embedding_model()
    if model == "ollama":
        return _ollama_embed([query], prefix=settings.embedding_query_prefix)[0]
    return model.get_query_embedding(query)
