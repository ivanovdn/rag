from llama_index.embeddings.ollama import OllamaEmbedding

from config import settings

_embedding_model: OllamaEmbedding | None = None


def get_embedding_model() -> OllamaEmbedding:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.ollama_base_url,
            ollama_additional_kwargs={"mirostat": 0},
        )
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts."""
    model = get_embedding_model()
    return [model.get_text_embedding(t) for t in texts]


def embed_query(query: str) -> list[float]:
    """Embed a single query."""
    model = get_embedding_model()
    return model.get_query_embedding(query)
