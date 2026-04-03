import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import settings

_embedding_model: HuggingFaceEmbedding | None = None


def get_embedding_model() -> HuggingFaceEmbedding:
    global _embedding_model
    if _embedding_model is None:
        if settings.hf_token:
            os.environ["HF_TOKEN"] = settings.hf_token
        _embedding_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            trust_remote_code=True,
            query_instruction=settings.embedding_query_prefix,
            text_instruction=settings.embedding_passage_prefix,
        )
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (passage prefix applied automatically)."""
    model = get_embedding_model()
    return [model.get_text_embedding(t) for t in texts]


def embed_query(query: str) -> list[float]:
    """Embed a single query (query prefix applied automatically)."""
    model = get_embedding_model()
    return model.get_query_embedding(query)
