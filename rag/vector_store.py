from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config import settings
from ingest.chunk_models import PolicyChunk

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.active_qdrant_url)
    return _client


def init_collection() -> None:
    """Create collection with payload indexes if it doesn't exist."""
    client = get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_dim,
                distance=Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="section",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="section_number",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="clause",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="clause_number",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="section_display",
            field_schema=PayloadSchemaType.TEXT,
        )


def upsert_chunks(chunks: list[PolicyChunk], embeddings: list[list[float]]) -> None:
    """Upsert chunks with their embeddings into Qdrant."""
    client = get_qdrant_client()
    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=embedding,
            payload=chunk.model_dump(),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + batch_size],
        )


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to a document."""
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )


def search_vectors(
    query_vector: list[float], top_k: int | None = None
) -> list:
    """Search for similar vectors, returns list of ScoredPoint."""
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k or settings.retrieval_top_k,
        with_payload=True,
    )
    return response.points


def scroll_by_filter(filter_conditions: Filter, limit: int = 10) -> list:
    """Scroll through points matching a filter."""
    client = get_qdrant_client()
    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=filter_conditions,
        limit=limit,
        with_payload=True,
    )
    return results
