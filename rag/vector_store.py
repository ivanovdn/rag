import json
from typing import TYPE_CHECKING

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
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
from rag.observability import get_tracer

if TYPE_CHECKING:
    from ingest.chunk_models import PolicyChunk

_client: QdrantClient | None = None

# Up to settings.reranker_candidates (20) chunks land on the search_vectors span,
# each posted individually over HTTP by a SimpleSpanProcessor — truncate document
# content so a debugging aid doesn't become real payload weight per request.
_DOCUMENT_CONTENT_MAX_CHARS = 1000


def _document_span_attributes(points: list) -> dict:
    """Flatten Qdrant ScoredPoints into OpenInference retrieval.documents.{i}.* attrs.

    Must never raise: this is purely a tracing side-channel, so every payload
    field is read with `.get()` — a missing/partial payload key must not break
    a search. Identifying fields go into DOCUMENT_METADATA as a JSON string,
    since those (doc_title/section/clause_number) are what a person scanning a
    trace actually reads.
    """
    attributes: dict = {}
    for i, point in enumerate(points):
        payload = getattr(point, "payload", None) or {}
        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
        attributes[prefix + DocumentAttributes.DOCUMENT_ID] = str(getattr(point, "id", ""))
        attributes[prefix + DocumentAttributes.DOCUMENT_CONTENT] = str(
            payload.get("text", "")
        )[:_DOCUMENT_CONTENT_MAX_CHARS]
        attributes[prefix + DocumentAttributes.DOCUMENT_SCORE] = float(
            getattr(point, "score", 0.0) or 0.0
        )
        attributes[prefix + DocumentAttributes.DOCUMENT_METADATA] = json.dumps(
            {
                "doc_title": payload.get("doc_title", ""),
                "section": payload.get("section", ""),
                "clause_number": payload.get("clause_number", ""),
            }
        )
    return attributes


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.active_qdrant_url, timeout=10)
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


def upsert_chunks(chunks: "list[PolicyChunk]", embeddings: list[list[float]]) -> None:
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
    tracer = get_tracer()
    limit = top_k or settings.retrieval_top_k
    with tracer.start_as_current_span(
        "search_vectors",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
            "qdrant.collection": settings.qdrant_collection,
            "qdrant.limit": limit,
        },
    ) as span:
        client = get_qdrant_client()
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        points = response.points
        span.set_attribute("qdrant.returned_count", len(points))
        if points:
            span.set_attribute("qdrant.top_score", points[0].score)
        span.set_attributes(_document_span_attributes(points))
        return points


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
