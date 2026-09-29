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
    Document,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVectorParams,
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

# The sparse vector's name inside the collection, and Qdrant's built-in BM25
# model. Only built-in models resolve on a self-hosted server; any other name
# fails with "InferenceService URL not configured", which would require
# server-side config on a shared host we do not own.
SPARSE_VECTOR_NAME = "bm25"
BM25_MODEL = "qdrant/bm25"


def bm25_document(text: str) -> Document:
    """Text handed to Qdrant for server-side BM25 encoding.

    avg_len goes in per-document `options` on EVERY call, at ingest and at query
    time alike: Qdrant 1.17.1 accepts a collection-level Bm25Config and silently
    discards it (verified 2026-09-29 — the collection reads back carrying only
    {"modifier": "idf"}), so this is the only place k/b/avg_len take effect.
    """
    return Document(
        text=text,
        model=BM25_MODEL,
        options={"avg_len": settings.bm25_avg_len},
    )


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
        # cloud_inference=True means "do not embed models.Document locally, send
        # the text to the server". Despite the name it is correct for a
        # self-hosted server: Qdrant 1.17.1 resolves the built-in qdrant/bm25
        # model itself, with no InferenceService configured (verified
        # 2026-09-29). The client default is False, i.e. encode locally via
        # fastembed — which works today only because fastembed is not installed.
        _client = QdrantClient(
            url=settings.active_qdrant_url, timeout=10, cloud_inference=True
        )
    return _client


def init_collection(collection_name: str | None = None) -> None:
    """Create the collection with its payload indexes if it does not exist.

    Takes an explicit name so scripts/migrate_collection.py can build a target
    other than the configured one from this single schema definition, rather
    than duplicating it and letting the two drift.
    """
    name = collection_name or settings.qdrant_collection
    client = get_qdrant_client()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            # Bare VectorParams, not a dict: that is what keeps the dense vector
            # unnamed, so no search call needs `using=`. Verified that an unnamed
            # dense vector coexists with a named sparse one.
            vectors_config=VectorParams(
                size=settings.qdrant_vector_dim,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        for field, schema in (
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("section", PayloadSchemaType.KEYWORD),
            ("section_number", PayloadSchemaType.KEYWORD),
            ("clause", PayloadSchemaType.KEYWORD),
            ("clause_number", PayloadSchemaType.KEYWORD),
            ("section_display", PayloadSchemaType.TEXT),
        ):
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema
            )


def assert_sparse_vector(collection_name: str, reason: str) -> None:
    """Refuse to proceed unless `collection_name` carries the sparse vector.

    The single definition of that check, shared by every caller that needs it —
    the query-side preflight below and the write-side guard in
    ingest/pipeline.py. `reason` is the caller's framing; the part an operator
    has to act on (which collection, which vector is missing, and the command
    that builds one) is appended here, once, so the two cannot drift.

    The collection name is explicit rather than defaulted, because pointing a
    write at the wrong collection is the mistake this whole family of guards
    exists to catch.
    """
    client = get_qdrant_client()
    info = client.get_collection(collection_name)
    sparse = info.config.params.sparse_vectors or {}
    if SPARSE_VECTOR_NAME in sparse:
        return
    raise RuntimeError(
        f"{reason} Collection '{collection_name}' has no '{SPARSE_VECTOR_NAME}' "
        f"sparse vector (found: {sorted(sparse) or 'none'}). Build a sparse-enabled "
        "collection with "
        "`PYTHONPATH=. python scripts/migrate_collection.py --target <name>`."
    )


def preflight_sparse_config() -> None:
    """Refuse to start if BM25 is on but the collection has no sparse vector.

    Qdrant answers a sparse query against a collection lacking that vector with
    "Not existing vector name error". That is NOT transient, so without this
    check it would surface as a content escalation on every single question --
    the bot would appear healthy while finding nothing.

    Deliberately raises instead of auto-disabling BM25. A bot that quietly drops
    to dense-only looks fine and answers worse, which is exactly the failure mode
    this migration exists to remove.

    Query-side only, hence the bm25_enabled gate. The WRITE side is ungated and
    calls assert_sparse_vector directly: upsert_chunks always writes a sparse
    vector, so a collection without one breaks ingest whatever this flag says.
    """
    if not settings.bm25_enabled:
        return
    assert_sparse_vector(
        settings.qdrant_collection,
        "BM25_ENABLED=true, but this collection cannot answer a sparse query "
        "(set BM25_ENABLED=false to run dense-only).",
    )


def upsert_chunks(chunks: "list[PolicyChunk]", embeddings: list[list[float]]) -> None:
    """Upsert chunks with their dense and sparse vectors into Qdrant.

    The sparse vector is written unconditionally, regardless of BM25_ENABLED:
    that flag is query-side only now. Gating the write on it is what let an
    index and a collection drift apart silently.
    """
    client = get_qdrant_client()
    points = [
        PointStruct(
            id=chunk.chunk_id,
            # "" is the unnamed dense vector; "bm25" is the named sparse one.
            vector={"": embedding, SPARSE_VECTOR_NAME: bm25_document(chunk.text)},
            payload=chunk.model_dump(),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
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


def search_chunks(
    query_text: str, query_vector: list[float], top_k: int | None = None
) -> list:
    """Retrieve candidate chunks: dense only, or dense + sparse fused by Qdrant.

    Returns list[ScoredPoint] in BOTH modes, because fusion happens server-side —
    so no caller needs a branch. The score means different things (an RRF score
    when bm25_enabled, a cosine similarity otherwise); callers record which in
    `score_type`, and anything comparing a score against a threshold must check
    which scale it is on. See search_policies' min_confidence_score guard.
    """
    tracer = get_tracer()
    limit = top_k or settings.retrieval_top_k
    client = get_qdrant_client()
    # The span name stays "search_vectors" although the function was renamed, so
    # Phoenix comparisons against runs recorded before this migration stay valid.
    with tracer.start_as_current_span(
        "search_vectors",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
            "qdrant.collection": settings.qdrant_collection,
            "qdrant.limit": limit,
            "qdrant.bm25_enabled": settings.bm25_enabled,
        },
    ) as span:
        if settings.bm25_enabled:
            span.set_attribute("qdrant.fusion", "rrf")
            span.set_attribute("qdrant.bm25_avg_len", settings.bm25_avg_len)
            response = client.query_points(
                collection_name=settings.qdrant_collection,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        limit=settings.hybrid_vector_candidates,
                    ),
                    Prefetch(
                        query=bm25_document(query_text),
                        using=SPARSE_VECTOR_NAME,
                        limit=settings.hybrid_bm25_candidates,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
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
