"""
Hybrid search: combines Qdrant vector search with BM25 keyword search
using Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
where k = 60 (standard constant), rank_i is the rank in each result list.

This gives a balanced merge — a document ranked high in both lists
gets the best combined score, but being #1 in either list alone is
still valuable.
"""

import logging

from config import settings
from rag.bm25_index import search_bm25
from rag.embeddings import embed_query
from rag.vector_store import search_vectors

logger = logging.getLogger(__name__)

# RRF constant — standard value from the original paper
_RRF_K = 60


def hybrid_search(
    query: str,
    top_k: int = 6,
    vector_candidates: int | None = None,
    bm25_candidates: int | None = None,
) -> list[dict]:
    """
    Run hybrid search combining vector + BM25, fused with RRF.

    Returns list of dicts, each with:
      - chunk_id, doc_id, doc_title, section_display, clause_number,
        doc_link, text
      - rrf_score (combined), vector_score, bm25_score
      - vector_rank, bm25_rank (None if not present in that list)
    """
    v_candidates = vector_candidates or settings.hybrid_vector_candidates
    b_candidates = bm25_candidates or settings.hybrid_bm25_candidates

    # 1. Vector search
    query_vector = embed_query(query)
    vector_results = search_vectors(query_vector, top_k=v_candidates)
    logger.info(
        f"Vector search: {len(vector_results)} results"
        + (f", top score: {vector_results[0].score:.3f}" if vector_results else "")
    )

    # 2. BM25 search
    bm25_results = search_bm25(query, top_k=b_candidates)
    logger.info(
        f"BM25 search: {len(bm25_results)} results"
        + (f", top score: {bm25_results[0][1]:.2f}" if bm25_results else "")
    )

    # 3. Build per-chunk data from vector results
    chunks: dict[str, dict] = {}  # chunk_id -> merged data

    for rank, point in enumerate(vector_results, start=1):
        cid = point.id
        p = point.payload
        chunks[cid] = {
            "chunk_id": cid,
            "doc_id": p.get("doc_id", ""),
            "doc_title": p.get("doc_title", ""),
            "section_display": p.get("section_display", ""),
            "clause_number": p.get("clause_number", ""),
            "doc_link": p.get("doc_link", ""),
            "text": p.get("text", ""),
            "vector_score": point.score,
            "vector_rank": rank,
            "bm25_score": 0.0,
            "bm25_rank": None,
        }

    # 4. Merge BM25 results
    for rank, (cid, bm25_score, meta) in enumerate(bm25_results, start=1):
        if cid in chunks:
            chunks[cid]["bm25_score"] = bm25_score
            chunks[cid]["bm25_rank"] = rank
        else:
            chunks[cid] = {
                "chunk_id": cid,
                "doc_id": meta.get("doc_id", ""),
                "doc_title": meta.get("doc_title", ""),
                "section_display": meta.get("section_display", ""),
                "clause_number": meta.get("clause_number", ""),
                "doc_link": meta.get("doc_link", ""),
                "text": meta.get("text", ""),
                "vector_score": 0.0,
                "vector_rank": None,
                "bm25_score": bm25_score,
                "bm25_rank": rank,
            }

    # 5. Compute RRF scores
    for chunk in chunks.values():
        rrf = 0.0
        if chunk["vector_rank"] is not None:
            rrf += 1.0 / (_RRF_K + chunk["vector_rank"])
        if chunk["bm25_rank"] is not None:
            rrf += 1.0 / (_RRF_K + chunk["bm25_rank"])
        chunk["rrf_score"] = rrf

    # 6. Sort by RRF score and return top_k
    ranked = sorted(chunks.values(), key=lambda x: x["rrf_score"], reverse=True)
    top = ranked[:top_k]

    if top:
        best = top[0]
        logger.info(
            f"Top result after RRF: {best['doc_title']} | "
            f"{best['section_display']}"
            + (f" | {best['clause_number']}" if best["clause_number"] else "")
            + f" (vector_rank={best['vector_rank']}, bm25_rank={best['bm25_rank']})"
        )

    return top


def hybrid_search_formatted(
    query: str,
    top_k: int = 6,
    min_confidence: float | None = None,
) -> str:
    """
    Run hybrid search and return formatted string for the agent tool.
    Falls back to NO_RELEVANT_POLICY_FOUND if no results above threshold.
    """
    threshold = min_confidence if min_confidence is not None else settings.min_confidence_score
    results = hybrid_search(query, top_k=top_k)

    if not results:
        return "NO_RELEVANT_POLICY_FOUND"

    # Filter: at least one result must have a vector_score above threshold
    # (BM25-only results don't have a cosine score to compare against)
    has_confident = any(
        r["vector_score"] >= threshold for r in results
    )
    if not has_confident:
        return "NO_RELEVANT_POLICY_FOUND"

    formatted = []
    for r in results:
        section_ref = f"Section: {r['section_display']}"
        if r["clause_number"]:
            section_ref += f" | Clause: {r['clause_number']}"

        score_info = f"[RRF: {r['rrf_score']:.4f}"
        if r["vector_rank"] is not None:
            score_info += f" | vec={r['vector_score']:.2f} rank={r['vector_rank']}"
        if r["bm25_rank"] is not None:
            score_info += f" | bm25={r['bm25_score']:.1f} rank={r['bm25_rank']}"
        score_info += "]"

        formatted.append(
            f"{score_info} "
            f"Document: {r['doc_title']} | "
            f"{section_ref} | "
            f"Doc ID: {r['doc_id']} | "
            f"Link: {r['doc_link']}\n"
            f"Text: {r['text']}\n"
        )

    return "\n---\n".join(formatted)
