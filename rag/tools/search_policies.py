from llama_index.core.tools import FunctionTool

from config import settings

_last_search_results: list[dict] = []


def search_policies(query: str, top_k: int = 6) -> str:
    """
    Search approved compliance policy documents for information relevant to the query.
    Uses hybrid search (semantic + keyword matching) for best accuracy.

    Args:
        query: Natural language search query describing what policy info you need.
               Be specific — include relevant terms, clause numbers, or policy names.
        top_k: Number of most relevant policy sections to return (default 6).

    Returns:
        Formatted policy excerpts with document name, section, clause, clause number,
        and full text. Returns "NO_RELEVANT_POLICY_FOUND" if no policies match.
    """
    global _last_search_results

    # How many candidates to retrieve (more when reranker will rescore)
    retrieve_k = settings.reranker_candidates if settings.reranker_enabled else top_k

    # Step 1: Retrieve candidates
    if settings.bm25_enabled:
        from rag.hybrid_search import hybrid_search

        raw = hybrid_search(query=query, top_k=retrieve_k)
        if not raw:
            _last_search_results = []
            return "NO_RELEVANT_POLICY_FOUND"

        results = [
            {
                "doc_title": r["doc_title"],
                "doc_id": r["doc_id"],
                "section": r.get("section", ""),
                "clause": r.get("clause", ""),
                "clause_number": r.get("clause_number", ""),
                "text": r["text"],
                "retrieval_score": r["rrf_score"],
                "score_type": "rrf",
            }
            for r in raw
        ]
    else:
        from rag.embeddings import embed_query
        from rag.vector_store import search_vectors

        query_vector = embed_query(query)
        raw = search_vectors(query_vector, top_k=retrieve_k)

        if not raw:
            _last_search_results = []
            return "NO_RELEVANT_POLICY_FOUND"

        # Apply confidence threshold only when reranker is OFF
        if not settings.reranker_enabled and raw[0].score < settings.min_confidence_score:
            _last_search_results = []
            return "NO_RELEVANT_POLICY_FOUND"

        results = [
            {
                "doc_title": r.payload["doc_title"],
                "doc_id": r.payload["doc_id"],
                "section": r.payload.get("section", ""),
                "clause": r.payload.get("clause", ""),
                "clause_number": r.payload.get("clause_number", ""),
                "text": r.payload["text"],
                "retrieval_score": r.score,
                "score_type": "cosine",
            }
            for r in raw
        ]

    # Step 2: Rerank (if enabled)
    if settings.reranker_enabled and results:
        from rag.reranker import rerank

        results = rerank(query, results, top_n=settings.reranker_top_n)

    # Step 3: Capture structured results for eval logging
    _last_search_results = [
        {
            "doc_title": r["doc_title"],
            "section": r.get("section", ""),
            "clause": r.get("clause", ""),
            "clause_number": r.get("clause_number", ""),
            "rerank_score": round(r.get("rerank_score", 0), 4),
            "retrieval_score": round(r.get("retrieval_score", 0), 4),
        }
        for r in results
    ]

    # Step 4: Format for the agent
    return format_sources(results)


def format_sources(search_results: list[dict]) -> str:
    """Format search results for the agent. No scores, no doc_id — just policy content."""
    if not search_results:
        return "=== RETRIEVED POLICY SOURCES ===\n\nNO_RELEVANT_POLICY_FOUND"

    lines = ["=== RETRIEVED POLICY SOURCES ==="]

    for i, r in enumerate(search_results):
        lines.append("")  # blank line between sources
        lines.append(f"[Source {i + 1}] {r['doc_title']}")
        lines.append(f"Section: {r['section']}")
        if r.get("clause_number"):
            lines.append(f"Clause Number: {r['clause_number']}")
        if r.get("clause"):
            lines.append(f"Clause Name: {r['clause']}")
        lines.append("---")
        lines.append(r["text"])

    return "\n".join(lines)


search_policies_tool = FunctionTool.from_defaults(fn=search_policies)
