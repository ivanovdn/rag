from llama_index.core.tools import FunctionTool

from config import settings


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
    # Step 1: Retrieve candidates
    if settings.bm25_enabled:
        from rag.hybrid_search import hybrid_search

        raw = hybrid_search(query=query, top_k=top_k)
        if not raw:
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
        raw = search_vectors(query_vector, top_k=top_k)

        if not raw or raw[0].score < settings.min_confidence_score:
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

        results = rerank(query, results, top_k=settings.reranker_top_k)

    # Step 3: Format for the agent
    formatted = []
    for i, r in enumerate(results, 1):
        score_label = f"Rerank: {r['rerank_score']:.3f}" if "rerank_score" in r else f"{r['score_type'].upper()}: {r['retrieval_score']:.4f}"
        lines = [
            f"--- Result {i} [{score_label}] ---",
            f"Document: {r['doc_title']}",
            f"Section: {r.get('section', '')}",
            f"Clause: {r.get('clause', '')}",
            f"Clause Number: {r.get('clause_number', '')}",
            f"Doc ID: {r['doc_id']}",
            f"Text: {r['text']}",
        ]
        formatted.append("\n".join(lines))

    return "\n\n".join(formatted)


search_policies_tool = FunctionTool.from_defaults(fn=search_policies)
