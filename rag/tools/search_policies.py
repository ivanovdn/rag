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
    if settings.bm25_enabled:
        from rag.hybrid_search import hybrid_search

        results = hybrid_search(query=query, top_k=top_k)

        if not results or results[0].get("rrf_score", 0) < settings.min_confidence_score:
            return "NO_RELEVANT_POLICY_FOUND"

        formatted = []
        for i, r in enumerate(results, 1):
            lines = [
                f"--- Result {i} [RRF Score: {r['rrf_score']:.4f}] ---",
                f"Document: {r['doc_title']}",
                f"Section: {r.get('section', '')}",
                f"Clause: {r.get('clause', '')}",
                f"Clause Number: {r.get('clause_number', '')}",
                f"Doc ID: {r['doc_id']}",
                f"Text: {r['text']}",
            ]
            formatted.append("\n".join(lines))

        return "\n\n".join(formatted)

    # Fallback: vector-only search
    from rag.embeddings import embed_query
    from rag.vector_store import search_vectors

    query_vector = embed_query(query)
    results = search_vectors(query_vector, top_k=top_k)

    if not results or results[0].score < settings.min_confidence_score:
        return "NO_RELEVANT_POLICY_FOUND"

    formatted = []
    for i, r in enumerate(results, 1):
        p = r.payload
        lines = [
            f"--- Result {i} [Score: {r.score:.4f}] ---",
            f"Document: {p['doc_title']}",
            f"Section: {p.get('section', '')}",
            f"Clause: {p.get('clause', '')}",
            f"Clause Number: {p.get('clause_number', '')}",
            f"Doc ID: {p['doc_id']}",
            f"Text: {p['text']}",
        ]
        formatted.append("\n".join(lines))

    return "\n\n".join(formatted)


search_policies_tool = FunctionTool.from_defaults(fn=search_policies)
