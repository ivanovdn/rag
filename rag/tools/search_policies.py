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
        Formatted policy excerpts with document name, section, clause number,
        and link. Returns "NO_RELEVANT_POLICY_FOUND" if no policies match.
    """
    if settings.bm25_enabled:
        from rag.hybrid_search import hybrid_search_formatted

        return hybrid_search_formatted(
            query=query,
            top_k=top_k,
            min_confidence=settings.min_confidence_score,
        )

    # Fallback: vector-only search
    from rag.embeddings import embed_query
    from rag.vector_store import search_vectors

    query_vector = embed_query(query)
    results = search_vectors(query_vector, top_k=top_k)

    if not results or results[0].score < settings.min_confidence_score:
        return "NO_RELEVANT_POLICY_FOUND"

    formatted = []
    for r in results:
        p = r.payload
        section_str = f"{p.get('section_number', '')}. {p.get('section', '')}" if p.get("section_number") else p.get("section", "")
        clause_str = f"{p.get('clause_number', '')}. {p.get('clause', '')}" if p.get("clause_number") and p.get("clause") else p.get("clause_number", "") or p.get("clause", "")

        line = f"[SCORE: {r.score:.2f}] Document: {p['doc_title']}"
        if section_str:
            line += f" | Section: {section_str}"
        if clause_str:
            line += f" | Clause: {clause_str}"
        line += f" | Doc ID: {p['doc_id']} | Link: {p['doc_link']}"
        line += f"\nText: {p['text']}\n"
        formatted.append(line)

    return "\n---\n".join(formatted)


search_policies_tool = FunctionTool.from_defaults(fn=search_policies)
