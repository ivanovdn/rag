from llama_index.core.tools import FunctionTool

from config import settings
from rag.embeddings import embed_query
from rag.vector_store import search_vectors


def search_policies(query: str, top_k: int = 6) -> str:
    """
    Search the approved internal policy and procedure documents
    using semantic similarity. Use this tool FIRST for any compliance question.
    Returns relevant policy chunks with their exact section references.
    Input: a natural language query describing what policy information is needed.
    Output: list of matching policy chunks with doc title, section path, clause number, and text.
    Always call this tool before attempting to answer any compliance question.
    """
    query_vector = embed_query(query)
    results = search_vectors(query_vector, top_k=top_k)

    if not results or results[0].score < settings.min_confidence_score:
        return "NO_RELEVANT_POLICY_FOUND"

    formatted = []
    for r in results:
        p = r.payload
        clause = p.get("clause_number", "")
        section_ref = f"Section: {p['section_display']}"
        if clause:
            section_ref += f" | Clause: {clause}"
        formatted.append(
            f"[SCORE: {r.score:.2f}] "
            f"Document: {p['doc_title']} | "
            f"{section_ref} | "
            f"Doc ID: {p['doc_id']} | "
            f"Link: {p['doc_link']}\n"
            f"Text: {p['text']}\n"
        )

    return "\n---\n".join(formatted)


search_policies_tool = FunctionTool.from_defaults(fn=search_policies)
