from llama_index.core.tools import FunctionTool
from qdrant_client.models import FieldCondition, Filter, MatchValue, MatchText

from rag.vector_store import scroll_by_filter


def get_section(doc_id: str, section_name: str) -> str:
    """
    Retrieve the FULL text of a specific policy section by document and section name.
    Use this tool when search_policies returned a partial chunk and you need
    the complete section text for precise citation.
    Input: doc_id (document slug, e.g. "acceptable-use-policy-internal")
           and section_name (section heading, e.g. "Corporate Workstation and Software Use").
    Output: all chunks belonging to that section with full text.
    """
    # Try matching section_display containing the section_name
    filter_conditions = Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="section_display", match=MatchText(text=section_name)),
        ]
    )

    results = scroll_by_filter(filter_conditions, limit=20)

    if not results:
        # Fallback: try matching as clause_number for numbered documents
        filter_conditions = Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="clause_number", match=MatchValue(value=section_name)),
            ]
        )
        results = scroll_by_filter(filter_conditions, limit=10)

    if not results:
        return f"No section found for doc_id='{doc_id}', section='{section_name}'"

    parts = []
    for point in sorted(results, key=lambda p: p.payload.get("chunk_index", 0)):
        p = point.payload
        clause = p.get("clause_number", "")
        header = f"Document: {p['doc_title']}\nSection: {p['section_display']}"
        if clause:
            header += f"\nClause: {clause}"
        header += f"\nLink: {p['doc_link']}"
        parts.append(f"{header}\nText: {p['text']}")

    return "\n\n---\n\n".join(parts)


get_section_tool = FunctionTool.from_defaults(fn=get_section)
