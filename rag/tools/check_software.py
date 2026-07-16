"""check_software tool: hybrid allowed/forbidden lookup.

Fuzzy name match first (in-memory index built by scrolling the software_registry
collection), then a score-gated semantic fallback for category questions
("what VPN can I use?"). Mirrors search_policies' resilience pattern: transient
backend failures set a module flag and return a sentinel instead of raising.
"""

from llama_index.core.tools import FunctionTool

from config import settings
from rag.embeddings import embed_query
from rag.observability import record_infra_unavailable
from rag.resilience import RETRY_BACKOFFS, is_transient, retry_transient
from rag.software_registry import (
    SoftwareRow,
    build_name_index,
    lookup_name,
)
from rag.vector_store import scroll_all, search_vectors

_name_index: dict[str, SoftwareRow] | None = None

# Read by channels/teams/bot.py after the agent run:
_software_unavailable: bool = False
_software_not_found: bool = False
_software_suggestion: dict | None = None
_software_query: str = ""

UNAVAILABLE = "SOFTWARE_LOOKUP_UNAVAILABLE"
NOT_LISTED = "SOFTWARE_NOT_LISTED"


def _get_name_index() -> dict[str, SoftwareRow]:
    global _name_index
    if _name_index is None:
        rows = [SoftwareRow(**p) for p in scroll_all(settings.software_collection)]
        _name_index = build_name_index(rows)
    return _name_index


def format_software_results(rows: list[SoftwareRow]) -> str:
    lines = ["=== SOFTWARE REGISTRY RESULTS ==="]
    for i, r in enumerate(rows):
        lines.append("")
        lines.append(f"[Software {i + 1}] {r.name}")
        lines.append(f"Status: {r.status}")
        lines.append(f"List: {r.source_list}")
        if r.category:
            lines.append(f"Category: {r.category}")
        if r.note:
            lines.append(f"Note: {r.note}")
        if r.alternative:
            lines.append(f"Alternative: {r.alternative}")
    return "\n".join(lines)


def check_software(name: str) -> str:
    """
    Look up whether a specific software/tool is ALLOWED or FORBIDDEN for company use,
    or find the approved alternative for a category of tool.

    Args:
        name: The specific software name (e.g. "Docker", "TeamViewer") OR a short
              category phrase (e.g. "personal VPN", "IDE"). Pass the name/category
              only — NOT the whole question.

    Returns:
        Formatted registry result(s) with Status (allowed/forbidden), source List,
        Category, Note, and Alternative. Returns "SOFTWARE_NOT_LISTED" (optionally with
        a "DID_YOU_MEAN" hint) if not found, or "SOFTWARE_LOOKUP_UNAVAILABLE" on a
        transient backend failure.
    """
    global _software_unavailable, _software_not_found, _software_suggestion, _software_query
    _software_unavailable = False
    _software_not_found = False
    _software_suggestion = None
    _software_query = name

    # Step 1: fuzzy name lookup (build the index from Qdrant on first use)
    try:
        index = retry_transient(_get_name_index)
    except Exception as exc:
        if is_transient(exc):
            _software_unavailable = True
            record_infra_unavailable("software_qdrant", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    result = lookup_name(name, index, settings.software_fuzzy_threshold)
    if result.row is not None:
        return format_software_results([result.row])

    # Step 2: score-gated semantic fallback (category questions)
    try:
        query_vector = retry_transient(lambda: embed_query(name))
    except Exception as exc:
        if is_transient(exc):
            _software_unavailable = True
            record_infra_unavailable("software_embeddings", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    try:
        raw = retry_transient(
            lambda: search_vectors(query_vector, top_k=5, collection_name=settings.software_collection)
        )
    except Exception as exc:
        if is_transient(exc):
            _software_unavailable = True
            record_infra_unavailable("software_qdrant", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    hits = [h for h in raw if h.score >= settings.software_min_semantic_score]
    if hits:
        return format_software_results([SoftwareRow(**h.payload) for h in hits])

    # Step 3: not found — offer the closest fuzzy candidate as a hint
    _software_not_found = True
    if result.suggestion is not None:
        _software_suggestion = {"name": result.suggestion.name, "status": result.suggestion.status}
        return f"{NOT_LISTED}\nDID_YOU_MEAN: {result.suggestion.name} ({result.suggestion.status})"
    return NOT_LISTED


check_software_tool = FunctionTool.from_defaults(fn=check_software)
