"""The `Document | Section | Clause` header every chunk is embedded with.

Shared by ingest (the dense embedding text) and the reranker (RERANKER_DOC_HEADER),
so the two can never build it differently.
"""


def with_header(text: str, doc_title: str = "", section: str = "", clause: str = "") -> str:
    parts = []
    if doc_title:
        parts.append(f"Document: {doc_title}")
    if section:
        parts.append(f"Section: {section}")
    if clause:
        parts.append(f"Clause: {clause}")
    prefix = " | ".join(parts)
    return f"{prefix}\n{text}" if prefix else text
