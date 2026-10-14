"""The header the embedding was built from, in one place (design 2026-10-09).

Ingest and the reranker must build it identically: if they drift, the reranker
judges a different text than the one retrieval matched, and nothing errors.
"""

from pathlib import Path

import ingest.pipeline as pl
from rag.chunk_header import with_header


def test_ingest_embeds_exactly_what_it_did_before(monkeypatch):
    """The 1602 stored vectors were built from this text; any drift silently
    desynchronises the next ingest from them. Expected strings are the old
    loop's output, written out by hand."""

    class C:
        def __init__(self, section, clause):
            self.doc_id, self.doc_title, self.text = "d", "Doc", "Body."
            self.section, self.clause = section, clause

    seen = {}
    monkeypatch.setattr(pl, "init_collection", lambda: None)
    monkeypatch.setattr(pl, "assert_sparse_vector", lambda *a, **kw: None)
    monkeypatch.setattr(pl, "parse_docx", lambda fp, link: [C("Sec", "Cl"), C("", "Cl")])
    monkeypatch.setattr(pl, "delete_document", lambda doc_id: None)
    monkeypatch.setattr(pl, "embed_texts", lambda xs: seen.setdefault("x", xs) and [[0.0]] * len(xs))
    monkeypatch.setattr(pl, "upsert_chunks", lambda chunks, vecs: None)

    pl.ingest_document(Path("x.docx"), "")

    assert seen["x"] == [
        "Document: Doc | Section: Sec | Clause: Cl\nBody.",
        "Document: Doc | Clause: Cl\nBody.",
    ]


def test_full_header():
    assert with_header("Body.", "Backup Policy [Internal]", "Backups", "Frequency") == (
        "Document: Backup Policy [Internal] | Section: Backups | Clause: Frequency\nBody."
    )


def test_empty_parts_are_omitted_not_left_blank():
    assert with_header("Body.", "Doc", "", "Cl") == "Document: Doc | Clause: Cl\nBody."
    assert with_header("Body.", "Doc") == "Document: Doc\nBody."


def test_no_parts_is_the_bare_text():
    assert with_header("Body.") == "Body."
