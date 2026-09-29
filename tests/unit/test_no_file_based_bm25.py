"""The file-based BM25 index is gone, not merely unused.

It was a gitignored JSON file, built only when BM25 was already on, absent from
the Docker image, and desynchronising invisibly because chunk_id is a fresh
uuid4 per ingest -- an index from an earlier ingestion had the right chunk COUNT
and entirely wrong ids (measured: 1602 vs 1602, 0 of 25 ids present). Leaving the
modules importable is how that comes back.
"""

from pathlib import Path


def test_the_file_based_modules_are_deleted():
    for path in (
        "rag/bm25_index.py",
        "rag/hybrid_search.py",
        "scripts/build_bm25_from_qdrant.py",
    ):
        assert not Path(path).exists(), f"{path} still exists"


def test_nothing_imports_them():
    """Source files only. A TEST importing a deleted module fails at collection,
    which is a louder signal than this scan — and tests legitimately mention the
    names in assertions and docstrings."""
    offenders = []
    for path in Path(".").rglob("*.py"):
        if any(part.startswith(".") or part in {"__pycache__", "tests"} for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        if "hybrid_search" in source or "bm25_index" in source:
            offenders.append(str(path))
    assert offenders == [], f"still referencing deleted modules: {offenders}"


def test_the_stale_gitignore_entry_is_gone():
    assert ".bm25_index.json" not in Path(".gitignore").read_text(encoding="utf-8")
