"""Parsing-coverage report over the local policy corpus.

Provides compute_parse_stats() (shared with the corpus test) and a CLI that
prints per-doc parsing metrics. The corpus is gitignored, so this only does
anything useful on a machine that has the .docx files locally.

Usage:
    PYTHONPATH=. python scripts/parse_coverage.py
"""
from pathlib import Path

from config import settings
from ingest.chunk_models import PolicyChunk
from ingest.docx_parser import parse_docx, _estimate_tokens


def find_policy_docx(folder=None) -> list[Path]:
    folder = Path(folder or settings.policy_docs_folder)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.docx"))


def compute_parse_stats(chunks: list[PolicyChunk]) -> dict:
    """Parsing-completeness metrics for one or more docs' chunks."""
    total = len(chunks)
    with_clause_number = sum(1 for c in chunks if c.clause_number)
    with_section_or_clause = sum(1 for c in chunks if c.section or c.clause)
    empty_text = sum(1 for c in chunks if not c.text.strip())
    oversized = sum(1 for c in chunks if _estimate_tokens(c.text) > settings.chunk_max_tokens)
    return {
        "chunk_count": total,
        "with_clause_number": with_clause_number,
        "with_section_or_clause": with_section_or_clause,
        "empty_text": empty_text,
        "oversized": oversized,
        "pct_clause_number": round(with_clause_number / total, 3) if total else 0.0,
        "pct_section_or_clause": round(with_section_or_clause / total, 3) if total else 0.0,
    }


def main() -> None:
    paths = find_policy_docx()
    if not paths:
        print(f"No .docx found in {settings.policy_docs_folder} (corpus is local-only).")
        return

    print(f"{'doc':48} {'chunks':>6} {'clause%':>8} {'sec/cl%':>8} {'empty':>6} {'oversz':>7}")
    print("-" * 86)
    all_chunks: list[PolicyChunk] = []
    for path in paths:
        chunks = parse_docx(path, doc_link="")
        all_chunks.extend(chunks)
        s = compute_parse_stats(chunks)
        print(f"{path.stem[:48]:48} {s['chunk_count']:>6} {s['pct_clause_number']:>8} "
              f"{s['pct_section_or_clause']:>8} {s['empty_text']:>6} {s['oversized']:>7}")

    g = compute_parse_stats(all_chunks)
    print("-" * 86)
    print(f"TOTAL: {len(paths)} docs, {g['chunk_count']} chunks | "
          f"clause#={g['pct_clause_number']} sec/clause={g['pct_section_or_clause']} "
          f"empty={g['empty_text']} oversized={g['oversized']}")


if __name__ == "__main__":
    main()
