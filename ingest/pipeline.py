import logging
from pathlib import Path

from config import settings
from ingest.docx_parser import parse_docx
from rag.embeddings import embed_texts
from rag.vector_store import delete_document, init_collection, upsert_chunks

logger = logging.getLogger(__name__)


def ingest_document(filepath: Path, doc_link: str) -> int:
    """Parse a DOCX file, embed chunks, and upsert to Qdrant."""
    init_collection()
    chunks = parse_docx(filepath, doc_link)
    if not chunks:
        return 0

    doc_id = chunks[0].doc_id

    # Remove old version (both stores)
    if settings.bm25_enabled:
        from rag.bm25_index import remove_document_from_bm25

        removed = remove_document_from_bm25(doc_id)
        if removed:
            logger.info(f"Removed {removed} old BM25 chunks for {doc_id}")

    delete_document(doc_id)

    # Build embedding text: prepend metadata so doc/section/clause
    # participate in cosine similarity (not just the chunk body)
    embed_inputs = []
    for c in chunks:
        prefix_parts = []
        if c.doc_title:
            prefix_parts.append(f"Document: {c.doc_title}")
        if c.section:
            prefix_parts.append(f"Section: {c.section}")
        if c.clause:
            prefix_parts.append(f"Clause: {c.clause}")
        prefix = " | ".join(prefix_parts)
        embed_inputs.append(f"{prefix}\n{c.text}" if prefix else c.text)

    embeddings = embed_texts(embed_inputs)
    upsert_chunks(chunks, embeddings)

    # Sync BM25 index
    if settings.bm25_enabled:
        from rag.bm25_index import add_chunks_to_bm25

        add_chunks_to_bm25(chunks)
        logger.info(f"Added {len(chunks)} chunks to BM25 index")

    return len(chunks)


def resolve_docx_paths(raw_paths: list[str]) -> tuple[list[Path], list[str]]:
    """Validate explicit file arguments. Returns (valid paths, error messages).

    Every path is checked before anything is ingested, so a single bad filename
    aborts the run instead of leaving the collection half re-ingested.
    """
    paths: list[Path] = []
    errors: list[str] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.name.startswith("~$"):
            errors.append(f"{raw}: Word lock file, not a document")
        elif path.suffix.lower() != ".docx":
            errors.append(f"{raw}: not a .docx file")
        elif not path.is_file():
            errors.append(f"{raw}: does not exist")
        else:
            paths.append(path)
    return paths, errors


def ingest_files(paths: list[Path], base_url: str) -> dict[str, int]:
    """Ingest an explicit list of .docx files. Returns {filename: chunk_count}.

    doc_link is built from the basename only, so a file ingested from any folder
    gets the same link — and the same doc_id — as a full-folder run would give it.
    """
    init_collection()
    results = {}
    for path in paths:
        doc_link = f"{base_url}/{path.name}"
        count = ingest_document(path, doc_link)
        results[path.name] = count
        print(f"Ingested {path.name}: {count} chunks")

    return results


def ingest_folder(folder: Path, base_url: str) -> dict[str, int]:
    """Batch ingest all .docx files in a folder. Returns {filename: chunk_count}."""
    docx_files = sorted(f for f in folder.glob("*.docx") if not f.name.startswith("~$"))

    if not docx_files:
        print(f"No .docx files found in {folder}")
        return {}

    return ingest_files(docx_files, base_url)
