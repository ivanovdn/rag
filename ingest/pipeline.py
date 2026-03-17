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

    # Embed and store in Qdrant
    embeddings = embed_texts([c.text for c in chunks])
    upsert_chunks(chunks, embeddings)

    # Sync BM25 index
    if settings.bm25_enabled:
        from rag.bm25_index import add_chunks_to_bm25

        add_chunks_to_bm25(chunks)
        logger.info(f"Added {len(chunks)} chunks to BM25 index")

    return len(chunks)


def ingest_folder(folder: Path, base_url: str) -> dict[str, int]:
    """Batch ingest all .docx files in a folder. Returns {filename: chunk_count}."""
    init_collection()
    results = {}
    docx_files = sorted(folder.glob("*.docx"))

    if not docx_files:
        print(f"No .docx files found in {folder}")
        return results

    for docx_file in docx_files:
        doc_link = f"{base_url}/{docx_file.name}"
        count = ingest_document(docx_file, doc_link)
        results[docx_file.name] = count
        print(f"Ingested {docx_file.name}: {count} chunks")

    return results
