from pathlib import Path

from ingest.docx_parser import parse_docx
from rag.embeddings import embed_texts
from rag.vector_store import delete_document, init_collection, upsert_chunks


def ingest_document(filepath: Path, doc_link: str) -> int:
    """Parse a DOCX file, embed chunks, and upsert to Qdrant."""
    init_collection()
    chunks = parse_docx(filepath, doc_link)
    if not chunks:
        return 0

    # Delete existing chunks for this doc (re-ingestion)
    delete_document(chunks[0].doc_id)

    embeddings = embed_texts([c.text for c in chunks])
    upsert_chunks(chunks, embeddings)
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
