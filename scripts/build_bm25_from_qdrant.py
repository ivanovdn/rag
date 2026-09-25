"""Rebuild the BM25 keyword index from the live Qdrant collection.

Why from Qdrant and not from the .docx files: chunk_id is a fresh uuid4 per
ingest (ingest/docx_parser.py), so re-parsing the documents produces IDs that do
not exist in the collection. Hybrid search fuses the two result lists by
chunk_id, so an index built from documents would be silently disjoint from
vector search — which is exactly the state this script was written to repair
(an index from 2026-04-09 matched Qdrant's chunk COUNT exactly, 1602, while
0 of 6 sampled IDs actually existed in the collection).

Qdrant's payload already carries every field the index stores, including the
text, so this is a faithful rebuild rather than a re-derivation.

READ-ONLY with respect to Qdrant and the shared host: it scrolls points and
writes one local JSON file. It never upserts, never deletes, and never touches
the collection's configuration.

    PYTHONPATH=. python scripts/build_bm25_from_qdrant.py
"""

import argparse
import sys

from config import settings
from ingest.chunk_models import PolicyChunk
from rag.bm25_index import get_bm25_index
from rag.vector_store import get_qdrant_client


def fetch_all_chunks(collection: str, batch: int = 512) -> list[PolicyChunk]:
    client = get_qdrant_client()
    chunks: list[PolicyChunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            if not pl.get("chunk_id") or not pl.get("text"):
                continue
            chunks.append(
                PolicyChunk(
                    chunk_id=pl["chunk_id"],
                    doc_id=pl.get("doc_id", ""),
                    doc_title=pl.get("doc_title", ""),
                    doc_filename=pl.get("doc_filename", ""),
                    doc_link=pl.get("doc_link", ""),
                    section=pl.get("section", ""),
                    section_number=pl.get("section_number", ""),
                    clause=pl.get("clause", ""),
                    clause_number=pl.get("clause_number", ""),
                    section_display=pl.get("section_display", ""),
                    text=pl["text"],
                )
            )
        if offset is None:
            break
    return chunks


def verify(collection: str, sample_size: int = 25) -> int:
    """Report whether an existing index actually matches this Qdrant.

    The failure this catches is silent and expensive: chunk_id is a fresh uuid4
    per ingest, so an index built against a DIFFERENT ingestion of the same
    corpus has the right chunk COUNT and entirely wrong IDs. Hybrid search then
    fuses two disjoint lists and the keyword half contributes nothing, which
    looks like "BM25 does not help" rather than "BM25 never ran".
    """
    import random

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    index = get_bm25_index()
    if not index.documents:
        print("  No existing index loaded — nothing to verify.")
        return 1

    client = get_qdrant_client()
    keys = list(index.documents)
    random.seed(0)
    sample = random.sample(keys, min(sample_size, len(keys)))
    hits = 0
    for cid in sample:
        points, _ = client.scroll(
            collection_name=collection, limit=1, with_payload=False,
            scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchValue(value=cid))]),
        )
        hits += bool(points)

    print(f"  Index chunks   : {len(keys)}")
    print(f"  Sampled        : {len(sample)}")
    print(f"  Found in Qdrant: {hits}")
    if hits == len(sample):
        print("  OK — the index matches this collection.")
        return 0
    print("  MISMATCH — this index was built against a different ingestion.")
    print("  Hybrid search would fuse two disjoint lists and the keyword half")
    print("  would contribute nothing. Rebuild with this script (no --verify).")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default=settings.qdrant_collection)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be indexed without writing the file")
    ap.add_argument("--verify", action="store_true",
                    help="check an EXISTING index against this Qdrant and report the overlap")
    args = ap.parse_args()

    print(f"  Qdrant     : {settings.active_qdrant_url}")
    print(f"  Collection : {args.collection}")

    if args.verify:
        return verify(args.collection)

    chunks = fetch_all_chunks(args.collection)
    print(f"  Chunks read: {len(chunks)}")
    if not chunks:
        print("  ERROR: no chunks found — refusing to write an empty index.")
        return 1

    docs = len({c.doc_id for c in chunks})
    print(f"  Documents  : {docs}")

    if args.dry_run:
        print("  --dry-run: nothing written.")
        return 0

    index = get_bm25_index()
    index.documents.clear()
    index.inverted_index.clear()
    index.add_chunks(chunks)
    index.save()
    print(f"  Wrote index with {len(index.documents)} chunks.")
    print("\n  NOTE: this file is gitignored and is NOT copied into the Docker image.")
    print("  Mount it or copy it to the host that runs with BM25_ENABLED=true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
