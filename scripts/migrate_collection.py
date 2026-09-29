#!/usr/bin/env python
"""Copy a collection's points into a new, sparse-enabled collection.

Why copy instead of re-ingesting the .docx corpus: this reuses the dense vectors
exactly as they are, so the dense half of retrieval is provably unchanged and any
eval delta is attributable to the sparse half alone. It also needs no Ollama call
and no access to the source documents.

Usage:
    PYTHONPATH=. python scripts/migrate_collection.py --target compliance_policies_v2 --dry-run
    PYTHONPATH=. python scripts/migrate_collection.py --target compliance_policies_v2
    PYTHONPATH=. python scripts/migrate_collection.py --target compliance_policies_v2 --verify
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import PointStruct

from config import settings
from rag.observability import init_observability
from rag.vector_store import (
    SPARSE_VECTOR_NAME,
    bm25_document,
    get_qdrant_client,
    init_collection,
)

SCROLL_PAGE = 500
UPSERT_BATCH = 100
VERIFY_SAMPLE = 25


def _dense_of(point) -> list[float]:
    """The source point's dense vector.

    qdrant-client returns a bare list for an unnamed vector and a dict keyed by
    name when the collection has named ones. Accept both, so re-running against
    an already-migrated collection works instead of writing a dict as a vector.
    """
    vector = point.vector
    return vector[""] if isinstance(vector, dict) else vector


def migrate(source: str, target: str, dry_run: bool) -> int:
    """Copy every point from `source` into `target`, adding a sparse vector."""
    # Which Qdrant, spelled out before anything else. get_qdrant_client()
    # resolves the URL from USE_REMOTE_QDRANT, so running this from a laptop with
    # that false builds the new collection in a LOCAL Qdrant -- and --verify then
    # checks that same local Qdrant and reports a fully green 1602/1602 for a
    # collection production cannot see. Local and remote are separate stores;
    # build on the host that will read it.
    print(f"qdrant:       {settings.active_qdrant_url}")
    client = get_qdrant_client()
    total = client.count(collection_name=source, exact=True).count

    if dry_run:
        print(f"[dry-run] source {source}: {total} points")
        print(f"[dry-run] target {target} exists: {client.collection_exists(target)}")
        print(f"[dry-run] would create {target} and copy {total} points")
        return 0

    init_collection(target)

    copied = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=source,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        batch = [
            PointStruct(
                id=p.id,
                vector={
                    "": _dense_of(p),
                    SPARSE_VECTOR_NAME: bm25_document(p.payload.get("text", "")),
                },
                payload=p.payload,
            )
            for p in points
        ]
        for i in range(0, len(batch), UPSERT_BATCH):
            client.upsert(
                collection_name=target, points=batch[i : i + UPSERT_BATCH], wait=True
            )
        copied += len(batch)
        print(f"  copied {copied}/{total}")

        if offset is None:
            break

    return copied


def verify(source: str, target: str) -> bool:
    """Check count parity, the sparse schema, dense/payload fidelity, and sparse
    presence on a sample of ids.

    Count parity, schema presence, and non-empty sparse vectors would all pass
    even if a dense vector were zeroed or scrambled in the copy — they say
    nothing about the dense half. The dense-vector check below is what actually
    protects the premise this whole script exists for: that copying rather than
    re-ingesting leaves the dense half byte-identical, so an eval delta on the
    new collection is attributable to the sparse half alone.
    """
    # First line, same as migrate(): a green report says nothing about WHICH
    # Qdrant it is green on, and verify() resolves the same URL the migration
    # did -- so a migration built locally verifies locally, perfectly.
    print(f"qdrant:       {settings.active_qdrant_url}")
    client = get_qdrant_client()

    src = client.count(collection_name=source, exact=True).count
    tgt = client.count(collection_name=target, exact=True).count
    counts_ok = src == tgt
    print(f"count:        source={src} target={tgt} {'OK' if counts_ok else 'MISMATCH'}")

    sparse = client.get_collection(target).config.params.sparse_vectors or {}
    schema_ok = SPARSE_VECTOR_NAME in sparse
    print(f"sparse config: {sorted(sparse) or 'none'} {'OK' if schema_ok else 'MISSING'}")

    sample, _ = client.scroll(
        collection_name=source, limit=VERIFY_SAMPLE, with_payload=True, with_vectors=True
    )
    ids = [p.id for p in sample]
    found = client.retrieve(
        collection_name=target, ids=ids, with_payload=True, with_vectors=True
    )
    found_by_id = {p.id: p for p in found}

    present = sum(
        1
        for p in found
        if isinstance(p.vector, dict) and p.vector.get(SPARSE_VECTOR_NAME)
    )
    sample_ok = present == len(ids)
    print(f"sampled ids:   {present}/{len(ids)} present with a non-empty sparse vector")

    # _dense_of is reused on both sides (not a raw == on .vector) so a bare-list
    # source point and an already-migrated, named-vector target point compare
    # equal instead of a list-vs-dict false negative.
    dense_matches = 0
    payload_matches = 0
    for p in sample:
        t = found_by_id.get(p.id)
        if t is None:
            continue
        if _dense_of(t) == _dense_of(p):
            dense_matches += 1
        if t.payload == p.payload:
            payload_matches += 1
    dense_ok = dense_matches == len(ids)
    payload_ok = payload_matches == len(ids)
    print(f"dense vectors:  {dense_matches}/{len(ids)} identical")
    print(f"payloads:       {payload_matches}/{len(ids)} identical")

    return counts_ok and schema_ok and sample_ok and dense_ok and payload_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy points into a sparse-enabled Qdrant collection",
    )
    parser.add_argument("--source", default=settings.qdrant_collection)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="check an existing target")
    parser.add_argument(
        "--force", action="store_true", help="write into a non-empty target"
    )
    args = parser.parse_args()

    # After parse_args(), not as the first statement of main(): argparse exits
    # from inside parse_args() itself on --help or a usage error, so calling
    # this any earlier would register a global TracerProvider and attempt a
    # network export before help text even prints. Still not at module top:
    # this script imports no LlamaIndex or Ollama code (it copies vectors, it
    # never embeds), so there is nothing for the usual ordering rule to order,
    # and the module stays importable in tests with no side effect regardless
    # of exactly where in main() this sits. If this script ever gains a
    # LlamaIndex or Ollama import, this call must move above that import.
    init_observability()

    if args.source == args.target:
        print("ERROR: --source and --target must differ")
        return 1

    client = get_qdrant_client()

    if args.verify:
        return 0 if verify(args.source, args.target) else 1

    if (
        not args.dry_run
        and not args.force
        and client.collection_exists(args.target)
        and client.count(collection_name=args.target, exact=True).count
    ):
        print(f"ERROR: {args.target} already has points. Use --force to write into it.")
        return 1

    copied = migrate(args.source, args.target, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nCopied {copied} points into {args.target}")
        print(f"Verify with: --target {args.target} --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
