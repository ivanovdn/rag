"""CLI: ingest policy DOCX files into Qdrant — the whole folder, or selected files."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()  # Must be before any LlamaIndex imports

from config import settings
from ingest.pipeline import ingest_files, ingest_folder, resolve_docx_paths
from rag.vector_store import assert_sparse_vector, init_collection


def main():
    parser = argparse.ArgumentParser(
        description="Ingest policy DOCX files into Qdrant",
        epilog=(
            "Examples:\n"
            "  # whole folder\n"
            "  ingest_all.py --folder ./policies\n"
            "  # selected files (tab-complete the names; --folder is ignored)\n"
            "  ingest_all.py 'policies/Backup Policy [Internal].docx'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific .docx files to ingest; omit to ingest the whole --folder",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=settings.policy_docs_folder,
        help="Folder containing .docx files",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=settings.policy_base_url,
        help="Base URL for document links",
    )
    args = parser.parse_args()

    # Fail here, before a single document is touched. ingest_document repeats
    # this check per document because that is where the delete-then-upsert
    # window is; this call is what turns "one policy silently vanished" into a
    # message the operator reads before starting. init_collection() first, so a
    # first-ever run creates the collection (already sparse-enabled) rather than
    # asserting against one that does not exist yet.
    init_collection()
    assert_sparse_vector(
        settings.qdrant_collection,
        "Ingest writes a sparse vector for every chunk.",
    )

    if args.files:
        # Validate every path before ingesting anything, so one bad filename
        # can't leave the collection half re-ingested.
        paths, errors = resolve_docx_paths(args.files)
        if errors:
            print("Error: invalid file arguments")
            for err in errors:
                print(f"  {err}")
            sys.exit(1)

        print(f"Ingesting {len(paths)} selected document(s)")
        print(f"Base URL: {args.base_url}")
        print()
        results = ingest_files(paths, args.base_url)
    else:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Error: folder '{folder}' does not exist")
            sys.exit(1)

        print(f"Ingesting documents from: {folder}")
        print(f"Base URL: {args.base_url}")
        print()
        results = ingest_folder(folder, args.base_url)

    print()
    print(f"Done. Ingested {len(results)} documents, {sum(results.values())} total chunks.")


if __name__ == "__main__":
    main()
