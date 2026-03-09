"""CLI: ingest all DOCX files from a folder into Qdrant."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()  # Must be before any LlamaIndex imports

from config import settings
from ingest.pipeline import ingest_folder


def main():
    parser = argparse.ArgumentParser(description="Ingest policy DOCX files into Qdrant")
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
