"""CLI: ingest software/software_registry.json into the software_registry collection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()  # must precede any LlamaIndex/embedding import

from config import settings  # noqa: E402
from rag.embeddings import embed_texts  # noqa: E402
from rag.software_registry import load_registry, software_passage_text  # noqa: E402
from rag.vector_store import init_software_collection, upsert_software_rows  # noqa: E402


def main():
    rows = load_registry(settings.software_registry_path)
    if not rows:
        print("No software rows found.")
        return
    init_software_collection()
    embed_inputs = [software_passage_text(r) for r in rows]
    embeddings = embed_texts(embed_inputs)
    upsert_software_rows([r.model_dump() for r in rows], embeddings)
    print(f"Ingested {len(rows)} software rows into '{settings.software_collection}'.")


if __name__ == "__main__":
    main()
