"""Probe whether the software_registry collection is ingested, so live software
tests auto-skip on machines without the ingested stack."""

from config import settings


def software_registry_ready() -> bool:
    try:
        from rag.vector_store import get_qdrant_client

        client = get_qdrant_client()
        if not client.collection_exists(settings.software_collection):
            return False
        return client.count(settings.software_collection).count > 0
    except Exception:
        return False
