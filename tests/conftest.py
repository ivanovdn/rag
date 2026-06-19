import pytest

from ingest.chunk_models import PolicyChunk


@pytest.fixture
def make_chunk():
    """Factory for a valid PolicyChunk with sensible defaults; override any field."""
    def _make(**overrides) -> PolicyChunk:
        defaults = dict(
            chunk_id="c1",
            doc_id="doc",
            doc_title="Doc Title",
            doc_filename="doc.docx",
            doc_link="http://example/doc",
            text="some text",
        )
        defaults.update(overrides)
        return PolicyChunk(**defaults)
    return _make
