import pytest

from ingest.docx_parser import parse_docx
from scripts.parse_coverage import compute_parse_stats
from tests._corpus import policy_docx_files

_DOCX = policy_docx_files()
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not _DOCX, reason="policy corpus not present (gitignored/local-only)"),
]


@pytest.mark.parametrize("docx_path", _DOCX, ids=[p.stem for p in _DOCX])
def test_doc_parses_to_valid_chunks(docx_path):
    chunks = parse_docx(docx_path, doc_link="http://test/policy")
    assert len(chunks) >= 1, f"{docx_path.name} produced no chunks"
    for c in chunks:
        assert c.text.strip(), f"empty chunk text in {docx_path.name}"
        assert c.doc_id, f"missing doc_id in {docx_path.name}"
        assert c.doc_title, f"missing doc_title in {docx_path.name}"


def test_corpus_stats_are_sane():
    all_chunks = []
    for path in _DOCX:
        all_chunks.extend(parse_docx(path, doc_link=""))
    stats = compute_parse_stats(all_chunks)
    assert stats["chunk_count"] >= len(_DOCX)  # >= 1 chunk per doc
    assert stats["empty_text"] == 0
