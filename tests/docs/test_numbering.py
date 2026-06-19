import pytest
from docx import Document

from ingest.numbering import NumberingResolver
from tests._corpus import policy_docx_files

_DOCX = policy_docx_files()
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not _DOCX, reason="policy corpus not present (gitignored/local-only)"),
]


def _level0_decimals(path):
    """Walk a doc, assert per-paragraph numbering invariants, and return the
    sequence of leading integers for level-0 decimal paragraphs."""
    doc = Document(path)
    resolver = NumberingResolver(doc)
    seq = []
    for element in doc.element.body:
        info = resolver.resolve(element)
        if info is None:
            continue
        # structural invariants for every numbered paragraph
        assert info["ilvl"] >= 0
        assert isinstance(info["numFmt"], str) and info["numFmt"]
        # Collect level-0 decimal section numbers for the continuity check below.
        # A decimal level may legitimately have an empty lvlText in some real docs,
        # yielding an empty resolved number; such paragraphs are simply skipped
        # here via the isdigit() guard rather than asserted non-empty.
        if info["numFmt"] == "decimal" and info["ilvl"] == 0:
            leading = info["resolved"].strip().rstrip(".").split(".")[0]
            if leading.isdigit():
                seq.append(int(leading))
    return seq


@pytest.mark.parametrize("docx_path", _DOCX, ids=[p.stem for p in _DOCX])
def test_numbering_resolves_structurally(docx_path):
    _level0_decimals(docx_path)  # assertions happen inside


def test_level0_decimal_numbering_is_continuous():
    """The resolver continues top-level decimal numbering across numIds, so the
    level-0 sequence should be non-decreasing and actually progress — never
    reset to 1 per numId (the documented gotcha)."""
    checked = 0
    for path in _DOCX:
        seq = _level0_decimals(path)
        if len(seq) >= 2:
            checked += 1
            assert seq == sorted(seq), f"level-0 numbering not non-decreasing in {path.name}: {seq}"
            assert seq[-1] > seq[0], f"level-0 numbering did not progress in {path.name}: {seq}"
    if checked == 0:
        pytest.skip("no doc with >=2 level-0 decimal sections")
