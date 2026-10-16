"""Every expected citation in the chatbot datasets names a clause that exists.

A label with a typo can never be cited, so it reads as a model failure forever.
Matching mirrors eval/evaluators._match_result: doc exact (case-insensitive),
section and clause as substrings, an empty field matching anything.
"""

import json
from pathlib import Path

import pytest

from ingest.docx_parser import parse_docx
from tests._corpus import policy_docx_files

_DOCX = policy_docx_files()
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not _DOCX, reason="policy corpus not present (gitignored/local-only)"),
]
DATASETS = ["eval/datasets/chatbot_test_cases.json"]


@pytest.fixture(scope="module")
def chunks():
    return [c for p in _DOCX for c in parse_docx(p, doc_link="")]


def _label_exists(label: dict, chunks) -> bool:
    doc, sec, cl = (label.get(k, "").strip().lower() for k in ("doc_id", "section", "clause"))
    return any(
        (not doc or doc == c.doc_title.strip().lower())
        and (not sec or sec in (c.section or "").lower())
        and (not cl or cl in (c.clause or "").lower())
        for c in chunks
    )


@pytest.mark.parametrize("path", DATASETS)
def test_every_expected_citation_names_a_real_clause(path, chunks):
    if not Path(path).exists():
        pytest.skip(f"{path} not present (gitignored/local-only)")
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["test_cases"]
    missing = [(c["id"], e) for c in cases for e in c["expected_citations"]
               if not _label_exists(e, chunks)]
    assert not missing, missing
