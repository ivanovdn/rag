"""Every expected citation in the chatbot datasets names a clause that exists.

A label with a typo can never be cited, so it reads as a model failure forever.
Matching mirrors eval/evaluators._match_result: doc exact (case-insensitive),
section as a substring, the clause label as the START of the clause name, an
empty field matching anything. Labels are also held to the shapes the graders
can score correctly (review 2026-10-16): a clause label names ONE clause, no
label is document-only, and a case never mixes clause- and section-level
alternatives.
"""

import json
from pathlib import Path

import pytest

from eval.evaluators import _clause_match
from ingest.docx_parser import parse_docx
from tests._corpus import policy_docx_files

_DOCX = policy_docx_files()
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not _DOCX, reason="policy corpus not present (gitignored/local-only)"),
]
DATASETS = ["eval/datasets/chatbot_test_cases.json", "eval/datasets/role_mismatch_cases.json"]


@pytest.fixture(scope="module")
def chunks():
    return [c for p in _DOCX for c in parse_docx(p, doc_link="")]


def _matches(label: dict, chunks) -> set[tuple[str, str, str]]:
    """Distinct (doc, section, clause) names a label matches."""
    doc, sec, cl = (label.get(k, "").strip() for k in ("doc_id", "section", "clause"))
    return {
        (c.doc_title, c.section or "", c.clause or "")
        for c in chunks
        if (not doc or doc.lower() == c.doc_title.strip().lower())
        and (not sec or sec.lower() in (c.section or "").lower())
        and (not cl or _clause_match(cl, c.clause or ""))
    }


def _label_exists(label: dict, chunks) -> bool:
    return bool(_matches(label, chunks))


def _cases(path):
    if not Path(path).exists():
        pytest.skip(f"{path} not present (gitignored/local-only)")
    return json.loads(Path(path).read_text(encoding="utf-8"))["test_cases"]


@pytest.mark.parametrize("path", DATASETS)
def test_every_expected_citation_names_a_real_clause(path, chunks):
    cases = _cases(path)
    missing = [(c["id"], e) for c in cases for e in c["expected_citations"]
               if not _label_exists(e, chunks)]
    assert not missing, missing


@pytest.mark.parametrize("path", DATASETS)
def test_a_clause_label_names_exactly_one_clause(path, chunks):
    """'Acceptable Use' once also matched 'Unacceptable Use' in the same section."""
    ambiguous = [(c["id"], e["clause"], sorted(m[2][:40] for m in _matches(e, chunks)))
                 for c in _cases(path) for e in c["expected_citations"]
                 if e.get("clause", "").strip()
                 and len({m[1:] for m in _matches(e, chunks)}) > 1]
    assert not ambiguous, ambiguous


@pytest.mark.parametrize("path", DATASETS)
def test_labels_have_a_shape_the_graders_score_correctly(path):
    """A document-only label grades as 0/0 = cited under 'all'; a case mixing
    clause- and section-level alternatives drops the section-level one."""
    bad = []
    for c in _cases(path):
        depths = {bool(e.get("clause", "").strip()) for e in c["expected_citations"]}
        if any(not e.get("section", "").strip() and not e.get("clause", "").strip()
               for e in c["expected_citations"]):
            bad.append((c["id"], "document-only label"))
        if len(depths) > 1:
            bad.append((c["id"], "mixes clause- and section-level labels"))
    assert not bad, bad
