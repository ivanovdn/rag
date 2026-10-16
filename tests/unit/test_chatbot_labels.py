"""The relabelled cases accept every clause that answers them (#10, step 2).

Checked on the MAPPED output, not only the JSON file: match_mode silently
dropped on upload made "any" a no-op until 2026-09-25 (review focus 5).

The dataset files are gitignored with the policies (they quote internal policy),
so this skips on a checkout without them, as the corpus tests do.
"""

import json
from pathlib import Path

import pytest

from scripts.make_dataset import map_chatbot

_PATH = Path("eval/datasets/chatbot_test_cases.json")
pytestmark = pytest.mark.skipif(not _PATH.exists(), reason="dataset not present (gitignored/local-only)")
CASES = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.exists() else {"test_cases": []}
BY_ID = {c["id"]: c for c in CASES["test_cases"]}
RELABELLED = {
    "CB-014": "Chat Messages",
    "CB-017": "Unacceptable Use",
    "CB-032": "Inactivity Logoff/Lockout",
    "CB-047": "Exceptions",
}


def test_each_relabelled_case_accepts_its_alternative_under_any():
    _, outputs, meta = map_chatbot([BY_ID[i] for i in RELABELLED])
    for (case_id, added), out in zip(RELABELLED.items(), outputs):
        assert out["match_mode"] == "any", case_id
        assert added in [c["clause"] for c in out["expected_citations"]], case_id
        assert len(out["expected_citations"]) == 2, case_id


def test_the_set_is_still_61_unique_questions():
    ids = [c["id"] for c in CASES["test_cases"]]
    assert len(ids) == 61 and len(set(ids)) == 61
    assert CASES["metadata"]["version"] == "2.0"
