"""role-mismatch-v1: four traps and two controls (#10, step 3).

One failing case cannot show a fix worked; the controls catch a prompt rule
that over-corrects into never citing a clause addressed to another role.
The file is gitignored with the other datasets, so this skips without it.
"""

import json
from pathlib import Path

import pytest

from scripts.make_dataset import detect_tier, map_chatbot

_PATH = Path("eval/datasets/role_mismatch_cases.json")
pytestmark = pytest.mark.skipif(not _PATH.exists(), reason="dataset not present (gitignored/local-only)")
CASES = json.loads(_PATH.read_text(encoding="utf-8"))["test_cases"] if _PATH.exists() else []


def test_it_uploads_as_a_chatbot_dataset():
    assert detect_tier(CASES) == "chatbot"
    inputs, outputs, _ = map_chatbot(CASES)
    assert len(inputs) == 6 and all(o["expected_citations"] for o in outputs)


def test_ids_are_unique_and_the_controls_are_marked():
    assert [c["id"] for c in CASES] == [f"RM-00{i}" for i in range(1, 7)]
    assert {c["id"] for c in CASES if c.get("control")} == {"RM-005", "RM-006"}
