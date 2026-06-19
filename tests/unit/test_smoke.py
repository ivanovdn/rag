"""Sanity: the suite collects and every unit under test imports cleanly."""
import importlib

import pytest


@pytest.mark.parametrize("module", [
    "rag.response",
    "eval.evaluators",
    "ingest.docx_parser",
    "ingest.numbering",
    "channels.teams.renderer",
    "channels.teams.utils",
])
def test_module_imports(module):
    assert importlib.import_module(module) is not None
