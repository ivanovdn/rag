import pytest

import config


@pytest.fixture(autouse=True)
def _no_network_hops_by_default(monkeypatch):
    """Pin every switch that adds a network hop to its off value.

    Settings reads .env and the environment, and a developer's .env points at
    the shared model host -- so a QUERY_REWRITE=multi left set while trying the
    feature made 17 unit tests send real LLM requests (branch review,
    2026-10-09). The env var is removed too, for tests that build a fresh
    Settings(). A test that wants the rewrite sets it itself.
    """
    monkeypatch.delenv("QUERY_REWRITE", raising=False)
    monkeypatch.setattr(config.settings, "query_rewrite", "off")
