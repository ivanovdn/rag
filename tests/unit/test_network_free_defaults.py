"""Unit tests never reach the model host, whatever the developer's .env says.

Settings reads .env and the process environment, and the local .env points at
the shared Spark host. A switch that adds a network hop (QUERY_REWRITE) must be
pinned off for every unit test; a test that wants it on sets it explicitly.
Found by the 2026-10-09 branch review: with QUERY_REWRITE=multi set, 17 unit
tests sent real LLM requests.
"""

import config
import rag.tools.search_policies as sp


def test_query_rewrite_is_off_for_every_unit_test():
    assert config.settings.query_rewrite == "off"
    assert sp.settings.query_rewrite == "off"
