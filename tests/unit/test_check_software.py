import types

import pytest

import rag.tools.check_software as cs
from rag.software_registry import SoftwareRow

ROWS = [
    SoftwareRow(name="Docker", status="allowed", note="license required", source_list="Allowed Software"),
    SoftwareRow(name="TeamViewer", status="forbidden",
                category="Prohibited software - remote desktop control",
                note="risk of unauthorized access",
                alternative="Microsoft Teams remote desktop sharing and control",
                source_list="Forbidden Software"),
    SoftwareRow(name="Any personal/non-corporate VPN", status="forbidden",
                category="Prohibited software - personal VPN",
                note="Personal VPN must not be used for business",
                alternative="Corporate Cisco AnyConnect VPN",
                aliases=["ExpressVPN", "NordVPN"],
                source_list="Forbidden Software"),
]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    cs._name_index = None
    cs._software_unavailable = False
    cs._software_not_found = False
    cs._software_suggestion = None
    monkeypatch.setattr(cs, "scroll_all", lambda coll: [r.model_dump() for r in ROWS])


def _hit(row, score):
    return types.SimpleNamespace(payload=row.model_dump(), score=score)


def test_exact_allowed_returns_status(monkeypatch):
    out = cs.check_software("Docker")
    assert "[Software 1] Docker" in out
    assert "Status: allowed" in out
    assert cs._software_not_found is False


def test_forbidden_includes_alternative():
    out = cs.check_software("TeamViewer")
    assert "Status: forbidden" in out
    assert "Alternative: Microsoft Teams" in out


def test_alias_resolves():
    out = cs.check_software("NordVPN")
    assert "Status: forbidden" in out
    assert "Cisco AnyConnect" in out


def test_category_query_uses_semantic_when_name_misses(monkeypatch):
    # name lookup misses "vpn"; semantic returns the VPN forbidden row above the floor
    monkeypatch.setattr(cs, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(cs, "search_vectors",
                        lambda qv, top_k, collection_name: [_hit(ROWS[2], 0.9)])
    out = cs.check_software("VPN")
    assert "Status: forbidden" in out and "Cisco AnyConnect" in out


def test_not_listed_when_semantic_below_floor(monkeypatch):
    monkeypatch.setattr(cs, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(cs, "search_vectors",
                        lambda qv, top_k, collection_name: [_hit(ROWS[0], 0.1)])  # below floor
    out = cs.check_software("SomeNicheToolXYZ")
    assert out.startswith("SOFTWARE_NOT_LISTED")
    assert cs._software_not_found is True


def test_transient_embed_failure_sets_unavailable(monkeypatch):
    def boom(_q):
        raise ConnectionError("embedding server down")
    monkeypatch.setattr(cs, "embed_query", boom)
    out = cs.check_software("SomethingNotAName")  # forces semantic path
    assert out == "SOFTWARE_LOOKUP_UNAVAILABLE"
    assert cs._software_unavailable is True
