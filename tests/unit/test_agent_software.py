from rag.agent import Citation, ALL_TOOLS


def test_citation_software_fields_default_empty():
    c = Citation(doc_title="Allowed Software", section="", clause="",
                 clause_number="", quote="license required")
    assert c.status == ""
    assert c.alternative == ""


def test_citation_accepts_software_status():
    c = Citation(doc_title="Forbidden Software", section="remote desktop", clause="",
                 clause_number="", quote="risk", status="forbidden",
                 alternative="Microsoft Teams")
    assert c.status == "forbidden"
    assert c.alternative == "Microsoft Teams"


def test_check_software_tool_registered():
    names = {getattr(t.metadata, "name", None) for t in ALL_TOOLS}
    assert "check_software" in names
