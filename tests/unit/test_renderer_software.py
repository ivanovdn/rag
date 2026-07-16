from channels.teams.renderer import render_answer, render_software_not_listed


def test_render_allowed_software():
    result = {"answer": "", "citations": [
        {"doc_title": "Allowed Software", "section": "", "clause": "", "clause_number": "",
         "quote": "A license is required for commercial usage", "status": "allowed", "alternative": ""},
    ]}
    html = render_answer(result)
    assert "✅ Allowed" in html
    assert "Allowed Software" in html
    assert '"A license is required for commercial usage"' in html
    assert "<div" not in html


def test_render_forbidden_software_with_alternative():
    result = {"answer": "", "citations": [
        {"doc_title": "Forbidden Software", "section": "remote desktop control", "clause": "",
         "clause_number": "", "quote": "risk of unauthorized access", "status": "forbidden",
         "alternative": "Microsoft Teams remote desktop sharing"},
    ]}
    html = render_answer(result)
    assert "⛔ Forbidden" in html
    assert "remote desktop control" in html
    assert "<b>Alternative:</b> Microsoft Teams remote desktop sharing" in html


def test_policy_citation_still_renders_old_way():
    result = {"answer": "", "citations": [
        {"doc_title": "AUP", "section": "Use", "clause": "Email", "clause_number": "4.7",
         "quote": "No spam."},
    ]}
    html = render_answer(result)
    assert "📄 AUP" in html
    assert "✅" not in html and "⛔" not in html


def test_not_listed_with_suggestion():
    html = render_software_not_listed("Dockerr", {"name": "Docker", "status": "allowed"})
    assert "isn't on the approved or forbidden software list" in html
    assert "Docker" in html and "Allowed" in html
    assert "support@trinetix.com" in html
    assert "<div" not in html


def test_not_listed_without_suggestion():
    html = render_software_not_listed("ZzzTool", None)
    assert "ZzzTool" in html
    assert "support@trinetix.com" in html
    assert "Did you mean" not in html
