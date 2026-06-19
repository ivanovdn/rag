from channels.teams.renderer import render_answer, render_escalation, render_error


def test_render_answer_no_citations_uses_prose():
    assert render_answer({"answer": "Just prose.", "citations": []}) == "<p>Just prose.</p>"


def test_render_answer_single_citation_structure():
    result = {"answer": "A", "citations": [
        {"doc_title": "AUP", "section": "Use", "clause": "Email", "clause_number": "4.7", "quote": "No spam."},
    ]}
    html = render_answer(result)
    assert "📄 AUP" in html
    assert "<b>Section:</b> Use" in html
    assert "<b>Clause 4.7:</b> Email" in html
    assert '<i>"No spam."</i>' in html
    assert "<div" not in html  # only Teams-safe tags


def test_render_answer_multi_citation_does_not_repeat_answer_text():
    result = {"answer": "DUPLICATE_ME", "citations": [
        {"doc_title": "A", "quote": "q1"},
        {"doc_title": "B", "quote": "q2"},
    ]}
    html = render_answer(result)
    assert "This is addressed in 2 policies" in html
    assert html.count("<hr>") == 1
    assert "DUPLICATE_ME" not in html  # the prose answer is NOT duplicated when citations exist


def test_render_answer_clause_equal_to_number_shows_number_only():
    result = {"answer": "", "citations": [
        {"doc_title": "A", "clause": "4.7", "clause_number": "4.7", "quote": "q"},
    ]}
    html = render_answer(result)
    assert "<b>Clause 4.7</b>" in html
    assert "Clause 4.7:" not in html  # not rendered as the name form


def test_render_escalation_includes_question_and_reason():
    html = render_escalation("Can I install software?", {"escalation": {"reason": "No policy found."}})
    assert "Escalated to Compliance Team" in html
    assert "Can I install software?" in html
    assert "No policy found." in html


def test_render_error_includes_error_text():
    html = render_error("Q?", "boom")
    assert "Compliance lookup failed" in html
    assert "boom" in html
