from eval.evaluators import (
    hit_evaluator,
    mrr_evaluator,
    citation_clause_accuracy,
    json_parse_success,
)


def _output(search_results=None, citations=None, **extra):
    out = {"search_results": search_results or [], "citations": citations or []}
    out.update(extra)
    return out


def test_hit_all_mode_pass():
    out = _output(search_results=[{"doc_title": "AUP", "section": "Acceptable Use", "clause": "Email"}])
    exp = {"expected_doc": "AUP", "expected_section": "Acceptable Use", "expected_clause": "Email"}
    assert hit_evaluator(out, exp)["score"] == 1.0


def test_hit_matches_section_as_substring_not_section_display():
    # expected "Private Information" matches inside the chunk's "7. Private Information"
    out = _output(search_results=[{"doc_title": "AUP", "section": "7. Private Information", "clause": "Blogging"}])
    exp = {"expected_doc": "AUP", "expected_section": "Private Information", "expected_clause": "Blogging"}
    assert hit_evaluator(out, exp)["score"] == 1.0


def test_hit_all_mode_miss():
    out = _output(search_results=[{"doc_title": "Other", "section": "X", "clause": "Y"}])
    exp = {"expected_doc": "AUP", "expected_section": "Acceptable Use", "expected_clause": "Email"}
    assert hit_evaluator(out, exp)["score"] == 0.0


def test_hit_any_mode_passes_when_one_alt_matches():
    out = _output(search_results=[{"doc_title": "AUP", "section": "Acceptable Use", "clause": "Email"}])
    exp = {"match_mode": "any", "expected_citations": [
        {"doc_id": "Wrong Doc", "section": "Z", "clause": "Z"},
        {"doc_id": "AUP", "section": "Acceptable Use", "clause": "Email"},
    ]}
    assert hit_evaluator(out, exp)["score"] == 1.0


def test_mrr_first_match_at_rank_two():
    out = _output(search_results=[
        {"doc_title": "X", "section": "", "clause": ""},
        {"doc_title": "AUP", "section": "Use", "clause": "Email"},
    ])
    exp = {"expected_doc": "AUP", "expected_section": "Use", "expected_clause": "Email"}
    assert mrr_evaluator(out, exp)["score"] == 0.5


def test_citation_clause_any_mode_substring():
    out = _output(citations=[{"doc_title": "AUP", "section": "Acceptable Use", "clause": "Email Use"}])
    exp = {"match_mode": "any", "expected_citations": [
        {"doc_id": "AUP", "section": "Acceptable Use", "clause": "Email"},
    ]}
    assert citation_clause_accuracy(out, exp)["score"] == 1.0


def test_citation_clause_skips_when_no_expected_clause():
    out = _output(citations=[])
    exp = {"expected_doc": "AUP", "expected_section": "Use"}
    result = citation_clause_accuracy(out, exp)
    assert result["score"] == 1.0
    assert result["label"] == "skip"


def test_json_parse_success_true_and_false():
    assert json_parse_success({"parse_success": True}, {})["score"] == 1.0
    assert json_parse_success({"parse_success": False, "raw_response": "x"}, {})["score"] == 0.0


def test_evaluator_handles_none_output():
    result = hit_evaluator(None, {})
    assert result["score"] == 0.0
    assert result["label"] == "error"
