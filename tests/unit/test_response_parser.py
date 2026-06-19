from rag.response import parse_agent_response, _extract_json


def test_parses_clean_json_object():
    raw = '{"answer": "Yes", "citations": [{"doc_title": "AUP"}], "escalation": {"needed": false, "reason": ""}}'
    result = parse_agent_response(raw)
    assert result["parse_success"] is True
    assert result["answer"] == "Yes"
    assert result["citations"] == [{"doc_title": "AUP"}]
    assert result["escalation"] == {"needed": False, "reason": ""}


def test_parses_json_in_code_fence():
    raw = '```json\n{"answer": "Fenced", "citations": []}\n```'
    result = parse_agent_response(raw)
    assert result["parse_success"] is True
    assert result["answer"] == "Fenced"


def test_extracts_json_embedded_in_prose():
    raw = 'Here is the result: {"answer": "Embedded", "citations": []} hope that helps'
    result = parse_agent_response(raw)
    assert result["parse_success"] is True
    assert result["answer"] == "Embedded"


def test_malformed_json_falls_back_to_raw():
    raw = '{"answer": "broken"'  # missing closing brace -> invalid JSON
    result = parse_agent_response(raw)
    assert result["parse_success"] is False
    assert result["answer"] == raw
    assert result["citations"] == []
    assert result["escalation"] == {"needed": False, "reason": ""}


def test_plain_text_falls_back():
    raw = "I could not find a relevant policy."
    result = parse_agent_response(raw)
    assert result["parse_success"] is False
    assert result["answer"] == raw


def test_extract_json_none_when_no_brace():
    assert _extract_json("no json here at all") is None
