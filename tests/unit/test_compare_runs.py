"""The decision rests on per-question movement, not means (design, Measurement 2)."""

from eval.compare_runs import compare, first_match_rank

REF = {"expected_citations": [{"doc_id": "Doc A", "section": "Sec", "clause": "Cl"}]}


def _run(q, ranks_doc, rewrite=None):
    results = [{"doc_title": d, "section": "Sec", "clause": "Cl"} for d in ranks_doc]
    return {"input": {"question": q}, "reference_output": REF,
            "output": {"search_results": results, "rewrite": rewrite or {}}}


def test_rank_is_one_based_and_none_when_absent():
    hit = {"doc_title": "Doc A", "section": "Sec", "clause": "Cl"}
    assert first_match_rank({"search_results": [{"doc_title": "X"}, hit]}, REF) == 2
    assert first_match_rank({"search_results": [{"doc_title": "X"}]}, REF) is None
    assert first_match_rank(None, REF) is None


def test_compare_classifies_each_question():
    base = [_run("recovered", ["X"]), _run("lost", ["Doc A"]), _run("better", ["X", "Doc A"]),
            _run("worse", ["Doc A"]), _run("same", ["Doc A"])]
    cand = [_run("recovered", ["Doc A"], {"latency_ms": 900}), _run("lost", ["X"], {"latency_ms": 1100}),
            _run("better", ["Doc A"]), _run("worse", ["X", "Doc A"]), _run("same", ["Doc A"], {"fallback": True})]

    c = compare(base, cand)

    assert [q for q, *_ in c["recovered"]] == ["recovered"]
    assert [q for q, *_ in c["lost"]] == ["lost"]
    assert [q for q, *_ in c["better"]] == ["better"]
    assert [q for q, *_ in c["worse"]] == ["worse"]
    assert [q for q, *_ in c["same"]] == ["same"]
    assert sorted(c["latency_ms"]) == [900, 1100]
    assert c["fallbacks"] == 1


def test_questions_are_matched_by_text_not_position():
    base = [_run("a", ["Doc A"]), _run("b", ["X"])]
    cand = [_run("b", ["Doc A"]), _run("a", ["Doc A"])]
    c = compare(base, cand)
    assert [q for q, *_ in c["recovered"]] == ["b"]
    assert [q for q, *_ in c["same"]] == ["a"]
