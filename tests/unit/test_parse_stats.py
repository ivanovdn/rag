from scripts.parse_coverage import compute_parse_stats


def test_compute_parse_stats_counts(make_chunk):
    chunks = [
        make_chunk(text="alpha", clause_number="1.1", section="S"),
        make_chunk(text="beta", clause_number="", section="", clause=""),
        make_chunk(text="   ", clause_number="2.0", clause="C"),  # blank text
    ]
    s = compute_parse_stats(chunks)
    assert s["chunk_count"] == 3
    assert s["with_clause_number"] == 2
    assert s["with_section_or_clause"] == 2
    assert s["empty_text"] == 1
    assert s["pct_clause_number"] == round(2 / 3, 3)


def test_compute_parse_stats_empty_list():
    s = compute_parse_stats([])
    assert s["chunk_count"] == 0
    assert s["pct_clause_number"] == 0.0
