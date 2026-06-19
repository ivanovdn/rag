from types import SimpleNamespace

from ingest.docx_parser import (
    extract_clause_number,
    extract_label,
    extract_heading_level,
    extract_clause_name,
    _estimate_tokens,
    _split_oversized,
    _table_to_text,
)


# --- pure string helpers ---

def test_extract_clause_number_variants():
    assert extract_clause_number("4.7 Software Installation") == "4.7"
    assert extract_clause_number("3. Introduction") == "3"
    assert extract_clause_number("4.2.1 Nested item") == "4.2.1"


def test_extract_clause_number_none_without_trailing_space():
    assert extract_clause_number("Introduction") is None
    assert extract_clause_number("4.7No space after number") is None


def test_extract_label():
    assert extract_label("Acceptable Use: limited and occasional") == "Acceptable Use"
    assert extract_label("lowercase start: nope") is None


def test_estimate_tokens_four_chars_per_token():
    assert _estimate_tokens("a" * 40) == 10


def test_split_oversized_under_limit_returns_single():
    text = "Short sentence."
    assert _split_oversized(text, max_tokens=100) == [text]


def test_split_oversized_splits_on_sentence_boundaries():
    text = "This is a sentence with several words. " * 20
    parts = _split_oversized(text, max_tokens=20)
    assert len(parts) > 1
    assert all(p.strip() for p in parts)


# --- helpers that read docx-like objects (lightweight fakes) ---

def _fake_run(text, bold):
    return SimpleNamespace(text=text, bold=bold)


def _fake_para(runs=(), style_name="Normal", text=""):
    return SimpleNamespace(runs=list(runs), style=SimpleNamespace(name=style_name), text=text)


def test_extract_heading_level():
    assert extract_heading_level(_fake_para(style_name="Heading 2")) == 2
    assert extract_heading_level(_fake_para(style_name="Normal")) is None


def test_extract_clause_name_from_leading_bold_runs():
    para = _fake_para(runs=[
        _fake_run("Blogging and Social Media:", True),
        _fake_run(" the rest of the text", False),
    ])
    assert extract_clause_name(para) == "Blogging and Social Media"


def test_extract_clause_name_empty_when_first_run_not_bold():
    para = _fake_para(runs=[_fake_run("not bold", False)])
    assert extract_clause_name(para) == ""


def test_table_to_text_pairs_headers_with_cells():
    def cell(t):
        return SimpleNamespace(text=t)

    def row(*texts):
        return SimpleNamespace(cells=[cell(t) for t in texts])

    table = SimpleNamespace(rows=[row("H1", "H2"), row("a", "b")])
    assert _table_to_text(table) == "H1: a | H2: b"
