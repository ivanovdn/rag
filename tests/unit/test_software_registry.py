import pytest

from rag.software_registry import (
    SoftwareRow,
    normalize_name,
    build_name_index,
    lookup_name,
    software_embed_text,
)


@pytest.fixture
def rows():
    return [
        SoftwareRow(name="Docker", status="allowed",
                    note="A license is required for commercial usage",
                    source_list="Allowed Software"),
        SoftwareRow(name="WinRar", status="forbidden",
                    category="High-risk russian software",
                    alternative="Windows 11 natively supports .tar, .tgz, .7z, .rar etc.",
                    source_list="Forbidden Software"),
        SoftwareRow(name="JetBrains Software", status="forbidden",
                    category="High-risk russian software",
                    alternative="MS Visual Studio, VS Code",
                    aliases=["IntelliJ IDEA", "PyCharm", "WebStorm", "Rider"],
                    source_list="Forbidden Software"),
    ]


def test_normalize_collapses_whitespace_and_case():
    assert normalize_name("  Docker   Desktop ") == "docker desktop"


def test_exact_name_match(rows):
    idx = build_name_index(rows)
    r = lookup_name("docker", idx, threshold=85.0)
    assert r.row is not None and r.row.name == "Docker" and r.exact is True


def test_alias_resolves_to_parent_row(rows):
    idx = build_name_index(rows)
    r = lookup_name("PyCharm", idx, threshold=85.0)
    assert r.row is not None and r.row.name == "JetBrains Software"


def test_fuzzy_typo_matches_above_threshold(rows):
    idx = build_name_index(rows)
    # genuine typo (extra 'r'), NOT just a case variant — a case variant would
    # normalize to an exact key hit; this must exercise the fuzzy path (exact=False).
    r = lookup_name("WinRarr", idx, threshold=85.0)
    assert r.row is not None and r.row.name == "WinRar" and r.exact is False


def test_unknown_returns_no_row_but_suggests_closest(rows):
    idx = build_name_index(rows)
    r = lookup_name("Dockerr Composer XYZ", idx, threshold=99.0)
    assert r.row is None
    assert r.suggestion is not None  # closest candidate offered as "did you mean"


def test_embed_text_forbidden_includes_category_and_alternative(rows):
    txt = software_embed_text(rows[1])
    assert "WinRar" in txt and "russian" in txt and "Alternative:" in txt


def test_embed_text_allowed_is_name_and_note(rows):
    txt = software_embed_text(rows[0])
    assert txt.startswith("Docker") and "license" in txt
