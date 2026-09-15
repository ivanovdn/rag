"""Selective ingest: explicit .docx paths instead of the whole --folder.

Validation resolves every path up front so a typo'd second filename can't leave
the collection half re-ingested. doc_link is built from the BASENAME only, so a
file ingested from anywhere gets the same link (and doc_id) a full-folder run
would give it — that is what makes a partial re-ingest overwrite cleanly.
"""
from pathlib import Path

import pytest

import ingest.pipeline as pipeline


# --- resolve_docx_paths: pure validation, no backends ---


def test_resolves_existing_docx_files(tmp_path):
    a = tmp_path / "Backup Policy [Internal].docx"
    b = tmp_path / "Data Retention Policy [Internal].docx"
    a.touch()
    b.touch()

    paths, errors = pipeline.resolve_docx_paths([str(a), str(b)])

    assert errors == []
    assert paths == [a, b]


def test_missing_file_is_an_error(tmp_path):
    paths, errors = pipeline.resolve_docx_paths([str(tmp_path / "Nope.docx")])

    assert paths == []
    assert len(errors) == 1
    assert "Nope.docx" in errors[0]


def test_non_docx_extension_is_an_error(tmp_path):
    f = tmp_path / "notes.txt"
    f.touch()

    paths, errors = pipeline.resolve_docx_paths([str(f)])

    assert paths == []
    assert ".docx" in errors[0]


def test_word_lock_file_is_an_error(tmp_path):
    lock = tmp_path / "~$Backup Policy [Internal].docx"
    lock.touch()

    paths, errors = pipeline.resolve_docx_paths([str(lock)])

    assert paths == []
    assert len(errors) == 1


def test_directory_named_like_a_docx_is_an_error(tmp_path):
    d = tmp_path / "bundle.docx"
    d.mkdir()

    paths, errors = pipeline.resolve_docx_paths([str(d)])

    assert paths == []
    assert len(errors) == 1


def test_every_bad_path_is_reported_not_just_the_first(tmp_path):
    """One aborted run listing all problems beats failing twice."""
    good = tmp_path / "Good Policy [Internal].docx"
    good.touch()
    other = tmp_path / "other.txt"
    other.touch()

    paths, errors = pipeline.resolve_docx_paths(
        [str(good), str(tmp_path / "Missing.docx"), str(other)]
    )

    assert len(errors) == 2
    assert paths == [good]


# --- ingest_files: the selective path ---


@pytest.fixture
def ingest_calls(monkeypatch):
    """Record (filepath, doc_link) per ingest_document call; no Qdrant, no embeddings."""
    calls = []
    monkeypatch.setattr(pipeline, "init_collection", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "ingest_document",
        lambda filepath, doc_link: calls.append((filepath, doc_link)) or 3,
    )
    return calls


def test_doc_link_uses_basename_even_from_another_folder(tmp_path, ingest_calls):
    """A file ingested from outside ./policies must still get the canonical link,
    so its doc_id and doc_link match what a full-folder run would produce."""
    elsewhere = tmp_path / "downloads"
    elsewhere.mkdir()
    f = elsewhere / "Backup Policy [Internal].docx"
    f.touch()

    results = pipeline.ingest_files([f], "http://intranet/policies")

    assert ingest_calls == [(f, "http://intranet/policies/Backup Policy [Internal].docx")]
    assert results == {"Backup Policy [Internal].docx": 3}


def test_ingest_files_reports_a_count_per_file(tmp_path, ingest_calls):
    a = tmp_path / "A Policy [Internal].docx"
    b = tmp_path / "B Policy [Internal].docx"
    a.touch()
    b.touch()

    results = pipeline.ingest_files([a, b], "http://intranet/policies")

    assert results == {"A Policy [Internal].docx": 3, "B Policy [Internal].docx": 3}
    assert len(ingest_calls) == 2


def test_ingest_files_with_no_paths_does_nothing(ingest_calls):
    assert pipeline.ingest_files([], "http://intranet/policies") == {}
    assert ingest_calls == []


# --- ingest_folder: unchanged behavior after delegating to ingest_files ---


def test_ingest_folder_still_skips_word_lock_files(tmp_path, ingest_calls):
    (tmp_path / "Real Policy [Internal].docx").touch()
    (tmp_path / "~$Real Policy [Internal].docx").touch()

    results = pipeline.ingest_folder(tmp_path, "http://intranet/policies")

    assert list(results) == ["Real Policy [Internal].docx"]


def test_ingest_folder_on_empty_folder_returns_empty(tmp_path, ingest_calls):
    assert pipeline.ingest_folder(tmp_path, "http://intranet/policies") == {}
    assert ingest_calls == []
