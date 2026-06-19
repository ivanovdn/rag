"""Locate the local policy corpus. The .docx files are gitignored, so this
returns an empty list on machines/CI without them and callers skip."""
from pathlib import Path

from config import settings


def policy_docx_files() -> list[Path]:
    folder = Path(settings.policy_docs_folder)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.docx"))
