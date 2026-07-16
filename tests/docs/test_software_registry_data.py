import pytest

from config import settings
from pathlib import Path
from rag.software_registry import load_registry

_PRESENT = Path(settings.software_registry_path).is_file()
pytestmark = pytest.mark.skipif(not _PRESENT, reason="software registry JSON not present")


def test_registry_loads_and_has_both_lists():
    rows = load_registry()
    assert len(rows) >= 50
    statuses = {r.status for r in rows}
    assert statuses == {"allowed", "forbidden"}
    for r in rows:
        assert r.name.strip()
        assert r.status in ("allowed", "forbidden")
        assert r.source_list in ("Allowed Software", "Forbidden Software")


def _find(rows, name):
    return next((r for r in rows if r.name.lower() == name.lower()), None)


def test_known_rows_are_correct():
    rows = load_registry()
    docker = _find(rows, "Docker")
    assert docker and docker.status == "allowed" and "license" in docker.note.lower()

    tv = _find(rows, "TeamViewer")
    assert tv and tv.status == "forbidden"

    jb = _find(rows, "JetBrains Software")
    assert jb and "vs code" in jb.alternative.lower()
    assert any("pycharm" in a.lower() for a in jb.aliases)

    vpn = next((r for r in rows if r.status == "forbidden" and "vpn" in r.name.lower()), None)
    assert vpn and "cisco anyconnect" in vpn.alternative.lower()
