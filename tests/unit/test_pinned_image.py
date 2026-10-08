"""The bot image installs exactly the versions proven in production.

requirements-bot.txt is all >= ranges, so the image was reproducible only by
accident of Docker's layer cache: a cache prune or a new host would have moved
every dependency to its newest release at once. That is how qdrant-client
1.19.1 reached production against a 1.17.1 server without anyone choosing it.
requirements-bot.lock is the 2026-10-08 `pip freeze` of the running container.
"""

import re
from pathlib import Path

DOCKERFILE = Path("Dockerfile").read_text(encoding="utf-8")


def _names(path):
    names = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(re.split(r"[<>=!~\[ ]", line, 1)[0].lower().replace("_", "-"))
    return names


def test_the_image_installs_under_the_constraints():
    assert re.search(r"pip install .*-r requirements-bot\.txt .*-c requirements-bot\.lock", DOCKERFILE)
    copy = re.search(r"^COPY .*requirements-bot\.lock.*$", DOCKERFILE, re.M)
    assert copy and copy.start() < DOCKERFILE.index("RUN pip install")


def test_every_direct_dependency_is_pinned():
    """A requirement missing from the lock would float again, silently."""
    missing = _names("requirements-bot.txt") - _names("requirements-bot.lock")
    assert missing == set()


def test_every_lock_line_is_an_exact_pin():
    lines = [
        l.strip() for l in Path("requirements-bot.lock").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    loose = [l for l in lines if not re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+\-]+", l)]
    assert lines and loose == []


def test_the_base_image_is_the_python_production_runs():
    assert "FROM python:3.12.15-slim" in DOCKERFILE
