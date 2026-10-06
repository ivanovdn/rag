"""`.env.example` may differ from the code defaults — but not about a floor.

CLAUDE.md is explicit that `config.py` defaults are lighter dev fallbacks and the
deployed `.env` is the source of truth, so the template legitimately carries
production values a default does not (RERANKER_CANDIDATES=25 against a default of
20, for one). Measured 2026-10-06: 34 of 37 example keys equal their default and
the three that differ are deliberate.

A relevance floor is the exception, because the difference is not tuning but
on/off. `.env.example` shipped `RERANKER_MIN_SCORE=0.0` — documented in the same
line as "0.0 = off" — while `config.py` defaulted to 0.2. A deployment that
copied the template ran with no floor; one that omitted the line ran with one.
Same repo, two safety behaviours, decided by whether someone copied a file.

Production had never noticed, because the floor has never fired: 0
retrieval_floor_rejected spans across 45 reranked questions, 2026-09-22 to
2026-10-06, with p10 at 0.9772. The only sub-threshold scores on record are three
askings of "Can I bring penguin into office" on 2026-09-24, before the floor
existed — all of which escalated correctly on model judgement anyway. That is the
case the floor is for, not a precision knob.
"""

from pathlib import Path

import pytest

from config import Settings

FLOORS = ("reranker_min_score", "min_confidence_score")


def _example_values() -> dict[str, str]:
    out = {}
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().lower()] = v.split("#")[0].strip()
    return out


@pytest.mark.parametrize("field", FLOORS)
def test_the_example_env_matches_the_code_default_for_every_relevance_floor(field):
    example = _example_values()
    assert field in example, f".env.example no longer documents {field.upper()}"
    assert float(example[field]) == pytest.approx(Settings.model_fields[field].default), (
        f"{field.upper()} differs between .env.example and config.py. For a floor "
        "that is not a tuning difference: whichever a deployment happens to use "
        "decides whether the guard runs at all."
    )


@pytest.mark.parametrize("field", FLOORS)
def test_no_relevance_floor_is_shipped_disabled(field):
    """0.0 means off for both of these. A template that ships a guard off is how
    a deployment ends up without one while the repo says it has one."""
    assert float(_example_values()[field]) > 0, f"{field.upper()} ships disabled"
