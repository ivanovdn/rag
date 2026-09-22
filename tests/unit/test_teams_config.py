from config import Settings


def test_lookback_below_hold_bound_is_silent(capsys):
    """The normal, safe configuration (defaults: 5 < 60) must not warn."""
    Settings(_env_file=None, teams_initial_lookback_minutes=5, teams_max_state_age_minutes=60)
    assert "WARNING" not in capsys.readouterr().out


def test_lookback_equal_to_hold_bound_warns(capsys):
    """At the boundary the startup clamp re-opens a window exactly as wide as
    the staleness that triggered it, so the backlog it just rejected as too old
    comes straight back as new. Equal is already incoherent, so the warning must
    cover it, not just the strictly-greater case."""
    Settings(_env_file=None, teams_initial_lookback_minutes=60, teams_max_state_age_minutes=60)
    logged = capsys.readouterr().out
    assert "TEAMS_INITIAL_LOOKBACK_MINUTES" in logged
    assert "TEAMS_MAX_STATE_AGE_MINUTES" in logged


def test_lookback_raised_past_hold_bound_warns(capsys):
    """The realistic scenario: an operator raises the lookback after an incident
    without also raising the hold bound. _load_state then clamps a stale
    watermark further back than the staleness threshold that triggered the
    clamp, so a long-stopped bot answers the very backlog the clamp exists to
    suppress. (This used to be justified by the runtime force-advance, which
    targeted now - lookback; that target is now plain `now`, so the startup
    clamp is the only reason left to warn.)"""
    Settings(_env_file=None, teams_initial_lookback_minutes=90, teams_max_state_age_minutes=60)
    logged = capsys.readouterr().out
    assert "TEAMS_INITIAL_LOOKBACK_MINUTES (90)" in logged
    assert "TEAMS_MAX_STATE_AGE_MINUTES (60)" in logged
