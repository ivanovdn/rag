from config import Settings


def test_lookback_below_hold_bound_is_silent(capsys):
    """The normal, safe configuration (defaults: 5 < 60) must not warn."""
    Settings(_env_file=None, teams_initial_lookback_minutes=5, teams_max_state_age_minutes=60)
    assert "WARNING" not in capsys.readouterr().out


def test_lookback_equal_to_hold_bound_warns(capsys):
    """Ruling N: at the boundary, the force-advance's strict '>' on held_for
    still guarantees a positive margin, but it can be made arbitrarily small by
    poll-cycle timing -- close enough to "never releases" that it must warn,
    not just the strictly-greater case."""
    Settings(_env_file=None, teams_initial_lookback_minutes=60, teams_max_state_age_minutes=60)
    logged = capsys.readouterr().out
    assert "TEAMS_INITIAL_LOOKBACK_MINUTES" in logged
    assert "TEAMS_MAX_STATE_AGE_MINUTES" in logged


def test_lookback_raised_past_hold_bound_warns(capsys):
    """The controller's own realistic scenario: an operator raises the lookback
    after an incident without also raising the hold bound. Above the bound, the
    force-advance's max(newest_message_time, now - lookback) can fail to clear
    last_check's prior value at all, so the hold never ends."""
    Settings(_env_file=None, teams_initial_lookback_minutes=90, teams_max_state_age_minutes=60)
    logged = capsys.readouterr().out
    assert "TEAMS_INITIAL_LOOKBACK_MINUTES (90)" in logged
    assert "TEAMS_MAX_STATE_AGE_MINUTES (60)" in logged
