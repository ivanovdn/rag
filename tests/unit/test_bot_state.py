"""Startup clamp on bot_state.json last_check — prevents a long-stopped bot from
re-answering the whole message backlog into the channel."""
import json
from datetime import datetime, timedelta, timezone

import channels.teams.bot as bot


def _write_state(path, last_check_dt, processed=None):
    path.write_text(json.dumps({
        "last_check": last_check_dt.isoformat(),
        "processed_messages": processed or [],
    }))


def test_stale_last_check_is_clamped(tmp_path, monkeypatch):
    state = tmp_path / "bot_state.json"
    _write_state(state, datetime.now(timezone.utc) - timedelta(days=3))  # very stale
    monkeypatch.setattr(bot, "STATE_FILE", state)
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(bot.settings, "teams_initial_lookback_minutes", 5)

    b = bot.TeamsBot(token_refresher=object())

    age = datetime.now(timezone.utc) - b.last_check
    assert age < timedelta(minutes=10), f"stale last_check was not clamped (age={age})"


def test_recent_last_check_is_preserved(tmp_path, monkeypatch):
    state = tmp_path / "bot_state.json"
    _write_state(state, datetime.now(timezone.utc) - timedelta(minutes=10))  # within max age
    monkeypatch.setattr(bot, "STATE_FILE", state)
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(bot.settings, "teams_initial_lookback_minutes", 5)

    b = bot.TeamsBot(token_refresher=object())

    age = datetime.now(timezone.utc) - b.last_check
    assert timedelta(minutes=9) < age < timedelta(minutes=11), \
        f"recent last_check should resume unchanged (age={age})"


def test_processed_messages_survive_clamp(tmp_path, monkeypatch):
    state = tmp_path / "bot_state.json"
    _write_state(state, datetime.now(timezone.utc) - timedelta(days=3), processed=["m1", "m2"])
    monkeypatch.setattr(bot, "STATE_FILE", state)
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(bot.settings, "teams_initial_lookback_minutes", 5)

    b = bot.TeamsBot(token_refresher=object())

    # clamping last_check must not drop the processed-id set (still dedups what it can)
    assert b.processed_messages == {"m1", "m2"}


def test_missing_state_file_uses_fresh_default(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "does_not_exist.json")
    monkeypatch.setattr(bot.settings, "teams_initial_lookback_minutes", 5)

    b = bot.TeamsBot(token_refresher=object())

    age = datetime.now(timezone.utc) - b.last_check
    assert age < timedelta(minutes=10)
    assert b.processed_messages == set()
