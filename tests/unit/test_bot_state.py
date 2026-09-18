"""Startup clamp on bot_state.json last_check — prevents a long-stopped bot from
re-answering the whole message backlog into the channel."""
import json
import pathlib
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
    assert set(b.processed_messages) == {"m1", "m2"}


def test_cleanup_evicts_the_oldest_ids_not_arbitrary_ones(tmp_path, monkeypatch):
    """Eviction must be oldest-first.

    A set iterates by hash, so the old slice evicted near-randomly — which matters
    since the rewound watermark relies on processed_messages to suppress re-answering
    messages already answered inside the rewind window.
    """
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    monkeypatch.setattr(bot.settings, "teams_max_processed_messages", 1000)
    b = bot.TeamsBot(token_refresher=object())

    for i in range(2000):
        # Mark through a real production path: a system message is recorded and skipped.
        b._should_process_message({"id": f"m{i:05d}", "messageType": "systemEventMessage"}, "me", "chat1")
    assert len(b.processed_messages) == 2000

    b._cleanup_processed_messages()

    kept = set(b.processed_messages)
    assert all(f"chat1:m{i:05d}" in kept for i in range(1600, 2000)), "the newest ids must be retained"
    assert not any(f"chat1:m{i:05d}" in kept for i in range(400)), "the oldest ids must be the evicted ones"


def test_missing_state_file_uses_fresh_default(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "does_not_exist.json")
    monkeypatch.setattr(bot.settings, "teams_initial_lookback_minutes", 5)

    b = bot.TeamsBot(token_refresher=object())

    age = datetime.now(timezone.utc) - b.last_check
    assert age < timedelta(minutes=10)
    assert not b.processed_messages


def test_state_file_is_written_atomically(tmp_path, monkeypatch):
    """A SIGKILL mid-write must not be able to leave truncated JSON: this file is the
    only carrier of the crash-recovery guarantee."""
    state = tmp_path / "bot_state.json"
    monkeypatch.setattr(bot, "STATE_FILE", state)
    b = bot.TeamsBot(token_refresher=object())
    for i in range(50):
        b._mark_processed(f"chat1:m{i}")

    replaced = {}
    real_replace = bot.os.replace

    def _spy(src, dst):
        # At the moment of the rename the target must still hold the previous, valid
        # content — never a partially written file.
        replaced["src"] = str(src)
        replaced["dst"] = str(dst)
        return real_replace(src, dst)

    monkeypatch.setattr(bot.os, "replace", _spy)
    b._save_state()

    assert replaced["dst"] == str(state), "the final write must be a rename onto the target"
    assert replaced["src"] != str(state), "content must be staged in a separate file"
    assert pathlib.Path(replaced["src"]).parent == state.parent, \
        "the temp file must share a directory with the target, or the rename is not atomic"
    assert json.loads(state.read_text())["processed_messages"][0] == "chat1:m0"


def test_processed_message_order_survives_the_state_file(tmp_path, monkeypatch):
    """Insertion order is the whole point of keying processed_messages by dict —
    it has to survive the save/load round trip, not just live in memory."""
    state = tmp_path / "bot_state.json"
    monkeypatch.setattr(bot, "STATE_FILE", state)
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    b = bot.TeamsBot(token_refresher=object())
    b.last_check = datetime.now(timezone.utc) - timedelta(minutes=1)

    keys = [f"chat1:m{i:03d}" for i in range(50)]
    for key in keys:
        b._mark_processed(key)
    b._save_state()

    reloaded = bot.TeamsBot(token_refresher=object())
    assert list(reloaded.processed_messages) == keys


def test_legacy_bare_id_state_is_suppressed_by_the_watermark(tmp_path, monkeypatch):
    """Upgrade path: state files written before composite keying hold bare message ids,
    which no longer match. last_check is what stops that becoming a re-answer storm."""
    state = tmp_path / "bot_state.json"
    last_check = datetime.now(timezone.utc) - timedelta(minutes=10)
    _write_state(state, last_check, processed=["m1", "m2"])  # legacy bare ids
    monkeypatch.setattr(bot, "STATE_FILE", state)
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    b = bot.TeamsBot(token_refresher=object())

    # The legacy entry no longer matches the composite key...
    assert bot._message_key("chat1", "m1") not in b.processed_messages
    # ...but the message predates last_check, so it is still skipped.
    older = (last_check - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    should, reason = b._should_process_message(
        {"id": "m1", "messageType": "message",
         "from": {"user": {"id": "someone"}}, "createdDateTime": older},
        "me", "chat1",
    )
    assert should is False and reason == "old_message"


def test_message_key_separates_identical_ids_in_different_chats(tmp_path, monkeypatch):
    """Graph ids are unique per chat, not globally — they are millisecond epochs."""
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    b = bot.TeamsBot(token_refresher=object())
    recent = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    message = {"id": "1757000000000", "messageType": "message",
               "from": {"user": {"id": "someone"}}, "createdDateTime": recent}

    assert b._should_process_message(message, "me", "chatA") == (True, None)
    b._mark_processed(bot._message_key("chatA", message["id"]))
    # Same id, different chat: must NOT be mistaken for already-processed.
    assert b._should_process_message(message, "me", "chatB") == (True, None)
