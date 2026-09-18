from datetime import datetime, timezone

import pytest

import channels.teams.bot as bot


@pytest.fixture
def pbot(tmp_path, monkeypatch):
    # process_new_messages() calls _save_state(); keep it off the real state file.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    return bot.TeamsBot(token_refresher=object())


def test_business_hours_use_the_fast_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 5)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 30)
    # Tuesday 11:00 UTC
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 11, 0, tzinfo=timezone.utc)) == 5


def test_nights_use_the_idle_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 5)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 30)
    # Tuesday 03:00 UTC
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)) == 30


def test_weekends_use_the_idle_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 5)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 30)
    # Saturday 11:00 UTC
    assert pbot._current_poll_interval(datetime(2026, 9, 19, 11, 0, tzinfo=timezone.utc)) == 30


def test_messages_are_requested_with_a_page_cap(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 5)
    urls = []
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")


    def fake_api(url, method="GET", json_data=None):
        urls.append(url)
        # Match on the chat-list call without assuming its query string: Task 5
        # appends $expand=lastMessagePreview to this same URL.
        if "/messages" not in url:
            return {"value": [{"id": "c1"}]}
        return {"value": []}

    monkeypatch.setattr(pbot, "_api_request", fake_api)
    pbot.process_new_messages()
    message_urls = [u for u in urls if "/messages" in u]
    # Check the per-chat message URLs only: the chat-list URL carries its own
    # $top=50, and "$top=5" is a substring of "$top=50".
    assert message_urls and all("$top=5" in u for u in message_urls), urls


def test_chat_list_follows_next_link(monkeypatch, pbot):
    """Graph pages /me/chats (20 per page by default). A chat on page 2 must still be polled."""
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 5)
    urls = []
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")

    def fake_api(url, method="GET", json_data=None):
        urls.append(url)
        if "/messages" in url:
            return {"value": []}
        if "skiptoken" in url:                      # page 2
            return {"value": [{"id": "chatB"}]}
        return {                                    # page 1
            "value": [{"id": "chatA"}],
            "@odata.nextLink": f"{bot.GRAPH_API}/me/chats?$top=50&$skiptoken=abc",
        }

    monkeypatch.setattr(pbot, "_api_request", fake_api)
    pbot.process_new_messages()
    assert any("chatB/messages" in u for u in urls), urls
