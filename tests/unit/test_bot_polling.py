from datetime import datetime, timedelta, timezone

import pytest

import channels.teams.bot as bot


@pytest.fixture
def pbot(tmp_path, monkeypatch):
    # process_new_messages() calls _save_state(); keep it off the real state file.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    return bot.TeamsBot(token_refresher=object())


# --- _current_poll_interval: business hours / idle, weekday/weekend -------------
#
# Every setting below is deliberately set to a value that differs from its
# config.py default, and the two business-hours bounds are always set explicitly
# (not left at their defaults) — otherwise a hardcoded
# `5 if (weekday and 7 <= hour < 19) else 30` would pass every one of these.

def test_business_hours_use_the_fast_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 9)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 18)
    # Tuesday 11:00 UTC
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 11, 0, tzinfo=timezone.utc)) == 3


def test_nights_use_the_idle_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 9)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 18)
    # Tuesday 03:00 UTC
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)) == 17


def test_weekends_use_the_idle_interval(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 9)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 18)
    # Saturday 11:00 UTC — inside the hour window, but not a weekday
    assert pbot._current_poll_interval(datetime(2026, 9, 19, 11, 0, tzinfo=timezone.utc)) == 17


def test_start_hour_is_inclusive(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 9)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 18)
    # Tuesday 09:00 UTC — exactly the start hour must already be fast
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)) == 3


def test_end_hour_is_exclusive(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 9)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 18)
    # Tuesday 18:00 UTC — the end hour itself must already be idle
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 18, 0, tzinfo=timezone.utc)) == 17


# --- _current_poll_interval: a window that wraps past midnight (Finding 5) ------
#
# start > end (e.g. 22-6) is a window that wraps past midnight, not an empty one.
# Comparing it as a plain start <= hour < end range would silently degrade to
# "always idle" for any wrapping configuration — a real footgun for a timezone
# whose business hours cross midnight in UTC (e.g. US Pacific).

def test_wrapped_window_is_fast_late_at_night(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 22)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 6)
    # Tuesday 23:00 UTC — after start, before midnight
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 23, 0, tzinfo=timezone.utc)) == 3


def test_wrapped_window_is_fast_just_after_midnight(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 22)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 6)
    # Tuesday 02:00 UTC — after midnight, before end
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 2, 0, tzinfo=timezone.utc)) == 3


def test_wrapped_window_is_idle_during_the_day(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 22)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 6)
    # Tuesday 12:00 UTC — outside the wrapped window on both sides
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)) == 17


# --- Per-chat message fetch: $top cap ------------------------------------------

def test_messages_are_requested_with_a_page_cap(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 7)  # not the default (5)
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
    # $top=50, and "$top=7" only ever appears on the per-chat URL.
    assert message_urls and all("$top=7" in u for u in message_urls), urls


# --- Chat list: @odata.nextLink pagination (Finding 4: cap + cycle guard) ------

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
    # Both pages must be polled: an implementation that overwrote the accumulator
    # instead of extending it would still find chatB but silently lose chatA.
    assert any("chatA/messages" in u for u in urls), urls
    assert any("chatB/messages" in u for u in urls), urls


def test_cyclic_next_link_terminates(monkeypatch, pbot):
    """A self-referential @odata.nextLink must not spin the poll thread forever
    (Finding 4): the loop never raises, so consecutive_errors would never trip
    and the bot would go silently dead while still 'running'."""
    calls = {"n": 0}
    start_url = f"{bot.GRAPH_API}/me/chats?$top=50"

    def fake_api(url, method="GET", json_data=None, retry=False):
        calls["n"] += 1
        return {"value": [{"id": "x"}], "@odata.nextLink": start_url}  # always the same link

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    items, complete = pbot._get_all_pages(start_url)

    assert complete is False
    assert calls["n"] <= bot._CHATS_MAX_PAGES


def test_unbounded_next_link_hits_the_page_cap(monkeypatch, pbot):
    """Distinct from the cyclic case above: an endless run of NOVEL links (no
    repeat for the cycle guard to catch) must still terminate via the numeric cap."""
    calls = {"n": 0}

    def fake_api(url, method="GET", json_data=None, retry=False):
        calls["n"] += 1
        return {"value": [{"id": f"x{calls['n']}"}], "@odata.nextLink": f"{url}&page={calls['n']}"}

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    items, complete = pbot._get_all_pages(f"{bot.GRAPH_API}/me/chats?$top=50")

    assert complete is False
    assert calls["n"] == bot._CHATS_MAX_PAGES


# --- Finding 1: an incomplete chat list must not advance last_check -----------

def test_truncated_chat_list_holds_last_check(monkeypatch, pbot):
    """A chat-list page fetch that fails partway through must not let last_check
    advance past chats it never saw. Otherwise those chats' messages are marked
    already-old — and lost for good — the moment the watermark passes them,
    while a failed call used to leave last_check (and everyone) untouched."""
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message",
                        lambda chat_id, text, content_type="html", retry=False: True)
    original_last_check = pbot.last_check

    future = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" in url:
            return {"value": [{
                "id": "mA", "messageType": "message",
                "from": {"user": {"id": "someone", "displayName": "Ann"}},
                "createdDateTime": future, "body": {"content": "chatA question"},
            }]}
        if "skiptoken" in url:
            return None  # page 2 of the chat list fails
        return {
            "value": [{"id": "chatA"}],
            "@odata.nextLink": f"{bot.GRAPH_API}/me/chats?$top=50&$skiptoken=xyz",
        }

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    # chatA — the page we did get — was still processed...
    assert pbot._work_q.qsize() == 1
    # ...but the watermark must not move: chatB (behind the failed page) was
    # never seen this cycle, so it must be retried, not marked already-seen.
    assert pbot.last_check == original_last_check


# --- Findings 2/3: a burst bigger than one page must not lose messages --------

def test_message_burst_larger_than_the_page_is_not_dropped(monkeypatch, pbot):
    """$top caps a single page at teams_messages_page_size; a burst bigger than
    that must page backward for the rest, not silently drop the oldest messages
    in it. This is also what keeps a restarted bot able to re-fetch an in-flight
    message once enough newer messages have landed on top of it (Finding 3) —
    same fix, both findings."""
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 5)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message",
                        lambda chat_id, text, content_type="html", retry=False: True)

    base = datetime.now(timezone.utc) + timedelta(minutes=1)  # comfortably after last_check

    def _msg(n):
        stamp = (base + timedelta(seconds=n)).isoformat().replace("+00:00", "Z")
        return {"id": f"m{n}", "messageType": "message",
                "from": {"user": {"id": "someone", "displayName": "Ann"}},
                "createdDateTime": stamp, "body": {"content": f"question {n}"}}

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" not in url:
            return {"value": [{"id": "chat1"}]}
        if "page2" in url:
            return {"value": [_msg(1)]}  # the 6th message, one page further back
        return {
            "value": [_msg(6), _msg(5), _msg(4), _msg(3), _msg(2)],  # newest-first
            "@odata.nextLink": f"{bot.GRAPH_API}/me/chats/chat1/messages?page2",
        }

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    queued = [pbot._work_q.get_nowait()[1] for _ in range(pbot._work_q.qsize())]
    assert queued == [f"question {n}" for n in range(1, 7)]


def test_message_paging_cap_marks_the_chat_incomplete(monkeypatch, pbot):
    """Symmetric to the chat-list page cap: a burst so large it never pages back
    to last_check must report itself incomplete rather than claim a full picture
    it does not have (see process_new_messages: an incomplete chat must not let
    the cycle advance last_check)."""
    calls = {"n": 0}

    def fake_api(url, method="GET", json_data=None, retry=False):
        calls["n"] += 1
        stamp = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
        msg = {"id": f"m{calls['n']}", "messageType": "message",
               "from": {"user": {"id": "someone"}}, "createdDateTime": stamp,
               "body": {"content": "x"}}
        return {"value": [msg], "@odata.nextLink": f"{url}&page={calls['n']}"}

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    messages, complete = pbot._get_chat_messages("chat1")

    assert complete is False
    assert calls["n"] == bot._MESSAGES_MAX_PAGES
