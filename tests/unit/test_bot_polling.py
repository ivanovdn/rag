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
    # Tuesday 23:00 UTC — after start, before midnight: fast under the wraparound
    # fix, idle under the pre-fix "start <= hour < end" formula (that formula is
    # unsatisfiable whenever start > end, so it is *always* idle). This assertion
    # is the one that discriminates the fix.
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 23, 0, tzinfo=timezone.utc)) == 3
    # Tuesday 12:00 UTC — outside the window either way. On its own this cannot
    # discriminate the fix (the pre-fix always-idle bug agrees with the correct
    # answer here) — R-6 — so it rides along with the 23:00 assertion above as a
    # same-config sanity check, not as independent proof.
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)) == 17


def test_wrapped_window_is_fast_just_after_midnight(monkeypatch, pbot):
    monkeypatch.setattr(bot.settings, "teams_poll_interval", 3)
    monkeypatch.setattr(bot.settings, "teams_idle_poll_interval", 17)
    monkeypatch.setattr(bot.settings, "teams_business_hours_start_utc", 22)
    monkeypatch.setattr(bot.settings, "teams_business_hours_end_utc", 6)
    # Tuesday 02:00 UTC — after midnight, before end: also discriminates (idle
    # under the pre-fix always-idle formula, fast under the wraparound fix).
    assert pbot._current_poll_interval(datetime(2026, 9, 15, 2, 0, tzinfo=timezone.utc)) == 3


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


def test_cyclic_next_link_terminates(monkeypatch, pbot, capsys):
    """A self-referential @odata.nextLink must not spin the poll thread forever
    (Finding 4): the loop never raises, so consecutive_errors would never trip
    and the bot would go silently dead while still 'running'.

    R-5: pytest-timeout is not installed, so if the guards ever regressed this
    test would hang the whole suite instead of failing it. The fake bounds itself
    with a call ceiling well above any correct call count, so a regression fails
    fast with a clear assertion instead of wedging CI.
    R-4/Ruling J: asserts the exact call count (1), not just "<= cap" — the numeric
    page cap alone would already satisfy a "<=" bound, so that weaker assertion
    passes even with the seen_urls revisit guard deleted. Only "== 1" proves the
    guard caught the repeat on the very next call, before the numeric cap would have.
    """
    calls = {"n": 0}
    start_url = f"{bot.GRAPH_API}/me/chats?$top=50"
    call_ceiling = bot._CHATS_MAX_PAGES + 5  # generous headroom above any correct count

    def fake_api(url, method="GET", json_data=None, retry=False):
        calls["n"] += 1
        if calls["n"] > call_ceiling:
            raise AssertionError(
                f"_get_all_pages made more than {call_ceiling} calls; a pagination "
                "guard regressed — failing fast instead of hanging the suite"
            )
        return {"value": [{"id": "x"}], "@odata.nextLink": start_url}  # always the same link

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    items, complete = pbot._get_all_pages(start_url)

    assert complete is False
    assert calls["n"] == 1  # the seen_urls guard must catch the repeat on the very next call
    # Fix 5: this WARNING is what tells an operator the chat list was cut short
    # by a repeated link, distinctly from the plain page-cap case below.
    logged = capsys.readouterr().out
    assert "repeated" in logged
    assert "incomplete" in logged


def test_unbounded_next_link_hits_the_page_cap(monkeypatch, pbot, capsys):
    """Distinct from the cyclic case above: an endless run of NOVEL links (no
    repeat for the cycle guard to catch) must still terminate via the numeric cap.

    R-5: same self-bounding-fake technique as the cyclic test above, so a
    regressed cap fails fast instead of hanging the suite (no pytest-timeout).
    """
    calls = {"n": 0}
    call_ceiling = bot._CHATS_MAX_PAGES + 5

    def fake_api(url, method="GET", json_data=None, retry=False):
        calls["n"] += 1
        if calls["n"] > call_ceiling:
            raise AssertionError(
                f"_get_all_pages made more than {call_ceiling} calls; the page cap "
                "regressed — failing fast instead of hanging the suite"
            )
        return {"value": [{"id": f"x{calls['n']}"}], "@odata.nextLink": f"{url}&page={calls['n']}"}

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    items, complete = pbot._get_all_pages(f"{bot.GRAPH_API}/me/chats?$top=50")

    assert complete is False
    assert calls["n"] == bot._CHATS_MAX_PAGES
    # Fix 5: this WARNING is what tells an operator the chat list was cut short
    # by the numeric cap, distinctly from the cyclic-link case above.
    logged = capsys.readouterr().out
    assert "page cap" in logged
    assert "incomplete" in logged


# --- Finding 1: an incomplete chat list must not advance last_check -----------

def test_truncated_chat_list_holds_last_check(monkeypatch, pbot, capsys):
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
    # Fix 5: this WARNING is the operator's only signal that the chat list was
    # cut short and last_check is being held because of it.
    logged = capsys.readouterr().out
    assert "chat list" in logged
    assert "incomplete" in logged


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


def test_message_paging_cap_marks_the_chat_incomplete(monkeypatch, pbot, capsys):
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
    # Fix 5: this WARNING is the operator's only signal that this chat's paging
    # gave up without ever reaching last_check.
    logged = capsys.readouterr().out
    assert "chat1" in logged
    assert "page" in logged and "cap" in logged


def test_message_paging_cap_holds_last_check_via_process_new_messages(monkeypatch, pbot):
    """R-4 (second bullet) / Ruling J: the accepted Ruling-B extension — a
    cap-exhausted chat must feed process_new_messages' fully_synced, not just
    _get_chat_messages' own return value — has to be proven end-to-end. Mutation-
    tested: deleting the `fully_synced = False` wiring at the chat_complete check
    leaves test_message_paging_cap_marks_the_chat_incomplete (above) green, because
    that test only calls _get_chat_messages directly. This one fails instead,
    because it drives process_new_messages and checks the one thing the wiring is
    actually for: last_check must not move."""
    # Comfortably above the ~0s this single cycle could ever hold for, so Ruling
    # G's force-advance can't mask the assertion regardless of the real .env value.
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    original_last_check = pbot.last_check
    calls = {"n": 0}

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" not in url:
            return {"value": [{"id": "chat1"}]}
        calls["n"] += 1
        stamp = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
        msg = {"id": f"m{calls['n']}", "messageType": "message",
               "from": {"user": {"id": "someone"}}, "createdDateTime": stamp,
               "body": {"content": "x"}}
        return {"value": [msg], "@odata.nextLink": f"{url}&page={calls['n']}"}

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    assert pbot.last_check == original_last_check


def test_message_fetch_failure_holds_last_check_via_process_new_messages(monkeypatch, pbot, capsys):
    """Branch review Fix 3: of the three incompleteness paths (chat-list failure,
    per-chat page-cap exhaustion, per-chat fetch failure), the last is uncovered
    end to end and the likeliest to fire in production -- any single
    4xx/5xx/timeout on one chat. Mutation-tested: flipping _get_chat_messages'
    `return messages, False` to `True` on outright fetch failure passed the
    entire 272-test suite before this test existed.

    A second, HEALTHY chat with a genuinely new message is essential here, not
    decoration: with only the stuck chat in play, newest_message_time never
    moves past last_check either way, so `last_check == original_last_check`
    would pass whether or not the fetch-failure is honoured -- that shape of
    test is exactly what let the mutation above survive on a first attempt.
    The healthy chat gives fully_synced something to wrongly act on if the
    stuck chat's incompleteness is ever ignored."""
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    original_last_check = pbot.last_check
    future = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" not in url:
            return {"value": [{"id": "stuck"}, {"id": "healthy"}]}
        if "healthy" in url:
            return {"value": [{
                "id": "hm1", "messageType": "message",
                "from": {"user": {"id": "someone"}},
                "createdDateTime": future, "body": {"content": "healthy chat question"},
            }]}
        return None  # "stuck" chat's message fetch fails outright, first page

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    # The healthy chat's newer message must not drag last_check forward while
    # "stuck" has unread history behind it.
    assert pbot.last_check == original_last_check
    # Fix 5: this is the only per-chat-fetch-failure log line an operator gets.
    logged = capsys.readouterr().out
    assert "stuck" in logged
    assert "fetch failed" in logged


# --- Ruling G: an unbounded hold must not become an unbounded duplicate-answer
# machine (R-1) ------------------------------------------------------------------

def test_hold_force_advances_last_check_past_the_threshold(monkeypatch, pbot, capsys):
    """R-1: making last_check conditional on fully_synced (Ruling A) turns one
    persistently-failing chat into an unbounded, self-latching watermark freeze —
    reproduced in the re-review as 8 re-answers of the same question in 39 cycles.
    Once the hold has lasted longer than teams_max_state_age_minutes, last_check
    must advance anyway, mirroring the exact trade _load_state's startup clamp
    already makes for the same reason."""
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    original_last_check = pbot.last_check
    # Simulate a hold that has already outlasted the threshold.
    pbot._hold_since = datetime.now(timezone.utc) - timedelta(minutes=61)

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" in url:
            return {"value": []}
        return None  # the chat list itself is what's failing this cycle

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    assert pbot.last_check > original_last_check
    assert pbot._hold_since is None  # force-advancing must reset the hold clock,
                                      # which is what un-latches a self-latched freeze
    # Fix 5: this is the ONLY record anywhere that user questions were
    # deliberately discarded — if this text is ever dropped, they vanish with
    # no trace at all. Fix 6: this scenario's cause is the chat list itself
    # (fake_api above), not one chat, so the warning must say so distinctly.
    logged = capsys.readouterr().out
    assert "force-advancing" in logged
    assert "permanently skipped" in logged
    assert "chat list itself" in logged


def test_force_advance_warning_distinguishes_one_chat_from_the_whole_list(monkeypatch, pbot, capsys):
    """Fix 6, other branch: the test above covers the chat-list-itself cause;
    this covers a single unreadable chat while the chat list and every other
    chat are fine -- an operator reading the warning needs to know it is NOT
    everyone's traffic at risk, just this one chat's."""
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    pbot._hold_since = datetime.now(timezone.utc) - timedelta(minutes=61)

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" not in url:
            return {"value": [{"id": "stuck"}]}  # the chat list itself is fine
        return None  # this one chat's message fetch always fails

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    logged = capsys.readouterr().out
    assert "one or more individual chats" in logged
    assert "chat list itself" not in logged


def test_force_advance_lets_cleanup_run_the_same_cycle(monkeypatch, pbot):
    """Branch review Fix A: `fully_synced = True` right after the force-advance
    survives mutation with the whole suite green. It is what lets the cleanup
    gate (`if fully_synced: self._cleanup_processed_messages()`, just below)
    fire at the end of a hold -- and since Fix 2 removed _save_state's own
    slice, _cleanup_processed_messages is now the ONLY eviction path left
    during a persistent failure. Delete this line and a chronically-failing
    chat means zero evictions, ever, for as long as it keeps failing --
    unbounded growth in both the in-memory dict and the persisted file."""
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(bot.settings, "teams_max_processed_messages", 10)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    pbot.processed_messages = dict.fromkeys([f"k{i}" for i in range(15)])  # already over cap
    pbot._hold_since = datetime.now(timezone.utc) - timedelta(minutes=61)

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" in url:
            return {"value": []}
        return None  # the chat list itself is what's failing this cycle

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    # The force-advance ended the hold this same cycle, so cleanup must run in
    # it too -- waiting for a separate, later fully-synced cycle is not good
    # enough while the same chat keeps failing and re-triggering the hold.
    assert len(pbot.processed_messages) < 15


def test_force_advance_targets_now_and_leaves_no_re_read_window(monkeypatch, pbot):
    """The force-advance must land on `now`, not behind it.

    This replaces an earlier test that asserted the opposite. Ruling M had the
    force-advance target `now - teams_initial_lookback_minutes`, to avoid
    burying a message that lands in a HEALTHY chat between that chat's own fetch
    earlier in the cycle and the force-advance later in it. That window is real,
    but it was paid for with a rewound watermark -- and `fully_synced = True`
    un-gates _cleanup_processed_messages in the SAME cycle, so the ids evicted
    there are exactly the ones inside the re-opened window. The load harness
    measured the result: 87 of 435 ids evicted, 87 questions answered twice
    (208 at the default teams_max_processed_messages). See
    tests/load/test_teams_load.py::test_force_advance_does_not_re_answer_the_lookback_window.

    Targeting `now` turns the safety of that eviction from an argument into an
    invariant: nothing left in processed_messages can be newer than last_check,
    so no evicted id can be re-accepted by a later fetch. If you are here
    because you spotted Ruling M's narrow mid-cycle hole and want the window
    back -- that is the trade, and it was made deliberately: one buried message
    per teams_max_state_age_minutes at worst, inside a state that already logs
    "may be permanently skipped", against 87-208 duplicate answers.
    """
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(bot.settings, "teams_initial_lookback_minutes", 5)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    pbot._hold_since = datetime.now(timezone.utc) - timedelta(minutes=61)

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" in url:
            return {"value": []}
        return None  # the chat list itself is what's failing this cycle

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    before = datetime.now(timezone.utc)
    pbot.process_new_messages()
    after = datetime.now(timezone.utc)

    # The watermark is `now`, not a lookback behind it. Bracketed rather than
    # compared to a single timestamp, because `now` is read inside the call.
    assert before <= pbot.last_check <= after
    # A message from 2 minutes ago is therefore already behind the watermark,
    # and reads as old on the next fetch -- no re-read window is left open.
    recent = datetime.now(timezone.utc) - timedelta(minutes=2)
    should_process, reason = pbot._should_process_message(
        {"id": "healthy1", "messageType": "message",
         "from": {"user": {"id": "someone"}},
         "createdDateTime": recent.isoformat().replace("+00:00", "Z")},
        my_user_id="me", chat_id="healthy",
    )
    assert should_process is False and reason == "old_message"


def test_force_advance_leaves_nothing_processed_newer_than_the_watermark(monkeypatch, pbot):
    """The invariant that makes the eviction below the force-advance safe.

    This is the load harness's finding reduced to its essence, and it replaces a
    test that asserted dedup could rely on the id set surviving eviction. It
    cannot: _cleanup_processed_messages runs in this very cycle and may drop any
    id at all. So the watermark alone must be enough -- every message already
    marked processed must ALSO be behind last_check once the force-advance has
    run, so that even a fully emptied processed_messages could not produce a
    duplicate answer.
    """
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(bot.settings, "teams_initial_lookback_minutes", 5)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    pbot._hold_since = datetime.now(timezone.utc) - timedelta(minutes=61)

    # A message a healthy chat had answered two minutes ago, during the hold.
    recent = datetime.now(timezone.utc) - timedelta(minutes=2)
    already_answered = {"id": "m1", "messageType": "message",
                        "from": {"user": {"id": "someone"}},
                        "createdDateTime": recent.isoformat().replace("+00:00", "Z")}
    pbot._mark_processed(bot._message_key("healthy", "m1"))

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" in url:
            return {"value": []}
        return None

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    assert pbot.last_check > recent, "the force-advance left an already-answered message ahead of the watermark"
    # Simulate the worst case eviction could ever produce: the id is gone.
    pbot.processed_messages.clear()
    should_process, reason = pbot._should_process_message(
        already_answered, my_user_id="me", chat_id="healthy")
    assert should_process is False and reason == "old_message"


def test_hold_does_not_force_advance_before_the_threshold(monkeypatch, pbot):
    """The other half of Ruling G: a hold that has NOT yet outlasted
    teams_max_state_age_minutes must still behave like Ruling A intended — held,
    not force-advanced. Otherwise every transient blip would force-advance
    immediately and Ruling A's protection would be worthless."""
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    original_last_check = pbot.last_check

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" in url:
            return {"value": []}
        return None

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    assert pbot.last_check == original_last_check
    assert pbot._hold_since is not None  # the hold clock must now be running


def test_hold_clock_resets_on_the_next_healthy_cycle(monkeypatch, pbot):
    """Branch review Fix 4: deleting `self._hold_since = None` on a healthy cycle
    passed the entire 272-test suite. Without it, the clock latches on the very
    first-ever blip and is never cleared by recovery -- from an hour after that
    first blip onward, every LATER transient blip sees held_for > 60 min
    immediately and force-advances on the spot, so Ruling A's protection (hold,
    don't lose) is worth nothing for the rest of the process's life. Two cycles:
    incomplete (the clock must start), then healthy (the clock must clear --
    not merely "not force-advance", which test_hold_does_not_force_advance_
    before_the_threshold above already covers without proving the clock itself
    resets)."""
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)

    # Cycle 1: chat list fails outright -> incomplete, the hold clock starts.
    monkeypatch.setattr(pbot, "_api_request",
                         lambda url, method="GET", json_data=None, retry=False: None)
    pbot.process_new_messages()
    assert pbot._hold_since is not None

    # Cycle 2: fully healthy (empty chat list, no chats to fail) -> the clock
    # must clear here, not just "not force-advance" -- nothing else in a
    # genuinely-synced cycle like this one touches _hold_since.
    monkeypatch.setattr(pbot, "_api_request",
                         lambda url, method="GET", json_data=None, retry=False: {"value": []})
    pbot.process_new_messages()
    assert pbot._hold_since is None


def test_eviction_is_skipped_while_the_watermark_is_held(monkeypatch, pbot):
    """Ruling H: evicting while the watermark is frozen is exactly what re-delivers
    an old answer (R-1) — an evicted id becomes re-answerable the moment its chat's
    back-paging (Ruling B) reaches far enough. Eviction must wait for a cycle that
    actually advances last_check (normally or via Ruling G's force-advance)."""
    monkeypatch.setattr(bot.settings, "teams_max_processed_messages", 5)
    # Comfortably above the ~0s this single cycle could ever hold for, so Ruling
    # G's force-advance can't mask the assertion regardless of the real .env value.
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    monkeypatch.setattr(pbot, "_send_message", lambda *a, **k: True)
    pbot.processed_messages = dict.fromkeys([f"old{i}" for i in range(10)])  # already over cap

    def fake_api(url, method="GET", json_data=None, retry=False):
        if "/messages" in url:
            return {"value": []}
        return None  # chat list fails -> fully_synced False, hold just begins

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    # Held (the hold just started, nowhere near the threshold) -> cleanup must
    # not have run, or "old0".."old9" would already be candidates for eviction.
    assert len(pbot.processed_messages) == 10


def test_eviction_still_runs_on_a_normal_synced_cycle(monkeypatch, pbot):
    """Regression guard for the Ruling H gate: it must only SKIP eviction while
    held, not accidentally stop cleanup from ever running again on a healthy bot."""
    monkeypatch.setattr(bot.settings, "teams_max_processed_messages", 5)
    monkeypatch.setattr(pbot, "_get_my_user_id", lambda: "me")
    pbot.processed_messages = dict.fromkeys([f"old{i}" for i in range(10)])

    def fake_api(url, method="GET", json_data=None, retry=False):
        return {"value": []}  # empty chat list; nothing to fail, nothing to hold

    monkeypatch.setattr(pbot, "_api_request", fake_api)

    pbot.process_new_messages()

    assert len(pbot.processed_messages) < 10
