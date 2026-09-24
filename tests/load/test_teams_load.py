"""Load/soak scenarios for the Teams poll loop at 30-chat scale.

Every existing bot test exercises one or two chats for a handful of cycles. The
four bugs found in this branch's review all lived in the INTERACTION between the
watermark gates -- the completeness flag, the hold timer, the bounded
force-advance and the eviction gate -- rather than in any single one of them.
These scenarios run all of it together, at 30 chats, over simulated hours.

The scenario code is only what makes the invariants meaningful; the invariants
themselves (see Simulation.assert_core_invariants) are the point:

  * no message is ever answered twice        * no accepted message is lost
  * ack precedes reply                       * last_check never moves backwards
  * processed_messages and the state file stay bounded
  * exactly one worker thread exists, ever   * no exception escapes the poll loop

Everything runs in-process: see tests/load/harness.py for the seams, and
tests/load/conftest.py for the autouse fixture that turns any real HTTP call
into a test failure.
"""

import json
import random
from datetime import datetime, timedelta

import httpx
import pytest
import requests

import channels.teams.bot as bot
from tests.load.harness import (
    CYCLIC_NEXTLINK,
    INTERMITTENT_TIMEOUT,
    PERSISTENT_4XX,
    FakeClock,
    FakeGraph,
    NetworkForbidden,
)

# Bounds asserted against, expressed once so a scenario cannot quietly relax one.
# teams_max_processed_messages is pinned to 200 in the build_sim fixture;
# _cleanup_processed_messages is a TRIGGER, not a cap (it removes 20% of the
# current size when crossed), so the healthy steady state sits just above 200.
MAX_PROCESSED_HEALTHY = 400
MAX_STATE_BYTES_HEALTHY = 200_000
# A burst marks every message in it within a SINGLE cycle, and the eviction gate
# runs once per cycle, at the end -- so the resident set legitimately overshoots
# the trigger by the size of the burst before it is trimmed. This is the "expect
# ~5x this number, not this number" behaviour documented on
# _cleanup_processed_messages; measured peak for a 180-message burst is 432.
MAX_PROCESSED_BURST = 1_000
# During a hold, eviction is blocked by design (Ruling H), so the set grows with
# the traffic seen across the hold. Bounded, but at a different order.
MAX_PROCESSED_HELD = 6_000
MAX_STATE_BYTES_HELD = 1_000_000


# --- 1. steady trickle --------------------------------------------------------

def test_steady_trickle_across_thirty_chats(build_sim):
    """Ordinary traffic: a few questions per cycle, scattered over 30 chats, for
    60 cycles (30 simulated minutes). Nothing fails; this is the baseline the
    failure scenarios are read against."""
    sim = build_sim(n_chats=30)
    rng = random.Random(4242)
    tokens = []

    def inject(cycle):
        for _ in range(3):
            chat_id = sim.graph.chat_ids[rng.randrange(len(sim.graph.chat_ids))]
            token = f"Q{len(tokens):04d}"
            tokens.append(token)
            sim.graph.add_user_message(chat_id, token)

    sim.run_cycles(60, inject=inject)
    sim.run_cycles(3)  # quiet cycles: nothing new must appear out of the backlog

    assert len(tokens) == 180
    sim.assert_core_invariants(tokens, max_processed=MAX_PROCESSED_HEALTHY,
                               max_state_bytes=MAX_STATE_BYTES_HEALTHY)
    assert sim.live_worker_count() == 1
    # Eviction must actually have fired, or "stays bounded" proves nothing: the
    # set only ever shrinks because _cleanup_processed_messages ran.
    sizes = sim.processed_sizes
    assert any(b < a for a, b in zip(sizes, sizes[1:])), \
        "processed_messages never shrank, so eviction was never exercised"


# --- 2. burst -----------------------------------------------------------------

def test_burst_of_many_messages_in_a_single_cycle(build_sim):
    """180 questions land in one poll cycle -- six per chat across all 30. The
    poll thread must detect, ack and enqueue every one of them without blocking
    on the worker, and the single worker must answer each exactly once."""
    sim = build_sim(n_chats=30)
    sim.run_cycle()  # establish a watermark first
    tokens = []

    def inject(cycle):
        for chat_id in sim.graph.chat_ids:
            for _ in range(6):
                token = f"B{len(tokens):04d}"
                tokens.append(token)
                sim.graph.add_user_message(chat_id, token)

    sim.run_cycle(inject=inject)
    sim.run_cycles(3)

    assert len(tokens) == 180
    # The burst genuinely piled up behind the single worker rather than being
    # answered inline by the poll thread.
    assert max(sim.queue_depths) >= 150, f"queue never filled: {max(sim.queue_depths)}"
    sim.assert_core_invariants(tokens, max_processed=MAX_PROCESSED_BURST,
                               max_state_bytes=MAX_STATE_BYTES_HEALTHY)
    assert sim.live_worker_count() == 1


# --- 3. one chat failing on every cycle, run past the 60-minute bound ----------

def test_persistently_failing_chat_holds_then_force_advances(build_sim, capsys):
    """The force-advance path is unreachable without simulating an hour.

    chat00 fails on every single cycle. For 60 simulated minutes the watermark
    must be HELD -- even though 29 healthy chats keep producing newer messages --
    and then force-advanced exactly once, with the loss logged. Throughout,
    healthy chats must keep being answered exactly once each."""
    sim = build_sim(n_chats=30)
    sim.graph.set_failure("chat00", PERSISTENT_4XX)
    rng = random.Random(97531)
    healthy_tokens = []
    stuck_tokens = []

    def inject(cycle):
        chat_id = sim.graph.chat_ids[1 + rng.randrange(len(sim.graph.chat_ids) - 1)]
        token = f"H{len(healthy_tokens):04d}"
        healthy_tokens.append(token)
        sim.graph.add_user_message(chat_id, token)
        if cycle % 20 == 0:
            token = f"S{len(stuck_tokens):04d}"
            stuck_tokens.append(token)
            sim.graph.add_user_message("chat00", token)

    original = sim.tbot.last_check
    # 130 cycles x 30s = 65 simulated minutes: past the 60-minute bound, and not
    # far enough past it for a second force-advance.
    sim.run_cycles(130, advance=timedelta(seconds=30), inject=inject)

    jumped = [i for i, w in enumerate(sim.watermarks) if w > original]
    assert jumped, "the watermark was never force-advanced past the 60-minute bound"
    first_jump = jumped[0]
    # 30s per cycle, so the bound cannot be crossed before cycle 120.
    assert first_jump >= 120, f"force-advanced too early, at cycle {first_jump}"
    # Held, flat, for every cycle before that -- despite healthy traffic.
    assert all(w == original for w in sim.watermarks[:first_jump]), \
        "last_check moved while a chat was unreadable"

    logged = capsys.readouterr().out
    assert "force-advancing" in logged
    assert "permanently skipped" in logged
    # This scenario is ONE unreadable chat, not the chat list itself.
    assert "one or more individual chats" in logged

    sim.assert_no_poll_loop_exceptions()
    sim.assert_no_duplicate_answers()
    sim.assert_all_answered(healthy_tokens)
    sim.assert_ack_precedes_reply()
    sim.assert_watermark_never_regresses()
    sim.assert_one_worker_ever()
    sim.assert_bounded(max_processed=MAX_PROCESSED_HELD,
                       max_state_bytes=MAX_STATE_BYTES_HELD)
    assert sim.live_worker_count() == 1
    # Nothing from the unreadable chat was ever answered -- that is the
    # documented trade the force-advance makes, not a surprise.
    answered = sim.answered_tokens()
    assert not [t for t in stuck_tokens if t in answered]
    # Eviction is blocked for the whole hold and must resume at the force-advance.
    assert min(sim.processed_sizes[first_jump:]) < max(sim.processed_sizes[:first_jump])


# --- 4. intermittent seeded failures across random chats ----------------------

def test_intermittent_failures_never_lose_or_duplicate_a_message(build_sim):
    """Eight chats time out at random (seeded) on ~30% of cycles. A failed fetch
    holds the watermark, so those chats are retried next cycle; every message
    must still be answered, exactly once, once the failures clear."""
    sim = build_sim(n_chats=30, seed=13579)
    rng = random.Random(24680)
    flaky = [sim.graph.chat_ids[i] for i in (2, 5, 9, 11, 17, 21, 26, 29)]
    for chat_id in flaky:
        sim.graph.set_failure(chat_id, INTERMITTENT_TIMEOUT, rate=0.3)
    tokens = []

    def inject(cycle):
        for _ in range(2):
            chat_id = sim.graph.chat_ids[rng.randrange(len(sim.graph.chat_ids))]
            token = f"I{len(tokens):04d}"
            tokens.append(token)
            sim.graph.add_user_message(chat_id, token)

    # 36 cycles of turbulence (18 simulated minutes -- comfortably inside the
    # 60-minute bound, so nothing is force-advanced away)...
    sim.run_cycles(36, advance=timedelta(seconds=30), inject=inject)
    held = sim.tbot.last_check
    # ...then the outage clears and the backlog must drain completely.
    for chat_id in flaky:
        sim.graph.failures.pop(chat_id, None)
    sim.run_cycles(4, advance=timedelta(seconds=30))

    assert len(tokens) == 72
    sim.assert_core_invariants(tokens, max_processed=MAX_PROCESSED_HELD,
                               max_state_bytes=MAX_STATE_BYTES_HELD)
    assert sim.live_worker_count() == 1
    # Recovery: once every fetch succeeds the watermark is released again.
    assert sim.tbot.last_check > held


# --- 5. cyclic @odata.nextLink ------------------------------------------------

def test_cyclic_chat_list_nextlink_terminates(build_sim, capsys):
    """A self-referential @odata.nextLink on /me/chats would spin the poll thread
    forever without ever raising: nothing would fail, consecutive_errors would
    never trip, and the bot would go silently dead while still "running"."""
    sim = build_sim(n_chats=30, call_ceiling=2_000)
    sim.graph.set_chat_list_failure(CYCLIC_NEXTLINK)
    original = sim.tbot.last_check

    sim.run_cycles(3)  # returns at all == it terminated

    sim.assert_no_poll_loop_exceptions()
    logged = capsys.readouterr().out
    assert "cyclic or self-referential" in logged
    # A cut-short chat list is incomplete, so the watermark must be held.
    assert sim.tbot.last_check == original
    sim.assert_watermark_never_regresses()


def test_cyclic_message_nextlink_terminates(build_sim, capsys):
    """Same failure one level down: a chat whose message pages link to
    themselves. _get_chat_messages has no seen-URL set, so the page cap is the
    only thing standing between this and an infinite loop."""
    sim = build_sim(n_chats=30, call_ceiling=5_000)
    sim.graph.set_failure("chat07", CYCLIC_NEXTLINK)
    original = sim.tbot.last_check
    tokens = []

    def inject(cycle):
        token = f"C{len(tokens):04d}"
        tokens.append(token)
        sim.graph.add_user_message("chat07", token)
        sim.graph.add_user_message("chat08", token + "x")

    sim.run_cycles(3, inject=inject)

    sim.assert_no_poll_loop_exceptions()
    logged = capsys.readouterr().out
    assert "cap" in logged and "chat07" in logged
    # Per-chat paging is bounded by _MESSAGES_MAX_PAGES, so the cycle ends.
    assert sim.graph.calls < 5_000
    # The looping chat leaves the cycle incomplete, so the watermark is held.
    assert sim.tbot.last_check == original
    sim.assert_no_duplicate_answers()


# --- 6. restart with messages in flight ---------------------------------------

def test_restart_redelivers_an_inflight_question_exactly_once(build_sim):
    """The at-least-once guarantee, end to end across a process restart.

    Phase 1 answers three questions normally. Phase 2 is a bot that accepts and
    acks a fourth and then dies before its worker ever runs it -- modelled by a
    fresh TeamsBot with no worker started, which is literally that situation.
    Phase 3 restarts from the same state file and must re-deliver exactly that
    one question, answer it exactly once, and re-answer nothing else."""
    clock = FakeClock()
    graph = FakeGraph(clock, n_chats=5, chats_per_page=3)
    tokens = []

    # --- phase 1: normal operation ---
    sim_a = build_sim(graph=graph, clock=clock)
    def inject_answered(cycle):
        token = f"P{len(tokens)}"
        tokens.append(token)
        graph.add_user_message("chat01", token)
    sim_a.run_cycles(3, inject=inject_answered)
    assert sorted(sim_a.answered_tokens()) == ["P0", "P1", "P2"]
    # One quiet cycle so _save_state runs with nothing in flight. Without it the
    # file still carries the rewind for P2 -- which drained AFTER cycle 3's save
    # -- and a restart re-delivers it. That is the at-least-once guarantee
    # behaving as designed (the window is one poll interval, cleared by the very
    # next save), not a defect, but it must not be mistaken for the in-flight
    # re-delivery this test is actually about.
    sim_a.run_cycle()
    clean = json.loads(bot.STATE_FILE.read_text())
    assert datetime.fromisoformat(clean["last_check"]) == sim_a.tbot.last_check

    # --- phase 2: a question accepted, acked, and lost to a crash ---
    sim_crash = build_sim(graph=graph, clock=clock, start_worker=False)
    sim_crash.run_cycle(inject=lambda c: graph.add_user_message("chat02", "INFLIGHT"))
    assert sim_crash.tbot._work_q.qsize() == 1, "the question was never enqueued"
    assert len(sim_crash.tbot._inflight) == 1, "the question was not held in flight"

    saved = json.loads(bot.STATE_FILE.read_text())
    inflight_created = next(iter(sim_crash.tbot._inflight.values()))
    # The persisted watermark is deliberately rewound behind the unanswered
    # question, and its id is deliberately withheld from processed_messages --
    # together, that is what makes the restart re-deliver it.
    assert datetime.fromisoformat(saved["last_check"]) < inflight_created
    assert not [k for k in saved["processed_messages"] if k in sim_crash.tbot._inflight]

    # --- phase 3: restart ---
    sim_b = build_sim(graph=graph, clock=clock)
    sim_b.run_cycles(3)

    answered = sim_b.answered_tokens()
    assert answered.get("INFLIGHT") == 1, \
        f"the in-flight question was not re-delivered exactly once: {answered}"
    sim_b.assert_no_duplicate_answers()
    sim_b.assert_all_answered(tokens + ["INFLIGHT"])
    sim_b.assert_ack_precedes_reply()
    sim_b.assert_watermark_never_regresses()
    sim_b.assert_no_poll_loop_exceptions()
    # Each bot instance ran exactly one worker of its own.
    sim_b.assert_one_worker_ever()
    sim_a.assert_one_worker_ever()


# --- harness self-check -------------------------------------------------------

def test_the_harness_actually_exercises_the_code_it_claims_to(build_sim, capsys):
    """A load test that silently does nothing is worse than no load test.

    Pins the volume down so a future refactor that short-circuits the fake Graph
    (or the worker) fails here instead of quietly passing every scenario above.
    """
    sim = build_sim(n_chats=30)
    tokens = []

    def inject(cycle):
        for i in range(3):
            token = f"V{len(tokens):04d}"
            tokens.append(token)
            sim.graph.add_user_message(sim.graph.chat_ids[(cycle + i) % 30], token)

    sim.run_cycles(20, inject=inject)

    # Every chat was polled every cycle, through real pagination.
    assert sim.graph.calls > 20 * 30, f"only {sim.graph.calls} Graph calls in 20 cycles"
    # Every question reached the stubbed pipeline exactly once...
    assert sorted(sim.rag_calls) == sorted(tokens)
    # ...and every ack and reply was a POST through the same seam.
    assert len(sim.graph.sends) == 2 * len(tokens)
    # The chat list really paged: 30 chats at 12 per page is 3 pages a cycle,
    # so @odata.nextLink was followed twice per cycle, not once and stopped.
    assert sim.graph.chat_list_pages == 3 * 20
    # And every chat's messages were fetched every cycle, at least one page each.
    assert sim.graph.message_pages >= 20 * 30
    sim.assert_core_invariants(tokens, max_processed=MAX_PROCESSED_HEALTHY,
                               max_state_bytes=MAX_STATE_BYTES_HEALTHY)


# --- regression: eviction and the force-advance must not overlap --------------

def test_force_advance_does_not_re_answer_the_lookback_window(build_sim, monkeypatch):
    """Regression for a duplicate-answer defect this harness found and measured.

    Before the fix, process_new_messages force-advanced last_check to
    ``now - teams_initial_lookback_minutes`` (Ruling M) and then set
    ``fully_synced = True``, which un-gated ``_cleanup_processed_messages`` in
    the SAME cycle. The eviction dropped the oldest 20% of the resident set --
    which, whenever the hold's traffic was concentrated in its final few
    minutes, were exactly the ids sitting inside the window the rewind had just
    re-opened. Measured here, cycle by cycle:

      cycle 113  a burst lands at 09:56:30, is answered, ids marked processed
      cycle 122  the hold passes 60 min -> last_check force-advanced 08:55 ->
                 09:56 (= now - lookback, because nothing arrived THIS cycle,
                 so newest_message_time never moved off the frozen watermark)
                 -> fully_synced = True -> cleanup evicted 87 of 435 ids
      cycle 123  those 87 messages were newer than last_check and no longer in
                 processed_messages, so they were re-fetched, re-acked and
                 RE-ANSWERED: 87 duplicate answers, 174 extra Graph sends

    The fix targets ``now`` instead, which makes the eviction safe by
    construction rather than by argument: nothing left in processed_messages can
    be newer than last_check, so no evicted id can be re-accepted. This scenario
    is kept exactly as it was when it produced 87 duplicates -- that is the
    proof. Restoring the ``- timedelta(minutes=...lookback)`` term in
    process_new_messages makes this test fail again (mutation-verified).
    """
    monkeypatch.setattr(bot.settings, "teams_max_processed_messages", 200)
    sim = build_sim(n_chats=30)
    sim.graph.set_failure("chat00", PERSISTENT_4XX)  # holds the watermark from cycle 1
    tokens = []

    # ~56 minutes of hold with no traffic at all (overnight, or a weekend).
    sim.run_cycles(112, advance=timedelta(seconds=30))

    def morning_burst(cycle):
        for chat_id in sim.graph.chat_ids[1:]:
            for _ in range(3):
                token = f"E{len(tokens):04d}"
                tokens.append(token)
                sim.graph.add_user_message(chat_id, token)

    sim.run_cycle(advance=timedelta(seconds=30), inject=morning_burst)
    # A lull, then the 60-minute bound trips the force-advance.
    sim.run_cycles(12, advance=timedelta(seconds=30))

    assert len(tokens) == 87
    # The scenario must actually reach the force-advance, or it proves nothing.
    assert sim.watermarks[-1] > sim.watermarks[0], "the hold never force-advanced"
    sim.assert_all_answered(tokens)
    sim.assert_no_duplicate_answers()
    # Eviction must actually have run in the force-advance cycle -- that is the
    # half of the interaction the fix makes safe rather than removes.
    assert any(b < a for a, b in zip(sim.processed_sizes, sim.processed_sizes[1:])), \
        "cleanup never fired, so the eviction/force-advance overlap was never exercised"
    sim.assert_ack_precedes_reply()
    sim.assert_watermark_never_regresses()
    sim.assert_no_poll_loop_exceptions()


# --- determinism --------------------------------------------------------------

def test_two_identical_runs_produce_identical_behaviour(build_sim):
    """A flaky load test is worse than none, because people start ignoring it.

    Rather than trusting that the harness is deterministic, prove it: run the
    same turbulent scenario twice, from independent state files, and require the
    two runs to agree byte for byte on everything observable -- the exact
    sequence of Graph sends, the watermark after every cycle, and the resident
    set size after every cycle.

    The one thing deliberately NOT compared is the logged queue depth. That is
    ``_work_q.qsize()``, which bot.py documents as approximate under concurrency
    ("fine for a log line, not worth a lock"); it varies by +/-1 with how quickly
    the worker thread wakes to take the first job off the queue. It is a
    diagnostic number, not behaviour, and it is the only thing that moves: 2217
    lines of bot log are otherwise identical across runs.
    """
    def one_run(state_name):
        sim = build_sim(n_chats=30, seed=13579, state_name=state_name)
        rng = random.Random(24680)
        flaky = [sim.graph.chat_ids[i] for i in (2, 5, 9, 11, 17, 21, 26, 29)]
        for chat_id in flaky:
            sim.graph.set_failure(chat_id, INTERMITTENT_TIMEOUT, rate=0.3)
        counter = [0]

        def inject(cycle):
            for _ in range(2):
                chat_id = sim.graph.chat_ids[rng.randrange(len(sim.graph.chat_ids))]
                counter[0] += 1
                sim.graph.add_user_message(chat_id, f"D{counter[0]:04d}")

        sim.run_cycles(30, advance=timedelta(seconds=30), inject=inject)
        return {
            "sends": [(s.chat_id, s.html) for s in sim.graph.sends],
            "watermarks": [w.isoformat() for w in sim.watermarks],
            "processed": list(sim.processed_sizes),
            "calls": (sim.graph.calls, sim.graph.chat_list_pages, sim.graph.message_pages),
        }

    first = one_run("run_a.json")
    second = one_run("run_b.json")

    assert first["calls"] == second["calls"]
    assert first["watermarks"] == second["watermarks"]
    assert first["processed"] == second["processed"]
    assert first["sends"] == second["sends"]
    assert len(first["sends"]) > 100, "the scenario did too little to prove anything"


# --- the isolation guard, proven rather than promised -------------------------

def test_the_network_poison_fixture_actually_fires():
    """The autouse guard is only worth having if it is known to work."""
    with pytest.raises(NetworkForbidden):
        requests.get("https://graph.microsoft.com/v1.0/me/chats")
    with pytest.raises(NetworkForbidden):
        requests.post("https://graph.microsoft.com/v1.0/me/chats/x/messages")
    with pytest.raises(NetworkForbidden):
        httpx.get("http://172.20.0.22:11434/api/ps")
    with pytest.raises(NetworkForbidden):
        httpx.post("http://172.20.0.22:6333/collections")


def test_an_unpatched_graph_call_cannot_be_swallowed(monkeypatch, tmp_path):
    """_api_request ends in ``except Exception: ... return None``.

    If the guard were an ordinary Exception, a code path that reached Graph for
    real would be caught there, logged as a routine "Request failed", and the
    cycle would carry on -- the suite would stay green while talking to
    production. The guard must escape that handler.
    """
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    unpatched = bot.TeamsBot(token_refresher=object())
    monkeypatch.setattr(unpatched, "_get_headers", lambda: {})

    with pytest.raises(NetworkForbidden):
        unpatched._api_request(f"{bot.GRAPH_API}/me/chats")
    with pytest.raises(NetworkForbidden):
        unpatched._send_message("chat01", "<p>should never be delivered</p>")
