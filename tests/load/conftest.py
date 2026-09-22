"""Fixtures for the Teams load/soak harness.

The network-poison fixture is autouse, so every test in this directory proves
its own isolation rather than promising it: if anyone later adds a code path
that reaches out to Graph, the GPU host, or anything else, these tests fail
loudly instead of silently hitting production.
"""

import queue
import threading

import httpx
import pytest
import requests

import channels.teams.bot as bot
from tests.load.harness import (
    BOT_USER_ID,
    FakeClock,
    FakeGraph,
    NetworkForbidden,
    Simulation,
    install_clock,
    make_rag_stub,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make any real HTTP call an immediate, unmistakable test failure."""
    def _forbid(*args, **kwargs):
        raise NetworkForbidden(
            "the load harness attempted a real network call -- it must never reach "
            "Microsoft Graph or the shared model host; check the _api_request / "
            "_run_rag seams"
        )

    for module, name in (
        (requests, "get"), (requests, "post"), (requests, "request"),
        (httpx, "get"), (httpx, "post"), (httpx, "request"),
    ):
        monkeypatch.setattr(module, name, _forbid)


@pytest.fixture
def build_sim(monkeypatch, tmp_path):
    """Build one or more TeamsBot simulations that share a clock, Graph and state file.

    Every seam that keeps this offline is applied here, in one place, so no
    scenario can forget one.
    """
    gates: list[threading.Event] = []
    # Point STATE_FILE away from the developer's real bot_state.json before any
    # bot can be built; _build overrides it per run when a scenario asks for its
    # own state. The rotating credential is never opened either:
    # token_refresher=object() means TokenRefresher is never constructed.
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "bot_state.json")
    # _answer imports rag.router and makes a real LLM call when this is on.
    monkeypatch.setattr(bot.settings, "router_enabled", False)
    monkeypatch.setattr(bot.settings, "teams_messages_page_size", 5)
    monkeypatch.setattr(bot.settings, "teams_max_state_age_minutes", 60)
    monkeypatch.setattr(bot.settings, "teams_initial_lookback_minutes", 5)
    # Small on purpose: eviction must actually fire during a run of this length,
    # so the eviction/watermark interaction is exercised rather than assumed.
    monkeypatch.setattr(bot.settings, "teams_max_processed_messages", 200)
    bot._pending_ratings.clear()

    def _build(graph=None, clock=None, *, n_chats=30, seed=20260916,
               start_worker=True, rag_calls=None, errors=None,
               state_name="bot_state.json", **graph_kwargs):
        # Bots built with the same state_name share a state file -- that is what
        # makes the restart scenario a restart. A different name gives a run its
        # own, independent state.
        this_state = tmp_path / state_name
        monkeypatch.setattr(bot, "STATE_FILE", this_state)
        clock = clock or FakeClock()
        install_clock(monkeypatch, clock)
        graph = graph if graph is not None else FakeGraph(
            clock, n_chats=n_chats, seed=seed, **graph_kwargs)

        gate = threading.Event()
        gates.append(gate)
        rag_calls = rag_calls if rag_calls is not None else []
        errors = errors if errors is not None else []
        # Stubbing _run_rag means the deferred rag.* imports inside it never run,
        # so llama-index, Ollama and Qdrant are never even imported.
        monkeypatch.setattr(bot, "_run_rag", make_rag_stub(gate, rag_calls, errors))

        tbot = bot.TeamsBot(token_refresher=object())
        # All Graph traffic -- chat list, per-chat messages, and every send --
        # funnels through this one method.
        monkeypatch.setattr(tbot, "_api_request", graph.api_request)
        # _get_my_user_id would otherwise call token_refresher.get_access_token().
        monkeypatch.setattr(tbot, "_get_my_user_id", lambda: BOT_USER_ID)
        tbot._work_q = queue.Queue()

        return Simulation(
            tbot=tbot, graph=graph, clock=clock, gate=gate, state_file=this_state,
            rag_calls=rag_calls, errors=errors, start_worker=start_worker,
        )

    yield _build

    # Release any worker parked on a gate so no thread is left mid-question.
    for gate in gates:
        gate.set()
