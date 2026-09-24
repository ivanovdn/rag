"""
Unit tests for the request-level Phoenix/OTel span channels/teams/bot.py's _answer
opens: "compliance_request", a single root span for the whole request that must be
attached (made "current") before the router call, so that the router's classification
span and — via _run_rag's asyncio.run + search_policies_tool's asyncio.to_thread — the
agent run and its retrieval leaves all nest under it instead of each becoming its own
disconnected trace.

Uses an in-memory OTel span exporter — never a live Phoenix instance — and mocks the
agent (rag.agent.build_agent), the router (rag.router.classify_message) and Graph sends
(TeamsBot._send_message), so nothing here touches the network or an LLM.
"""

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import channels.teams.bot as bot_mod
import rag.agent as agent_mod
import rag.embeddings as embeddings_mod
import rag.observability as observability_mod
import rag.reranker as reranker_mod
import rag.response as response_mod
import rag.router as router_mod
import rag.search_first as search_first_mod
import rag.tools.search_policies as search_policies_mod
import rag.vector_store as vector_store_mod
from rag.router import Category, RouterDecision


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def request_span_tracer(monkeypatch):
    """Patch every module's get_tracer to a single tracer backed by an in-memory
    exporter: the module-top-level bound copies in embeddings/vector_store/reranker
    (stale-import gotcha — see CLAUDE.md) AND rag.observability's own `get_tracer`,
    which bot.py's and record_classification's deferred, function-local imports
    re-resolve fresh on every call."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(observability_mod, "get_tracer", lambda: tracer)
    monkeypatch.setattr(embeddings_mod, "get_tracer", lambda: tracer)
    monkeypatch.setattr(vector_store_mod, "get_tracer", lambda: tracer)
    monkeypatch.setattr(reranker_mod, "get_tracer", lambda: tracer)
    return tracer, exporter


@pytest.fixture
def teams_bot(monkeypatch, tmp_path):
    """A TeamsBot with Graph sends mocked; records every HTML it 'sends'.
    Mirrors tests/unit/test_bot_routing.py's fixture of the same name."""
    monkeypatch.setattr(bot_mod, "STATE_FILE", tmp_path / "bot_state.json")
    b = bot_mod.TeamsBot(token_refresher=object())
    sent = []
    monkeypatch.setattr(
        b,
        "_send_message",
        lambda chat_id, text, content_type="html", retry=False: sent.append(text) or True,
    )
    bot_mod._pending_ratings.clear()
    b._sent = sent
    return b


class _FakePointWithPayload:
    """Minimal Qdrant ScoredPoint stand-in, just enough for search_policies()'s
    non-BM25 path (`r.score`, `r.payload[...]`) to run end to end."""

    def __init__(self, score, payload, point_id="chunk-id"):
        self.score = score
        self.payload = payload
        self.id = point_id


class _FakeQueryResponse:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
    def __init__(self, points):
        self._points = points

    def query_points(self, **kwargs):
        return _FakeQueryResponse(self._points)


def _fake_agent_class(tracer):
    """A fake `build_agent()` result whose async `run()` opens a span the way
    OpenInference's real llama-index instrumentor does: `context=None`, i.e.
    "parent = whatever is current". This is the exact mechanism the root span
    exists to feed — nothing here is llama-index, so no LLM/network is touched."""

    class _FakeAgent:
        async def run(self, user_msg):
            with tracer.start_as_current_span("fake_agent_root", context=None):
                pass
            return "fake response"

    return _FakeAgent()


# --- the asyncio.run hop (_run_rag) --------------------------------------------


def test_run_rag_asyncio_hop_shares_root_trace_id(monkeypatch, request_span_tracer):
    """_run_rag's `asyncio.run(_run())` must not disconnect spans opened inside the
    agent it builds. Proves the asyncio hop specifically, independent of the
    to_thread hop covered below."""
    tracer, exporter = request_span_tracer
    # Retrieval now runs before the agent is even built (rag.search_first.prefetch) —
    # it must resolve to "ok" or this would reach a real, un-mocked search and never
    # get as far as build_agent at all.
    monkeypatch.setattr(search_first_mod, "prefetch", lambda q: search_first_mod.PrefetchResult("ok", "[Source 1] x"))
    monkeypatch.setattr(agent_mod, "build_agent", lambda: _fake_agent_class(tracer))
    monkeypatch.setattr(
        response_mod,
        "parse_agent_response",
        lambda s: {"answer": "ok", "citations": [], "escalation": {"needed": False}},
    )

    with tracer.start_as_current_span("compliance_request") as root:
        root_trace_id = root.get_span_context().trace_id
        bot_mod._run_rag("does the VPN policy cover contractors?")

    span = next(s for s in exporter.get_finished_spans() if s.name == "fake_agent_root")
    assert span.context.trace_id == root_trace_id


# --- the asyncio.to_thread hop (search_policies_tool, Change 1) ----------------


def test_search_policies_to_thread_hop_shares_root_trace_id(monkeypatch, request_span_tracer):
    """search_policies_tool's async_fn (Change 1) runs the sync search_policies via
    asyncio.to_thread. Under an ambient current span, a leaf span opened inside the
    tool's worker thread (embed_query) must share the parent's trace id — Change 1
    actually exercised end to end, not just unit-tested in isolation."""
    tracer, exporter = request_span_tracer
    monkeypatch.setattr(search_policies_mod.settings, "bm25_enabled", False)
    monkeypatch.setattr(search_policies_mod.settings, "reranker_enabled", False)
    monkeypatch.setattr(search_policies_mod.settings, "min_confidence_score", 0.0)
    monkeypatch.setattr(embeddings_mod, "_embedding_model", None)
    monkeypatch.setattr(embeddings_mod.settings, "embedding_source", "ollama")
    monkeypatch.setattr(embeddings_mod, "_ollama_embed", lambda texts, prefix="": [[0.1, 0.2]])
    fake_points = [_FakePointWithPayload(0.9, {"doc_title": "T", "doc_id": "d", "text": "x"}, "c1")]
    monkeypatch.setattr(vector_store_mod, "get_qdrant_client", lambda: _FakeQdrantClient(fake_points))

    with tracer.start_as_current_span("compliance_request") as root:
        root_trace_id = root.get_span_context().trace_id
        asyncio.run(search_policies_mod.search_policies_tool.acall(query="q", top_k=1))

    span = next(s for s in exporter.get_finished_spans() if s.name == "embed_query")
    assert span.context.trace_id == root_trace_id


# --- end-to-end through _answer: router + agent both mocked --------------------


def test_answer_nests_router_and_agent_spans_under_compliance_request(
    monkeypatch, request_span_tracer, teams_bot
):
    """Drives the real _answer with the agent, router and Graph sends mocked (never
    the network or an LLM): the classification span (router path) and a span opened
    inside the mocked agent must both land in the same trace as the compliance_request
    root span _answer opens before either runs."""
    tracer, exporter = request_span_tracer
    monkeypatch.setattr(bot_mod.settings, "router_enabled", True)
    monkeypatch.setattr(
        router_mod,
        "classify_message",
        lambda text: RouterDecision(category=Category.IN_SCOPE, confidence=0.95),
    )
    # Retrieval now runs before the agent is even built (rag.search_first.prefetch) —
    # it must resolve to "ok" or this would reach a real, un-mocked search and never
    # get as far as build_agent at all.
    monkeypatch.setattr(search_first_mod, "prefetch", lambda q: search_first_mod.PrefetchResult("ok", "[Source 1] x"))
    monkeypatch.setattr(agent_mod, "build_agent", lambda: _fake_agent_class(tracer))
    monkeypatch.setattr(
        response_mod,
        "parse_agent_response",
        lambda s: {"answer": "See AUP.", "citations": [], "escalation": {"needed": False}},
    )

    assert teams_bot._answer("chat1", "Can I install software?")

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert {"compliance_request", "classification", "fake_agent_root"} <= set(spans)
    root_trace_id = spans["compliance_request"].context.trace_id
    assert spans["classification"].context.trace_id == root_trace_id
    assert spans["fake_agent_root"].context.trace_id == root_trace_id


# --- outcome attribute ----------------------------------------------------------


def test_outcome_attribute_set_on_greeting_path(monkeypatch, request_span_tracer, teams_bot):
    _tracer, exporter = request_span_tracer
    monkeypatch.setattr(bot_mod.settings, "router_enabled", True)
    monkeypatch.setattr(
        router_mod,
        "classify_message",
        lambda text: RouterDecision(category=Category.GREETING, confidence=0.95),
    )

    teams_bot._answer("chat1", "hello")

    span = next(s for s in exporter.get_finished_spans() if s.name == "compliance_request")
    assert span.attributes["compliance_request.outcome"] == "greeting"
    assert span.attributes["compliance_request.question"] == "hello"
    # Never record anything that identifies a person on this span.
    assert "chat_id" not in span.attributes
    assert "sender_name" not in span.attributes


def test_outcome_attribute_set_on_answered_path(monkeypatch, request_span_tracer, teams_bot):
    _tracer, exporter = request_span_tracer
    monkeypatch.setattr(bot_mod.settings, "router_enabled", False)
    monkeypatch.setattr(
        bot_mod,
        "_run_rag",
        lambda q: {"answer": "See AUP.", "citations": [], "escalation": {"needed": False}},
    )

    teams_bot._answer("chat1", "Can I install software?")

    span = next(s for s in exporter.get_finished_spans() if s.name == "compliance_request")
    assert span.attributes["compliance_request.outcome"] == "answered"


# --- queue wait ----------------------------------------------------------------
#
# compliance_request opens AFTER the worker dequeues, so its duration measures
# processing and says nothing about what the user waited. With one worker, a burst
# puts the real cost in the queue: measured in production 2026-09-24, nine questions
# from three people left someone at depth 5 waiting ~35s while their span reported
# ~7s. _handle_inbound stamps time.monotonic() onto the queue tuple for this.


def test_queue_wait_is_recorded_on_the_root_span(monkeypatch, request_span_tracer, teams_bot):
    _tracer, exporter = request_span_tracer
    monkeypatch.setattr(bot_mod.settings, "router_enabled", True)
    monkeypatch.setattr(
        router_mod,
        "classify_message",
        lambda text: RouterDecision(category=Category.GREETING, confidence=0.95),
    )

    # 2.5s earlier on the monotonic clock: this message sat in the queue that long.
    queued_at = bot_mod.time.monotonic() - 2.5
    teams_bot._answer("chat1", "hello", queued_at=queued_at)

    span = next(s for s in exporter.get_finished_spans() if s.name == "compliance_request")
    wait_ms = span.attributes["compliance_request.queue_wait_ms"]
    assert 2400 <= wait_ms <= 2700, wait_ms


def test_queue_wait_defaults_to_zero_without_a_stamp(monkeypatch, request_span_tracer, teams_bot):
    """A direct call (tests, or any future non-queued path) must not crash or
    report a nonsense wait — the attribute is always present so it stays a usable
    filter in Phoenix."""
    _tracer, exporter = request_span_tracer
    monkeypatch.setattr(bot_mod.settings, "router_enabled", True)
    monkeypatch.setattr(
        router_mod,
        "classify_message",
        lambda text: RouterDecision(category=Category.GREETING, confidence=0.95),
    )

    teams_bot._answer("chat1", "hello")

    span = next(s for s in exporter.get_finished_spans() if s.name == "compliance_request")
    assert span.attributes["compliance_request.queue_wait_ms"] == 0


def test_the_worker_passes_the_enqueue_stamp_through(monkeypatch, teams_bot):
    """End to end through the real queue: _handle_inbound stamps, _worker_loop
    unpacks and forwards. Guards the tuple contract between the two threads —
    if either side stops agreeing on the shape, queue_wait_ms silently becomes 0."""
    seen = {}

    def fake_answer(chat_id, text, sender_name="Unknown", queued_at=None):
        seen["queued_at"] = queued_at
        return True

    monkeypatch.setattr(teams_bot, "_answer", fake_answer)
    monkeypatch.setattr(teams_bot, "_send_message", lambda *a, **k: True)

    before = bot_mod.time.monotonic()
    teams_bot._handle_inbound("chat1", "Can I install software?", "Ann", "chat1:m1", None)
    teams_bot._ensure_worker()
    teams_bot._work_q.join()

    assert seen["queued_at"] is not None, "the enqueue stamp never reached _answer"
    assert before <= seen["queued_at"] <= bot_mod.time.monotonic()
