"""Deterministic, fully-offline load/soak harness for the Teams poll loop.

Nothing in here may touch Microsoft Graph, the shared GPU host, the real
bot_state.json, or the rotating refresh-token file. The isolation rests on five
seams that the existing unit tests already use, all applied in build_sim():

  * ``bot.STATE_FILE``        -> a tmp_path file (never the real state file)
  * ``TeamsBot(token_refresher=object())`` -> TokenRefresher is never constructed,
    so ``channels/teams/data/refresh_token.json`` is never opened
  * ``bot_instance._api_request`` -> FakeGraph.api_request; ALL Graph traffic,
    reads and sends alike, funnels through this one method
  * ``bot._run_rag``          -> a stub, so the deferred ``rag.*`` imports inside it
    (llama-index / Ollama / Qdrant) never even execute
  * ``settings.router_enabled = False`` -> ``_answer`` never imports ``rag.router``
    and never makes a real LLM call

``run()`` is never called: it takes a PID lock on the real
``channels/teams/data/bot.pid`` and would collide with a running bot. The
harness drives ``process_new_messages()`` directly, exactly as ``run()``'s loop
body does (``_ensure_worker()`` then ``process_new_messages()``).

Determinism rules observed throughout:
  * every RNG is a seeded ``random.Random`` instance -- never the module-level one
  * no ``time.sleep`` anywhere; the clock is simulated and the only waits are
    bounded ``Event.wait`` / ``Queue.join`` calls that fail loudly on timeout
  * the worker never mutates FakeGraph while a poll cycle is running (see
    ``make_rag_stub``), so every GET the poll thread makes sees the same chat
    contents on every run
"""

import itertools
import random
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import channels.teams.bot as bot
from channels.teams.renderer import ACK_HTML

# The oid the bot decodes from its own access token. Messages from this sender
# are the bot's own posts and must be filtered as "self_message".
BOT_USER_ID = "bot-user-oid"


class NetworkForbidden(BaseException):
    """Raised by the autouse no_network fixture when real HTTP is attempted.

    Deliberately a BaseException, not an Exception: TeamsBot._api_request ends
    in a broad ``except Exception: ... return None``, which would quietly
    swallow this guard and turn a real Graph call into an ordinary "fetch
    failed" cycle. The point of the guard is to be impossible to ignore.
    """

# Answers are tagged so a reply can be traced back to the exact question that
# produced it -- that is what makes "answered exactly once" checkable.
ANSWER_PREFIX = "ANSWER::"

# --- fetch failure modes ------------------------------------------------------
#
# _api_request is the seam, so these are expressed as what IT returns, not as
# wire-level HTTP. Production collapses all three of the first modes onto a
# falsy return (4xx -> None, exhausted timeout retries -> None, falsy 200 body ->
# None); FALSY_BODY_200 deliberately returns an empty dict rather than None so
# the callers' ``if not data`` guards are exercised with a non-None falsy value
# too.
OK = "ok"
PERSISTENT_4XX = "persistent_4xx"
INTERMITTENT_TIMEOUT = "intermittent_timeout"
FALSY_BODY_200 = "falsy_body_200"
CYCLIC_NEXTLINK = "cyclic_nextlink"

_CHAT_ID_RE = re.compile(r"/me/chats/([^/?]+)/messages")
_TOP_RE = re.compile(r"[?&]\$top=(\d+)")
_SKIP_RE = re.compile(r"[?&]\$skiptoken=(\d+)")


# ------------------------------------------------------------------ clock ----

class FakeClock:
    """Simulated wall clock. Nothing here ever sleeps.

    The hold bound is teams_max_state_age_minutes (60), so the force-advance
    path is simply unreachable without simulating an hour of wall clock.
    """

    def __init__(self, start=None):
        # A Tuesday, 09:00 UTC -- inside the default business-hours window, so
        # _current_poll_interval reads as a normal working day if it is consulted.
        self._now = start or datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)
        self._lock = threading.Lock()

    def now(self):
        with self._lock:
            return self._now

    def advance(self, delta):
        with self._lock:
            self._now += delta


def install_clock(monkeypatch, clock):
    """Point bot.py's ``datetime`` at the simulated clock.

    bot.py does ``from datetime import datetime``, so the name to patch is
    ``bot.datetime``. Subclassing the real datetime keeps fromisoformat,
    isoformat, comparison and arithmetic behaving exactly as production's do --
    only ``now()`` is simulated.
    """
    class _SimulatedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now()

    monkeypatch.setattr(bot, "datetime", _SimulatedDatetime)
    return _SimulatedDatetime


# ------------------------------------------------------------- fake Graph ----

@dataclass
class Send:
    seq: int
    chat_id: str
    html: str


class GraphCallCeiling(AssertionError):
    """Raised instead of letting a pathological pagination loop hang the suite."""


class FakeGraph:
    """N chats, each with its own message list, shaped like the real Graph API.

    Reproduces the parts of the API's shape that the code under test actually
    navigates: ``$top`` on the per-chat message fetch, ``@odata.nextLink`` paging
    on both the chat list and per-chat messages, newest-first message ordering,
    and POSTed sends appended to the chat (so the bot's own acks and replies come
    back on the next fetch and must be filtered as self_messages).
    """

    def __init__(self, clock, *, n_chats, chats_per_page=12, seed=20260916,
                 call_ceiling=200_000):
        self.clock = clock
        self.chat_ids = [f"chat{i:02d}" for i in range(n_chats)]
        self.messages = {cid: [] for cid in self.chat_ids}  # oldest-first
        self.chats_per_page = chats_per_page
        self.sends: list[Send] = []
        self.failures: dict[str, tuple[str, float]] = {}
        self.chat_list_mode = OK
        self.rng = random.Random(seed)
        self.calls = 0
        self.chat_list_pages = 0
        self.message_pages = 0
        self.call_ceiling = call_ceiling
        self._lock = threading.RLock()
        self._seq = itertools.count()
        self._msg_seq = itertools.count()

    # --- instrumentation -------------------------------------------------

    def set_failure(self, chat_id, mode, rate=1.0):
        self.failures[chat_id] = (mode, rate)

    def set_chat_list_failure(self, mode):
        self.chat_list_mode = mode

    def add_user_message(self, chat_id, token, text=None):
        """Inject a user message. ``token`` is the identity carried into the answer."""
        body = text or f"{token} what does the policy say about this"
        return self._append(chat_id, body, sender_name="Ann",
                            sender_id=f"user-{chat_id}", token=token)

    def user_message_tokens(self):
        return [m["_token"] for msgs in self.messages.values() for m in msgs if m.get("_token")]

    # --- internals -------------------------------------------------------

    def _append(self, chat_id, body, *, sender_name, sender_id, token=None):
        with self._lock:
            n = next(self._msg_seq)
            # Microsecond offsets keep every message strictly ordered and unique
            # even though the simulated clock only ticks between cycles.
            stamp = (self.clock.now() + timedelta(microseconds=n)).isoformat().replace("+00:00", "Z")
            message = {
                "id": f"m{n:07d}",
                "messageType": "message",
                "from": {"user": {"id": sender_id, "displayName": sender_name}},
                "createdDateTime": stamp,
                "body": {"contentType": "html", "content": body},
                "_token": token,
            }
            self.messages[chat_id].append(message)
            return message["id"]

    def _fail(self, mode):
        if mode == FALSY_BODY_200:
            return {}          # a 200 with a falsy body
        return None            # a persistent 4xx, or a timeout that ran out of retries

    def _chat_failure(self, chat_id):
        mode, rate = self.failures.get(chat_id, (OK, 0.0))
        if mode == INTERMITTENT_TIMEOUT:
            # Consulted only for chats configured as intermittent, and only from
            # the poll thread, so the draw sequence is identical on every run.
            return mode if self.rng.random() < rate else OK
        return mode

    # --- the seam: a drop-in for TeamsBot._api_request --------------------

    def api_request(self, url, method="GET", json_data=None, retry=False):
        with self._lock:
            self.calls += 1
            if self.calls > self.call_ceiling:
                raise GraphCallCeiling(
                    f"FakeGraph exceeded {self.call_ceiling} calls -- a pagination "
                    "loop is not terminating; failing fast instead of hanging"
                )
        if method == "POST":
            return self._post(url, json_data)
        if method != "GET":
            return None
        if "/messages" in url:
            return self._get_messages(url)
        return self._get_chat_list(url)

    def _post(self, url, json_data):
        match = _CHAT_ID_RE.search(url)
        if not match:
            return None
        chat_id = match.group(1)
        html = ((json_data or {}).get("body") or {}).get("content", "")
        with self._lock:
            self.sends.append(Send(seq=next(self._seq), chat_id=chat_id, html=html))
            mid = self._append(chat_id, html, sender_name="Compliance Bot",
                               sender_id=BOT_USER_ID)
        return {"id": mid}

    def _get_chat_list(self, url):
        self.chat_list_pages += 1
        if self.chat_list_mode == CYCLIC_NEXTLINK:
            # A self-referential nextLink: the same URL the caller just fetched.
            return {"value": [{"id": cid} for cid in self.chat_ids[:2]],
                    "@odata.nextLink": url}
        if self.chat_list_mode != OK:
            return self._fail(self.chat_list_mode)

        top_match = _TOP_RE.search(url)
        top = int(top_match.group(1)) if top_match else 20
        page = min(top, self.chats_per_page)
        skip_match = _SKIP_RE.search(url)
        offset = int(skip_match.group(1)) if skip_match else 0

        window = self.chat_ids[offset:offset + page]
        data = {"value": [{"id": cid} for cid in window]}
        if offset + page < len(self.chat_ids):
            data["@odata.nextLink"] = (
                f"{bot.GRAPH_API}/me/chats?$top={top}&$skiptoken={offset + page}"
            )
        return data

    def _get_messages(self, url):
        match = _CHAT_ID_RE.search(url)
        if not match:
            return None
        chat_id = match.group(1)
        if chat_id not in self.messages:
            return None

        self.message_pages += 1
        mode = self._chat_failure(chat_id)
        if mode == CYCLIC_NEXTLINK:
            with self._lock:
                newest = list(reversed(self.messages[chat_id]))[:1]
            return {"value": [dict(m) for m in newest], "@odata.nextLink": url}
        if mode != OK:
            return self._fail(mode)

        top_match = _TOP_RE.search(url)
        top = int(top_match.group(1)) if top_match else 20
        skip_match = _SKIP_RE.search(url)
        offset = int(skip_match.group(1)) if skip_match else 0

        with self._lock:
            newest_first = list(reversed(self.messages[chat_id]))
        window = newest_first[offset:offset + top]
        data = {"value": [dict(m) for m in window]}
        if offset + top < len(newest_first):
            data["@odata.nextLink"] = (
                f"{bot.GRAPH_API}/me/chats/{chat_id}/messages"
                f"?$top={top}&$skiptoken={offset + top}"
            )
        return data


# ------------------------------------------------------------- RAG stub ------

def make_rag_stub(gate, record, errors):
    """A stand-in for the ~16s RAG pipeline.

    It waits on ``gate``, which the driver opens only after
    ``process_new_messages()`` has returned. That models production timing (a
    poll cycle finishes long before a 16s pipeline does) and buys two things the
    harness depends on: the ack, sent inline by the poll thread, always precedes
    the reply; and the worker never appends to FakeGraph while the poll thread is
    reading it, which is what makes every run byte-identical. It is a bounded
    wait on an Event, not a sleep -- a gate that never opens fails the test.
    """
    def _rag(question):
        if not gate.wait(10.0):
            errors.append("cycle gate never opened; the worker was left parked")
            raise AssertionError("cycle gate never opened")
        token = question.split()[0]
        record.append(token)
        # A citation is required now: bot.py's grounding backstop (Task 6) escalates
        # any answer with no citations instead of rendering it, so an uncited stub
        # would never reach the html the assertions below scan for ANSWER_PREFIX.
        # doc_title (not quote) carries the token: render_answer emits it as
        # "<b>...{doc_title}</b>" with no character between the token and the "<",
        # which is what answered_tokens()'s ANSWER_PREFIX(\S+?)</ regex requires.
        return {
            "answer": f"{ANSWER_PREFIX}{token}",
            "citations": [{"doc_title": f"{ANSWER_PREFIX}{token}", "quote": "load-test citation"}],
            "escalation": {"needed": False},
        }
    return _rag


# ------------------------------------------------------------ simulation -----

@dataclass
class Simulation:
    """Drives process_new_messages() cycle by cycle and records the invariants."""

    # repr=False throughout: a dataclass repr lands in pytest's assertion output,
    # and a 130-cycle run's recorded series would bury the actual failure.
    tbot: object = field(repr=False)
    graph: FakeGraph = field(repr=False)
    clock: FakeClock = field(repr=False)
    gate: threading.Event = field(repr=False)
    state_file: object = field(repr=False)
    rag_calls: list = field(repr=False)
    errors: list = field(repr=False)
    start_worker: bool = True
    cycles: int = 0
    watermarks: list = field(default_factory=list, repr=False)
    processed_sizes: list = field(default_factory=list, repr=False)
    state_sizes: list = field(default_factory=list, repr=False)
    queue_depths: list = field(default_factory=list, repr=False)
    worker_idents: set = field(default_factory=set, repr=False)
    baseline_workers: frozenset = field(default=frozenset(), repr=False)

    def __post_init__(self):
        # Scope the thread-count invariant to workers THIS simulation created:
        # a previous test's bot leaves its own idle worker parked on an empty
        # queue, and that must not be mistaken for a second consumer here.
        self.baseline_workers = frozenset(
            t for t in threading.enumerate() if t.name == "rag-worker"
        )

    # --- driving ---------------------------------------------------------

    def run_cycle(self, advance=timedelta(seconds=30), drain=True, inject=None):
        self.clock.advance(advance)
        if inject:
            inject(self.cycles)
        self.gate.clear()
        if self.start_worker:
            self.tbot._ensure_worker()      # exactly what run()'s loop body does
            self._assert_one_worker()
        try:
            self.tbot.process_new_messages()
        except Exception as exc:            # nothing may escape the poll loop
            self.errors.append(f"cycle {self.cycles}: {type(exc).__name__}: {exc}")
        self.queue_depths.append(self.tbot._work_q.qsize())
        self.watermarks.append(self.tbot.last_check)
        self.gate.set()
        if drain and self.start_worker:
            self.drain()
        self.processed_sizes.append(len(self.tbot.processed_messages))
        if self.state_file.exists():
            self.state_sizes.append(self.state_file.stat().st_size)
        self.cycles += 1

    def run_cycles(self, n, **kwargs):
        for _ in range(n):
            self.run_cycle(**kwargs)

    def drain(self, timeout=15):
        """Bounded wait for the worker to finish the queue.

        queue.Queue.join() takes no timeout, so a worker that dies before
        task_done() would hang the whole suite. Wait on a helper thread instead.
        """
        joiner = threading.Thread(target=self.tbot._work_q.join, daemon=True)
        joiner.start()
        joiner.join(timeout)
        assert not joiner.is_alive(), (
            f"work queue did not drain within {timeout}s at cycle {self.cycles} "
            f"(depth {self.tbot._work_q.qsize()}, in-flight {len(self.tbot._inflight)})"
        )

    # --- invariants ------------------------------------------------------

    def _live_workers(self):
        return [t for t in threading.enumerate()
                if t.name == "rag-worker" and t not in self.baseline_workers]

    def live_worker_count(self):
        """How many rag-worker threads this simulation is responsible for, right now."""
        return len(self._live_workers())

    def _assert_one_worker(self):
        live = self._live_workers()
        assert len(live) == 1, (
            f"expected exactly one rag-worker thread, found {len(live)} at cycle {self.cycles}"
        )
        self.worker_idents.add(live[0].ident)

    def acks(self, chat_id):
        return [s.seq for s in self.graph.sends if s.chat_id == chat_id and s.html == ACK_HTML]

    def replies(self, chat_id):
        return [s.seq for s in self.graph.sends
                if s.chat_id == chat_id and ANSWER_PREFIX in s.html]

    def answered_tokens(self):
        """token -> how many replies carried it."""
        counts = {}
        for send in self.graph.sends:
            for token in re.findall(rf"{ANSWER_PREFIX}(\S+?)</", send.html):
                counts[token] = counts.get(token, 0) + 1
        return counts

    def assert_no_duplicate_answers(self):
        dupes = {t: n for t, n in self.answered_tokens().items() if n > 1}
        assert not dupes, f"messages answered more than once: {sorted(dupes.items())[:10]}"

    def assert_all_answered(self, tokens):
        answered = self.answered_tokens()
        missing = [t for t in tokens if t not in answered]
        assert not missing, (
            f"{len(missing)} accepted message(s) never got a reply, e.g. {missing[:10]}"
        )

    def assert_ack_precedes_reply(self):
        for chat_id in self.graph.chat_ids:
            acks, replies = self.acks(chat_id), self.replies(chat_id)
            assert len(acks) >= len(replies), (
                f"{chat_id}: {len(replies)} replies but only {len(acks)} acks"
            )
            for i, (ack_seq, reply_seq) in enumerate(zip(acks, replies)):
                assert ack_seq < reply_seq, (
                    f"{chat_id}: reply #{i} was sent before its ack "
                    f"(ack seq {ack_seq}, reply seq {reply_seq})"
                )

    def assert_watermark_never_regresses(self):
        for i in range(1, len(self.watermarks)):
            assert self.watermarks[i] >= self.watermarks[i - 1], (
                f"last_check moved backwards at cycle {i}: "
                f"{self.watermarks[i - 1].isoformat()} -> {self.watermarks[i].isoformat()}"
            )

    def assert_no_poll_loop_exceptions(self):
        assert not self.errors, f"exception(s) escaped the poll loop: {self.errors[:5]}"

    def assert_one_worker_ever(self):
        """Exactly one worker thread for this bot, ever -- checked on live threads.

        _assert_one_worker() has already asserted the live rag-worker count was
        exactly 1 at the top of every cycle; this pins down that it was always
        the SAME thread and that the bot still owns it. Scoped to this bot, not
        to the process: the restart scenario deliberately has more than one
        TeamsBot alive, and each must own exactly one consumer of its own queue.
        """
        assert len(self.worker_idents) == 1, (
            f"more than one worker thread was ever alive: {self.worker_idents}"
        )
        worker = self.tbot._worker
        assert worker is not None and worker.is_alive(), "the bot's worker is not running"
        assert worker.ident in self.worker_idents, "the bot swapped its worker thread"

    def assert_bounded(self, *, max_processed, max_state_bytes):
        peak = max(self.processed_sizes)
        assert peak <= max_processed, (
            f"processed_messages peaked at {peak}, above the {max_processed} bound"
        )
        peak_bytes = max(self.state_sizes)
        assert peak_bytes <= max_state_bytes, (
            f"state file peaked at {peak_bytes} bytes, above the {max_state_bytes} bound"
        )

    def assert_core_invariants(self, tokens, *, max_processed, max_state_bytes):
        self.assert_no_poll_loop_exceptions()
        self.assert_no_duplicate_answers()
        self.assert_all_answered(tokens)
        self.assert_ack_precedes_reply()
        self.assert_watermark_never_regresses()
        self.assert_one_worker_ever()
        self.assert_bounded(max_processed=max_processed, max_state_bytes=max_state_bytes)
