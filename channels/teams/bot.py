"""Teams bot — polls Microsoft Graph API for new messages, runs RAG pipeline directly."""

import base64
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from config import settings
from channels.teams.utils import safe_get_nested, strip_html
from channels.teams.renderer import (
    ACK_HTML,
    RATING_PROMPT_HTML,
    RATING_THANKS_HTML,
    WELCOME_HTML,
    render_answer,
    render_escalation,
    render_error,
    render_out_of_scope,
    render_unavailable,
    render_unintelligible,
)
from channels.teams.feedback import save_feedback

GRAPH_API = "https://graph.microsoft.com/v1.0"
STATE_FILE = Path("channels/teams/data/bot_state.json")
PID_FILE = Path("channels/teams/data/bot.pid")

# Graph pages /me/chats at 20 per page by default; 50 is the documented maximum.
# Fewer pages per cycle — and _get_all_pages follows @odata.nextLink for the rest.
_CHATS_PAGE_SIZE = 50
# Hard caps on backward/forward pagination. Both guard against the same class of
# failure: a page fetch that never terminates (a cyclic or self-referential
# @odata.nextLink) would spin the poll thread forever without ever raising, so
# consecutive_errors never trips and the bot goes silently dead while still
# "running". Bounding every pagination loop means the worst case is a loud log
# line and an incomplete cycle, never a hang.
_CHATS_MAX_PAGES = 20      # /me/chats: bounds a cyclic/self-referential nextLink
_MESSAGES_MAX_PAGES = 10   # per-chat messages: bounds how far back a burst pages

# Branch review Fix D: this reach constant sets a ceiling on outage recovery,
# not just burst handling. Once an outage runs longer than roughly
# _MESSAGES_MAX_PAGES x teams_messages_page_size messages of traffic in a
# chat, that chat can no longer page back far enough to reach the frozen
# last_check once it recovers — so it stays "incomplete" AFTER the underlying
# failure has cleared, and the hold runs all the way to Ruling G's
# force-advance (teams_max_state_age_minutes) regardless. That makes a
# full-length hold the NORMAL outcome of a long outage, not a worst case —
# which is what makes the resident-size growth in Fix B/C's comments
# (measured: 1723 ids / 136 KB for one full hold at 30 users) an expected
# operating point, not an anomaly to chase.

# Branch review Fix C: _load_state logs (never truncates) above this multiple
# of teams_max_processed_messages — see _load_state for why truncation itself
# would be the bug.
_ABNORMAL_PROCESSED_MESSAGES_MULTIPLIER = 10

_VALID_RATINGS = {"-1", "0", "1", "2"}

# Bounded retry for transient Graph failures (timeouts, connection errors, 5xx).
# Deliberately short and deliberately local: rag.resilience is for the model/vector
# backends, its _TRANSIENT_TYPES does not cover requests' exceptions, and importing it
# here would pull qdrant_client/openai into this module's header (observability-first).
# The poll thread's ack goes through the same call, so this must not become a stall.
_GRAPH_RETRY_BACKOFFS: tuple[float, ...] = (0.5, 1.0)

# Pending ratings: chat_id → context dict
# Lost on restart — acceptable
_pending_ratings: dict[str, dict] = {}


def _message_key(chat_id, message_id):
    """Identify a message by chat AND id.

    Graph documents chatMessage.id as unique within its containing chat, not
    globally — in practice they are millisecond epochs, so two people posting in
    different chats in the same millisecond collide and the second would be
    silently skipped as already-processed. A plain string, not a tuple, because
    processed_messages is persisted as a JSON list and tuples do not round-trip.
    """
    return f"{chat_id}:{message_id}"


def _run_rag(question: str) -> dict:
    """Run the RAG pipeline directly (no HTTP). Returns a result dict.

    Outcomes:
      - {"status": "unavailable"}            transient backend failure (retried)
      - {"answer", "citations", "escalation"} normal ComplianceAnswer
    """
    # Deferred imports: init_observability() (start_teams_bot.py) must run before LlamaIndex loads.
    import asyncio
    import rag.tools.search_policies as sp
    from rag.agent import build_agent
    from rag.response import parse_agent_response
    from rag.resilience import retry_transient, is_transient, RETRY_BACKOFFS
    from rag.observability import record_infra_unavailable

    sp._retrieval_unavailable = False

    async def _run():
        agent = build_agent()
        return await agent.run(user_msg=question)

    try:
        response = retry_transient(lambda: asyncio.run(_run()))
    except Exception as e:
        if is_transient(e):
            record_infra_unavailable("llm", type(e).__name__, len(RETRY_BACKOFFS))
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            # +1: retry_transient makes len(backoffs)+1 total calls (its own docstring) —
            # the retries_attempted passed above is deliberately the smaller retry count,
            # not the call count. Different numbers on purpose; don't "fix" them to match.
            print(
                f"[worker] Unavailable (llm): {type(e).__name__}: {msg}; "
                f"gave up after {len(RETRY_BACKOFFS) + 1} attempts"
            )
            return {"status": "unavailable"}
        print(f"RAG pipeline error: {e}")
        return {
            "answer": "",
            "citations": [],
            "escalation": {"needed": True, "reason": str(e)},
        }

    # Retrieval failed inside the tool (LlamaIndex swallows tool exceptions) →
    # the flag was set in search_policies; surface the unavailable outcome.
    # search_policies itself does not print/log — it only emits the Phoenix
    # infra_unavailable span — so this is the only container-log record that
    # retrieval (not the LLM) was the failing component.
    if sp._retrieval_unavailable:
        print("[worker] Unavailable (retrieval): search_policies flagged the backend unavailable (embeddings/qdrant)")
        return {"status": "unavailable"}

    return parse_agent_response(str(response))


class TeamsBot:
    def __init__(self, token_refresher):
        self.token_refresher = token_refresher
        self._my_user_id = None
        state = self._load_state()
        self.last_check = state["last_check"]
        self.processed_messages = state["processed_messages"]
        # Wall-clock time the current last_check hold began; None when not held.
        # process_new_messages sets this the first cycle any fetch comes back
        # incomplete, and clears it the moment last_check next advances (normally
        # or by the force-advance below). Mirrors _load_state's startup staleness
        # clamp, but enforced continuously at runtime: an unbounded hold lets
        # _cleanup_processed_messages evict ids the frozen watermark still needs,
        # and Ruling B's own back-paging then re-fetches and re-answers them — see
        # process_new_messages for the bounded force-advance that prevents it.
        self._hold_since: datetime | None = None
        # EXACTLY ONE worker consumes this queue. Do not raise the worker count.
        # rag/tools/search_policies.py keeps _retrieval_unavailable and
        # _last_search_results as module globals, reset before an agent run and
        # read ~16s later. A second worker interleaves those resets and turns a
        # transient infra failure into a false content escalation, silently.
        # Fix those globals (ToolCallResult.tool_output is per-request) before
        # ever running more than one.
        self._work_q: "queue.Queue[tuple[str, str, str, str]]" = queue.Queue()
        self._worker: threading.Thread | None = None  # started by _ensure_worker() in run()
        # message key -> createdDateTime, for messages accepted but not yet answered.
        self._inflight: dict[str, datetime] = {}
        self._inflight_lock = threading.Lock()
        # Guards _ensure_worker: self._worker is assigned before start(), so two
        # concurrent callers would each see a not-alive worker and start their own.
        # Only the poll thread calls it today — this keeps the one-worker invariant
        # true in code rather than by convention.
        self._worker_lock = threading.Lock()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        now = datetime.now(timezone.utc)
        fresh_check = now - timedelta(minutes=settings.teams_initial_lookback_minutes)
        default = {
            "last_check": fresh_check,
            "processed_messages": {},
        }
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            last_check = datetime.fromisoformat(data["last_check"])
            # dict, not set: insertion order is what makes eviction genuinely oldest-first.
            processed = dict.fromkeys(data.get("processed_messages", []))
            # Branch review Fix C: visibility only, never a cap here. _load_state
            # used to inherit a size bound for free, because the writer
            # (_save_state) capped the file on every save — it no longer does
            # (Fix 2 removed that slice; _cleanup_processed_messages, gated on
            # fully_synced, is the only eviction site left). Truncating on load
            # would drop ids that are still newer than last_check — exactly the
            # duplicate-answer bug this branch spent four rounds eliminating —
            # so this only logs. A full-length hold is the NORMAL outcome of a
            # long outage (see Fix D, next to _MESSAGES_MAX_PAGES), so treat the
            # warning below as "go look", not "something is broken".
            print(f"Loaded {len(processed)} processed message id(s) from {STATE_FILE}")
            abnormal_threshold = _ABNORMAL_PROCESSED_MESSAGES_MULTIPLIER * settings.teams_max_processed_messages
            if len(processed) > abnormal_threshold:
                print(
                    f"WARNING: {len(processed)} processed message ids loaded from "
                    f"{STATE_FILE} — more than {_ABNORMAL_PROCESSED_MESSAGES_MULTIPLIER}x "
                    f"teams_max_processed_messages ({settings.teams_max_processed_messages}); "
                    "worth checking for a chat stuck in a long hold. Visibility only — "
                    "nothing here truncates it."
                )
            # Clamp a stale last_check so a long-stopped bot can't treat the whole
            # backlog as new and answer it all into the channel. Normal restarts
            # (downtime < teams_max_state_age_minutes) still resume from last_check.
            if now - last_check > timedelta(minutes=settings.teams_max_state_age_minutes):
                print(
                    f"WARNING: bot_state last_check {last_check.isoformat()} is older than "
                    f"{settings.teams_max_state_age_minutes} min; clamping to {fresh_check.isoformat()} "
                    "to avoid answering the backlog."
                )
                last_check = fresh_check
            return {"last_check": last_check, "processed_messages": processed}
        except (FileNotFoundError, KeyError, ValueError):
            return default

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with self._inflight_lock:
            pending_times = list(self._inflight.values())
            pending_ids = set(self._inflight)

        # Persist a watermark BEHIND anything still queued, and keep queued ids
        # out of the processed set, so a restart re-delivers unanswered work
        # instead of skipping it. In-memory last_check still advances, so the
        # running process never re-enqueues what it already holds.
        # Bounded by the staleness clamp in _load_state: this re-delivery guarantee
        # only holds for a restart within teams_max_state_age_minutes. After a longer
        # outage the rewound watermark is discarded by the clamp, and these ids were
        # deliberately left out of processed_messages, so in-flight questions are lost
        # rather than re-delivered. That is the anti-backlog-flood trade, not a bug.
        #
        # Branch review Fix 1: must never persist AHEAD of self.last_check. In the
        # healthy case in-flight times are always <= last_check (a message is only
        # ever in-flight because it was just accepted as newer-than-last_check, and
        # last_check itself only ever advances to cover it in the same cycle), so
        # min(pending) - 1ms already sits behind last_check and this floor changes
        # nothing there — see test_saved_watermark_is_held_before_the_oldest_inflight_message
        # and test_saved_watermark_is_last_check_when_nothing_inflight, unchanged, for that.
        # But while last_check is HELD (frozen behind an unreadable chat), a healthy
        # chat can keep producing in-flight messages far NEWER than the freeze point;
        # without the floor, min(pending) - 1ms then runs ahead of the hold — reproduced
        # as a 28-minute jump — and a restart loads that jumped watermark and marks
        # everything the hold was protecting as old_message, permanently. self._hold_since
        # is not itself persisted, but that is fine: _load_state's own staleness clamp
        # (same teams_max_state_age_minutes) bounds a crash-looping restart the same way.
        watermark = self.last_check
        if pending_times:
            watermark = min(watermark, min(pending_times) - timedelta(milliseconds=1))

        # Branch review Fix 2: no slice here. _cleanup_processed_messages (gated on
        # fully_synced in process_new_messages) is the ONLY place eviction happens —
        # this used to apply its own identical [-teams_max_processed_messages:] slice
        # on every save, held or not, silently dropping ids while the in-memory
        # cleanup was correctly blocked. Persist the set as-is; only in-flight ids
        # are still held back, same as always.
        ids = [mid for mid in self.processed_messages if mid not in pending_ids]
        # Atomic: this file is the sole carrier of the crash-recovery guarantee, and
        # a plain open("w") truncates first — a SIGKILL mid-dump would leave truncated
        # JSON, _load_state would fall back to its default, and every in-flight question
        # would be lost. Write beside it and rename over it (atomic on POSIX; the temp
        # file must be in the same directory, since os.replace across filesystems is not).
        tmp_file = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
        with open(tmp_file, "w") as f:
            json.dump(
                {
                    "last_check": watermark.isoformat(),
                    "processed_messages": ids,
                },
                f,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, STATE_FILE)

    # ------------------------------------------------------------------
    # PID lock
    # ------------------------------------------------------------------

    @staticmethod
    def _pid_is_running(pid):
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _acquire_pid_lock(self):
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        my_pid = os.getpid()
        if PID_FILE.exists():
            try:
                existing_pid = int(PID_FILE.read_text().strip())
                # In Docker, both old and new containers run as PID 1 — the file
                # is stale across restarts. Same-PID always means stale.
                if existing_pid != my_pid and self._pid_is_running(existing_pid):
                    print(f"Another bot instance is already running (PID {existing_pid}). Exiting.")
                    sys.exit(1)
            except ValueError:
                pass
        PID_FILE.write_text(str(my_pid))

    def _release_pid_lock(self):
        try:
            PID_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Graph API helpers
    # ------------------------------------------------------------------

    def _get_headers(self):
        token = self.token_refresher.get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _api_request(self, url, method="GET", json_data=None, retry=False):
        """Call Graph. With retry=True, transient failures get a bounded retry.

        `retry` is OPT-IN, and deliberately off by default. Retrying costs up to
        ~31s of wall clock, and the poll thread must never block that long — that is
        the whole point of the worker queue. Its calls (chat list, per-chat messages,
        the ack) are self-healing anyway: the loop comes back in teams_poll_interval
        seconds and re-reads the same chats, so a transient miss there costs nothing.
        Turn it on ONLY where a lost call cannot be recovered by the next poll — the
        worker's reply in _answer, where ~16s of GPU work is already spent and a
        failed send leaves the user with an acknowledgement and nothing else.

        Retries timeouts, connection errors and 5xx — a Graph blip that clears. Never
        retries 4xx: a deleted chat or a bad payload will not come back, and retrying
        it only burns a poll cycle. Returns None once the attempts are exhausted.
        """
        backoffs = _GRAPH_RETRY_BACKOFFS if retry else ()
        for attempt in range(len(backoffs) + 1):
            try:
                if method == "GET":
                    response = requests.get(url, headers=self._get_headers(), timeout=settings.teams_api_timeout)
                elif method == "POST":
                    response = requests.post(url, json=json_data, headers=self._get_headers(), timeout=settings.teams_api_timeout)
                else:
                    return None

                response.raise_for_status()

                if method == "POST":
                    try:
                        return response.json() or True
                    except Exception:
                        return True
                data = response.json()
                return data if data else None

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                reason = f"{type(e).__name__} for {url}"
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status is None or status < 500:
                    print(f"HTTP error for {url}: {e}")
                    return None
                reason = f"HTTP {status} for {url}"
            except Exception as e:
                print(f"Request failed for {url}: {e}")
                return None

            if attempt < len(backoffs):
                delay = backoffs[attempt]
                print(f"Transient Graph failure ({reason}); retrying in {delay}s")
                time.sleep(delay)
            elif backoffs:
                print(f"Transient Graph failure ({reason}); gave up after "
                      f"{len(backoffs) + 1} attempts")
            else:
                print(f"Transient Graph failure ({reason}); not retried")
        return None

    def _send_message(self, chat_id, text, content_type="html", retry=False):
        """Send to a chat. See _api_request for when `retry` may be turned on."""
        url = f"{GRAPH_API}/me/chats/{chat_id}/messages"
        payload = {"body": {"contentType": content_type, "content": text}}
        return self._api_request(url, method="POST", json_data=payload, retry=retry)

    def _get_all_pages(self, url):
        """GET a Graph collection, following @odata.nextLink until exhausted.

        Graph pages every collection. Reading only the first page of /me/chats
        silently stops polling chats past the first 20 — invisible at 5 users,
        a dropped-user bug at 30.

        Returns (items, complete). `complete` is False when a page fetch failed,
        the page cap (_CHATS_MAX_PAGES) was hit, or a @odata.nextLink was revisited
        (a cyclic or self-referential link would otherwise spin this loop, and
        therefore the poll thread, forever). Running out of @odata.nextLink is the
        only normal, complete termination.

        The caller must not treat an incomplete result as the full chat list: see
        process_new_messages, which holds last_check when this returns incomplete
        so chats behind the gap are retried next cycle instead of being marked
        "already seen" and lost for good.
        """
        items = []
        seen_urls = set()
        pages = 0
        while url:
            if pages >= _CHATS_MAX_PAGES:
                print(
                    f"WARNING: /me/chats pagination hit the {_CHATS_MAX_PAGES}-page cap; "
                    "treating the chat list as incomplete this cycle"
                )
                return items, False
            if url in seen_urls:
                print(
                    "WARNING: @odata.nextLink on /me/chats repeated a URL already "
                    "fetched this cycle (cyclic or self-referential link?); stopping "
                    "pagination and treating the chat list as incomplete this cycle"
                )
                return items, False
            seen_urls.add(url)
            pages += 1
            data = self._api_request(url)
            if not data:
                return items, False
            items.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        return items, True

    def _get_chat_messages(self, chat_id):
        """Fetch one chat's new messages, paging backward until the window covers last_check.

        A fixed $top page only returns the newest N messages. That is enough in
        the steady state — a quiet chat's newest teams_messages_page_size messages
        already reach back past last_check — but two things make a burst of more
        than N messages in one chat ordinary rather than exotic: the idle interval
        is up to teams_idle_poll_interval seconds, and the bot itself posts 3
        replies (ack, answer, rating prompt) into the same chat per question. A
        fixed small page then silently drops the oldest messages in the burst, and
        also breaks restart re-delivery: a message correctly held in-flight and
        rewound in the persisted watermark becomes unfetchable once enough newer
        messages push it off the single page.

        So: follow @odata.nextLink, accumulating pages, until the oldest message
        retrieved so far is at or before self.last_check — at that point the
        window provably covers everything the watermark claims is unprocessed.
        Capped at _MESSAGES_MAX_PAGES; a short chat history that runs out of
        @odata.nextLink first is a normal, complete result, not a failure.

        Returns (messages, complete). `complete` is False when the page cap was
        hit, or a page fetch failed, before the window reached last_check — the
        caller must not let this chat's absence of older messages advance the
        global last_check watermark this cycle (same reasoning as _get_all_pages).
        """
        messages = []
        oldest_seen = None
        url = (
            f"{GRAPH_API}/me/chats/{chat_id}/messages"
            f"?$top={settings.teams_messages_page_size}"
        )
        for _ in range(_MESSAGES_MAX_PAGES):
            data = self._api_request(url)
            if not data:
                print(
                    f"WARNING: message fetch failed for chat {chat_id}; treating this "
                    "chat as incomplete this cycle so the watermark is not advanced "
                    "past unread messages"
                )
                return messages, False
            page = data.get("value", []) or []
            messages.extend(page)
            for message in page:
                stamp = (message or {}).get("createdDateTime")
                if not stamp:
                    continue
                try:
                    created = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if oldest_seen is None or created < oldest_seen:
                    oldest_seen = created
            if oldest_seen is not None and oldest_seen <= self.last_check:
                return messages, True
            url = data.get("@odata.nextLink")
            if not url:
                return messages, True  # short history; ran out of pages normally
        print(
            f"WARNING: message paging for chat {chat_id} hit the {_MESSAGES_MAX_PAGES}-page "
            f"cap without reaching last_check ({self.last_check.isoformat()}); treating this "
            "chat as incomplete this cycle so the watermark is not advanced past unread messages"
        )
        return messages, False

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _handle_inbound(self, chat_id, message_text, sender_name="Unknown",
                        message_key=None, created_time=None):
        """Poll thread: answer cheap cases inline, acknowledge and enqueue the rest.

        Nothing here may make an LLM call — the poll loop must keep polling.
        """
        if not message_text or not message_text.strip():
            return False

        text = message_text.strip()

        if text.lower() in ("start", "/start", "help", "/help"):
            _pending_ratings.pop(chat_id, None)
            self._send_message(chat_id, WELCOME_HTML)
            return True

        if text == "[media/emoji]":
            return True

        # Check if this is a rating for a pending answer
        if chat_id in _pending_ratings and text in _VALID_RATINGS:
            ctx = _pending_ratings.pop(chat_id)
            save_feedback(
                question=ctx["question"],
                answer=ctx["answer"],
                citations=ctx["citations"],
                rating=int(text),
                user=ctx["user"],
                chat_id=chat_id,
            )
            self._send_message(chat_id, RATING_THANKS_HTML)
            print(f"Feedback saved: rating={text} for '{ctx['question'][:50]}...'")
            return True

        # Not a rating — clear any pending state and hand off as a question.
        _pending_ratings.pop(chat_id, None)

        if message_key and created_time:
            with self._inflight_lock:
                self._inflight[message_key] = created_time
        else:
            # Untracked: the watermark advances past this message, so a crash before
            # it is answered drops it silently. Unreachable from process_new_messages
            # (_should_process_message rejects messages with no id or no parseable
            # timestamp) — log it rather than let a future caller lose work quietly.
            print(
                f"WARNING: enqueuing untracked message in {chat_id} "
                f"(message_key={message_key!r}, created_time={created_time!r}); "
                "a crash before it is answered will drop it"
            )

        # Enqueue first: a slow ack POST must not delay the work it acknowledges.
        # In-flight registration stays above both, so a crash anywhere here re-delivers.
        self._work_q.put((chat_id, text, sender_name, message_key or ""))
        # Depth right after enqueueing, so it shows how deep this sender landed in the
        # backlog. qsize() is approximate under concurrency — fine for a log line, not
        # worth a lock. Kept immediately adjacent to process_new_messages' "New message
        # from ..." print: the gap between those two lines is the ack latency.
        depth = self._work_q.qsize()
        acked = self._send_message(chat_id, ACK_HTML)
        if acked:
            print(f"Ack sent to chat {chat_id} (queue depth {depth})")
        else:
            # User-visible degradation: they'll get an answer out of nowhere with no
            # acknowledgement. The question itself is not lost — it was enqueued above.
            print(
                f"ERROR: ack not delivered to chat {chat_id} (queue depth {depth}); "
                "question is still queued and will be answered without ever being acknowledged"
            )
        return True

    def _worker_loop(self):
        """The single consumer. See the __init__ comment before adding a second."""
        while True:
            chat_id, text, sender_name, message_key = self._work_q.get()
            try:
                if not self._answer(chat_id, text, sender_name=sender_name):
                    # The answer was produced but the reply POST failed even after
                    # retries. Nothing re-sends it — the user has an ack and then
                    # silence — so say so loudly instead of dropping it quietly.
                    print(
                        f"[worker] ERROR: answer not delivered to chat {chat_id} "
                        f"(message={message_key or 'unknown'}, question={text[:80]!r})"
                    )
            except Exception as e:
                # Log the detail; never send raw exception text to a user — the
                # renderer does not HTML-escape, and today these never reach the chat.
                print(f"[worker] Error answering in {chat_id}: {e!r}")
                self._send_message(
                    chat_id,
                    render_error(text, "Something went wrong while looking this up."),
                )
            finally:
                if message_key:
                    with self._inflight_lock:
                        self._inflight.pop(message_key, None)
                self._work_q.task_done()

    def _ensure_worker(self):
        """Start the single worker, or restart it if it has died.

        Called once at startup and once per poll cycle. A dead worker would
        otherwise be silent: the poll thread keeps acknowledging and nobody is
        ever answered.
        """
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            if self._worker is not None:
                print("ERROR: rag-worker thread died; restarting it")
            self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="rag-worker")
            self._worker.start()

    def _answer(self, chat_id, text, sender_name="Unknown"):
        """Worker thread: route, run RAG, reply. Never called from the poll loop."""
        # Deferred import: init_observability() (start_teams_bot.py) must run before
        # LlamaIndex loads, so this module never imports anything observability-adjacent
        # at module level — same reasoning as the deferred imports below and in _run_rag.
        from rag.observability import get_tracer
        from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

        tracer = get_tracer()
        # One root span for the whole request, opened before the router call and made
        # "current" (start_as_current_span attaches it) so every span started
        # underneath it — the router's own classification span, and, via _run_rag,
        # the agent run and its retrieval leaves (embed_query/search_vectors/rerank) —
        # nests under it instead of each becoming its own disconnected trace. Without
        # the attach, OpenInference's llama-index instrumentor finds no current span
        # and opens its own root, no matter how deep the call stack goes.
        with tracer.start_as_current_span(
            "compliance_request",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                # Full message text recorded deliberately, matching record_classification's
                # audit convention for this compliance bot. Never chat id or sender name —
                # those identify a person and the span does not need them.
                "compliance_request.question": text,
            },
        ) as span:
            # Pre-retrieval classification: only in-scope questions reach policy search.
            if settings.router_enabled:
                from rag.router import classify_message, resolve, Category  # deferred: observability-first
                from rag.observability import record_classification

                decision = classify_message(text)
                category = resolve(decision, settings.router_confidence_floor)
                record_classification(
                    category.value,
                    decision.confidence,
                    fallback=(category != decision.category or decision.fallback),
                    message=text,
                )

                # These three report their send like every other path out of _answer, so a
                # failed delivery reaches the worker's ERROR log instead of being silent.
                # No retry though: no GPU work is lost and the user can just say hello again.
                if category == Category.GREETING:
                    span.set_attribute("compliance_request.outcome", "greeting")
                    return bool(self._send_message(chat_id, WELCOME_HTML))
                if category == Category.OUT_OF_SCOPE:
                    span.set_attribute("compliance_request.outcome", "out_of_scope")
                    return bool(self._send_message(chat_id, render_out_of_scope()))
                if category == Category.UNINTELLIGIBLE:
                    span.set_attribute("compliance_request.outcome", "unintelligible")
                    return bool(self._send_message(chat_id, render_unintelligible()))
                # Category.IN_SCOPE falls through to the RAG pipeline below.

            result = _run_rag(text)

            # Transient backend failure — not an answer, not an escalation; no rating prompt.
            if result.get("status") == "unavailable":
                span.set_attribute("compliance_request.outcome", "unavailable")
                sent = self._send_message(chat_id, render_unavailable(), retry=True)
                if sent:
                    print("[worker] Unavailable notice sent")
                return bool(sent)

            # Render response
            escalation = result.get("escalation", {})
            if escalation.get("needed"):
                span.set_attribute("compliance_request.outcome", "escalated")
                html = render_escalation(text, result)
            elif result.get("answer"):
                span.set_attribute("compliance_request.outcome", "answered")
                html = render_answer(result)
            else:
                span.set_attribute("compliance_request.outcome", "error")
                html = render_error(text, "No answer returned from the pipeline.")

            # Answer and rating prompt as one Graph send, not two: RATING_PROMPT_HTML is
            # a self-contained <p><i>...</i></p> block and <hr> is Teams-allowed (CLAUDE.md's
            # rendering gotcha), so they concatenate cleanly. With one worker thread
            # serialising every question, the second round-trip (~490ms measured) was pure
            # queue wait for the next question — halving Graph round-trips here removes it.
            # Worth retrying: the pipeline already spent ~16s producing this.
            sent = self._send_message(chat_id, html + "<hr>" + RATING_PROMPT_HTML, retry=True)
            if sent:
                print("[worker] Reply sent")
                # Arm rating capture only if the combined send succeeded. This is now
                # structural rather than a second check: the answer and the prompt are
                # one message, so they always arrive together or not at all — it is no
                # longer possible for the user to see the answer but not the prompt (or
                # vice versa) and have their next message silently become a rating.
                _pending_ratings[chat_id] = {
                    "question": text,
                    "answer": result.get("answer", ""),
                    "citations": result.get("citations", []),
                    "user": sender_name,
                }
            return bool(sent)

    def _get_my_user_id(self):
        if self._my_user_id:
            return self._my_user_id

        token = self.token_refresher.get_access_token()
        if not token:
            return None

        try:
            payload_b64 = token.split(".")[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            claims = json.loads(base64.urlsafe_b64decode(payload_b64))
            self._my_user_id = claims.get("oid")
        except Exception as e:
            print(f"Could not decode user ID from token: {e}")

        return self._my_user_id

    def _should_process_message(self, message, my_user_id, chat_id):
        message_id = message.get("id")
        if not message_id:
            return False, "no_id"

        key = _message_key(chat_id, message_id)
        if key in self.processed_messages:
            return False, "already_processed"

        if message.get("messageType") != "message":
            self._mark_processed(key)
            return False, "system_message"

        sender_id = safe_get_nested(message, "from", "user", "id")
        if sender_id == my_user_id:
            self._mark_processed(key)
            return False, "self_message"

        created_datetime = message.get("createdDateTime")
        if not created_datetime:
            self._mark_processed(key)
            return False, "no_timestamp"

        try:
            created_time = datetime.fromisoformat(created_datetime.replace("Z", "+00:00"))
            if created_time <= self.last_check:
                self._mark_processed(key)
                return False, "old_message"
        except ValueError:
            self._mark_processed(key)
            return False, "invalid_timestamp"

        return True, None

    def _mark_processed(self, message_key):
        """Record a message key as seen.

        processed_messages is a dict (values unused) rather than a set purely for
        insertion order: a set iterates by hash, so evicting a slice of it drops
        near-random ids instead of the oldest ones.
        """
        self.processed_messages[message_key] = None

    def _cleanup_processed_messages(self):
        # Branch review Fix B: teams_max_processed_messages is a TRIGGER here,
        # not a cap — crossing it fires this, and this removes only 20% of the
        # current size (remove_count below), so resident size can run well past
        # the configured number (roughly 5x it in practice). Worse during a
        # hold: this only runs when fully_synced (Ruling H), i.e. at most once
        # per teams_max_state_age_minutes instead of every cycle. An operator
        # tuning this number to bound memory should expect "~5x this number",
        # not "this number" — see the setting's own comment in config.py.
        if len(self.processed_messages) > settings.teams_max_processed_messages:
            remove_count = len(self.processed_messages) // 5
            self.processed_messages = dict.fromkeys(list(self.processed_messages)[remove_count:])

    @staticmethod
    def _current_poll_interval(now):
        """Poll fast during the working week, slowly otherwise.

        The bot is a business-hours tool; polling every 5s around the clock
        spends roughly 70% of its Graph budget on hours nobody is asking.
        `now` must be timezone-aware UTC. The window is configured in UTC
        (default 07-19, i.e. 09/10-21/22 Kyiv) so it needs no tz database.

        start/end need not be ordered: start > end (e.g. 22-6) is a window that
        wraps past midnight, not an empty one — treating it as start <= hour < end
        would silently degrade to always-idle for any wrapping configuration.
        """
        is_weekday = now.weekday() < 5
        start = settings.teams_business_hours_start_utc
        end = settings.teams_business_hours_end_utc
        if start <= end:
            is_business_hours = start <= now.hour < end
        else:
            is_business_hours = now.hour >= start or now.hour < end  # wraps past midnight
        if is_weekday and is_business_hours:
            return settings.teams_poll_interval
        return settings.teams_idle_poll_interval

    def process_new_messages(self):
        """One poll cycle: read the chat list, read each chat's new messages, answer.

        last_check only advances when every fetch this cycle succeeded in full —
        see the `fully_synced` handling after the main loop below, and R-1 in
        task-4-rereview.md for why an *unbounded* hold on that watermark is itself
        a bug (it lets _cleanup_processed_messages evict ids the hold still needs,
        which then get re-fetched and re-answered by Ruling B's own back-paging).
        """
        my_user_id = self._get_my_user_id()
        if not my_user_id:
            return

        url = f"{GRAPH_API}/me/chats?$top={_CHATS_PAGE_SIZE}"
        chats, fully_synced = self._get_all_pages(url)
        # Remembered past this point (fully_synced gets overwritten below) so the
        # force-advance warning can say WHICH kind of incompleteness this cycle
        # had — Fix 6: "a chat" undersold it when the chat list itself is what
        # failed (e.g. a revoked refresh token surfacing as a swallowed 401),
        # which affects every chat's traffic, not one.
        chat_list_incomplete = not fully_synced
        any_chat_incomplete = False
        if chat_list_incomplete:
            print(
                f"WARNING: chat list fetch incomplete ({len(chats)} chat(s) retrieved); "
                "processing them but holding last_check so the chats behind the gap are "
                "retried next cycle instead of being marked as already seen"
            )

        newest_message_time = self.last_check

        for chat in chats:
            if not chat:
                continue
            chat_id = chat.get("id")
            if not chat_id:
                continue

            messages, chat_complete = self._get_chat_messages(chat_id)
            if not chat_complete:
                # _get_chat_messages logs a WARNING on every path that returns
                # complete=False (fetch failure or page-cap exhaustion) — see its
                # docstring; both branches print before returning. This chat's
                # unread history is not fully in hand, so — same reasoning as the
                # chat-list case above — the cycle as a whole cannot advance
                # last_check without risking messages behind the gap.
                fully_synced = False
                any_chat_incomplete = True
            # Graph returns newest-first. Answer people in the order they asked:
            # createdDateTime is a fixed-width ISO-8601 UTC string, so it sorts
            # chronologically as text. The watermark logic below is order-agnostic.
            messages = sorted(messages, key=lambda m: (m or {}).get("createdDateTime") or "")

            for message in messages:
                if not message:
                    continue

                should_process, _ = self._should_process_message(message, my_user_id, chat_id)
                if not should_process:
                    continue

                message_key = _message_key(chat_id, message.get("id"))
                message_text = safe_get_nested(message, "body", "content", default="")
                sender_name = safe_get_nested(message, "from", "user", "displayName", default="Unknown")

                created_datetime = message.get("createdDateTime")
                created_time = None  # never carry the previous iteration's timestamp
                if created_datetime:
                    try:
                        created_time = datetime.fromisoformat(created_datetime.replace("Z", "+00:00"))
                        if created_time > newest_message_time:
                            newest_message_time = created_time
                    except ValueError:
                        pass

                clean_message = strip_html(message_text)
                display = clean_message if clean_message.strip() else "[media/emoji]"
                print(f'\nNew message from {sender_name}: "{display}"')

                self._mark_processed(message_key)
                self._handle_inbound(
                    chat_id, clean_message,
                    sender_name=sender_name,
                    message_key=message_key,
                    created_time=created_time,
                )

        if fully_synced:
            self._hold_since = None
            # Known, pre-existing, out of this branch's scope (branch review
            # finding 8): newest_message_time is the max across ALL chats, so a
            # message arriving in chat A after A's own fetch, earlier this same
            # cycle, is buried if a later-read chat carries something newer —
            # the identical intra-cycle race Ruling M reasons about for the
            # force-advance below, just unguarded here. Unchanged since before
            # this branch; the window is a fraction of one poll cycle. Not
            # widening this branch's scope to fix it — recorded here so the
            # asymmetry with the force-advance's ten lines of reasoning reads
            # as a deliberate choice, not an oversight.
            if newest_message_time > self.last_check:
                self.last_check = newest_message_time
        else:
            # Bound the hold (Ruling G / R-1) — mirrors _load_state's startup
            # staleness clamp, applied continuously at runtime instead of only at
            # startup. Without this, one persistently-unreadable chat freezes
            # last_check forever: nothing here raises, so run()'s
            # consecutive_errors guard never trips; every message after the
            # freeze point stays permanently "new" by timestamp
            # (_should_process_message); and _cleanup_processed_messages evicts
            # the oldest ids with no regard for the freeze, so Ruling B's own
            # back-paging then re-fetches and re-answers them — the exact
            # "bot spams old answers" incident this project already fixed once
            # for the restart path. A bounded, logged, one-time skip is strictly
            # better than an unbounded stream of duplicate answers to everyone
            # else — but "whatever is stuck behind an unreadable chat" undersells
            # it (branch review Fix 6): this same path fires when the CHAT LIST
            # itself is unreadable (e.g. a revoked refresh token surfacing as a
            # swallowed 401, since _api_request never raises on one), in which
            # case it is not one chat's traffic that gets skipped but everyone's,
            # for up to teams_max_state_age_minutes. Either way nothing raises,
            # so run()'s consecutive_errors guard never trips and the process
            # looks healthy throughout — the WARNING below, which now says which
            # of the two happened, is the only signal.
            now = datetime.now(timezone.utc)
            if self._hold_since is None:
                self._hold_since = now
            held_for = now - self._hold_since
            if held_for > timedelta(minutes=settings.teams_max_state_age_minutes):
                # Target now - teams_initial_lookback_minutes, not now itself
                # (Ruling M): advancing all the way to now would silently bury
                # any message that arrives in a HEALTHY chat between that
                # chat's fetch earlier in this cycle and this point, later in
                # the same cycle — narrow, but exactly the silent loss this fix
                # exists to eliminate. _load_state's startup clamp — the
                # precedent this whole force-advance mirrors — makes the same
                # choice for the same reason: it resets to now - lookback, not
                # to now, deliberately leaving a small re-read window rather
                # than a hard cut at the instant of recovery. Re-reading that
                # window cannot create a duplicate: Ruling H has been blocking
                # eviction for the whole hold, so every id a healthy chat
                # already produced in that window is still in
                # processed_messages, and _should_process_message rejects it
                # by key before it ever reaches the timestamp check.
                forced_watermark = max(
                    newest_message_time,
                    now - timedelta(minutes=settings.teams_initial_lookback_minutes),
                )
                # Fix 6: say which kind of incompleteness this is. "A chat" reads
                # as one user's traffic; "the chat list itself" is everyone's.
                if chat_list_incomplete:
                    scope = (
                        "the chat list itself was unreadable this cycle, so every "
                        "chat's traffic (not just one) may be affected"
                    )
                elif any_chat_incomplete:
                    scope = "one or more individual chats were unreadable this cycle"
                else:
                    # Defensive: fully_synced is False, so one of the two flags
                    # above should always be set. Should not be reachable.
                    scope = "an unrecognized incompleteness"
                print(
                    f"WARNING: last_check held for {held_for} (exceeds "
                    f"{settings.teams_max_state_age_minutes} min) because {scope}; "
                    f"force-advancing to {forced_watermark.isoformat()} anyway. "
                    "Messages in the affected chat(s) may be permanently skipped."
                )
                self.last_check = forced_watermark
                # The hold is over: treat this cycle as synced from here on (the
                # cleanup gate below may run), and — this is what un-latches a
                # chat that by itself holds the freeze once far enough behind
                # (R-1's "self-latching" case) — a still-unreadable chat starts a
                # brand new hold timer against the now-current watermark rather
                # than perpetuating this one. It also re-bounds Ruling B's
                # back-paging reach (R-3): reach only grows with how long the
                # hold has run, and the hold is now capped.
                fully_synced = True
                self._hold_since = None

        # Ruling H: never evict while the watermark is held. An evicted id is
        # precisely what becomes re-answerable once its chat's back-paging (Ruling
        # B) reaches far enough to refetch it — eviction and an active hold must
        # never overlap. Safe to run immediately after a forced advance above:
        # the hold (if any) has just ended for this cycle. _cleanup_processed_messages
        # is the ONLY place eviction happens: _save_state (branch review Fix 2) no
        # longer applies its own slice, so there is nothing else to gate — a second,
        # ungated eviction site is exactly how this comment went false once already.
        if fully_synced:
            self._cleanup_processed_messages()
        self._save_state()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self._acquire_pid_lock()
        try:
            # Inside the try: a failure to start the worker must still release the PID lock.
            self._ensure_worker()
            print("Starting Compliance Teams Bot...")
            print("=" * 50)
            print(f"LLM: {settings.llm_model} ({settings.active_ollama_url})")
            print(
                f"Polling every {settings.teams_poll_interval}s "
                f"({settings.teams_business_hours_start_utc:02d}-{settings.teams_business_hours_end_utc:02d} UTC Mon-Fri), "
                f"{settings.teams_idle_poll_interval}s otherwise"
            )
            print("Workers: 1 (single-threaded by design — see _work_q comment)")
            print("=" * 50)
            print("\nWaiting for messages...\n")

            consecutive_errors = 0

            while True:
                try:
                    self._ensure_worker()  # restarts the worker if it ever died
                    self.process_new_messages()
                    consecutive_errors = 0
                    time.sleep(self._current_poll_interval(datetime.now(timezone.utc)))
                except KeyboardInterrupt:
                    print("\n\nBot stopped by user")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error in main loop ({consecutive_errors}/{settings.teams_max_consecutive_errors}): {e}")
                    if consecutive_errors >= settings.teams_max_consecutive_errors:
                        print("Too many consecutive errors, stopping bot")
                        break
                    error_sleep = min(settings.teams_poll_interval * (2 ** consecutive_errors), 60)
                    print(f"Retrying in {error_sleep}s...")
                    time.sleep(error_sleep)
        finally:
            self._release_pid_lock()
