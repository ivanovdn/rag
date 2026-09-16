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
            return {"status": "unavailable"}
        print(f"RAG pipeline error: {e}")
        return {
            "answer": "",
            "citations": [],
            "escalation": {"needed": True, "reason": str(e)},
        }

    # Retrieval failed inside the tool (LlamaIndex swallows tool exceptions) →
    # the flag was set in search_policies; surface the unavailable outcome.
    if sp._retrieval_unavailable:
        return {"status": "unavailable"}

    return parse_agent_response(str(response))


class TeamsBot:
    def __init__(self, token_refresher):
        self.token_refresher = token_refresher
        self._my_user_id = None
        state = self._load_state()
        self.last_check = state["last_check"]
        self.processed_messages = state["processed_messages"]
        # EXACTLY ONE worker consumes this queue. Do not raise the worker count.
        # rag/tools/search_policies.py keeps _retrieval_unavailable and
        # _last_search_results as module globals, reset before an agent run and
        # read ~16s later. A second worker interleaves those resets and turns a
        # transient infra failure into a false content escalation, silently.
        # Fix those globals (ToolCallResult.tool_output is per-request) before
        # ever running more than one.
        self._work_q: "queue.Queue[tuple[str, str, str, str]]" = queue.Queue()
        self._worker: threading.Thread | None = None  # started by _ensure_worker() in run()
        # message_id -> createdDateTime, for messages accepted but not yet answered.
        self._inflight: dict[str, datetime] = {}
        self._inflight_lock = threading.Lock()

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
        watermark = min(pending_times) - timedelta(milliseconds=1) if pending_times else self.last_check

        # processed_messages is insertion-ordered, so this slice really is "the newest N".
        ids = [
            mid for mid in list(self.processed_messages)[-settings.teams_max_processed_messages:]
            if mid not in pending_ids
        ]
        with open(STATE_FILE, "w") as f:
            json.dump(
                {
                    "last_check": watermark.isoformat(),
                    "processed_messages": ids,
                },
                f,
                indent=2,
            )

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

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _handle_inbound(self, chat_id, message_text, sender_name="Unknown",
                        message_id=None, created_time=None):
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

        if message_id and created_time:
            with self._inflight_lock:
                self._inflight[message_id] = created_time
        else:
            # Untracked: the watermark advances past this message, so a crash before
            # it is answered drops it silently. Unreachable from process_new_messages
            # (_should_process_message rejects messages with no id or no parseable
            # timestamp) — log it rather than let a future caller lose work quietly.
            print(
                f"WARNING: enqueuing untracked message in {chat_id} "
                f"(message_id={message_id!r}, created_time={created_time!r}); "
                "a crash before it is answered will drop it"
            )

        self._send_message(chat_id, ACK_HTML)
        self._work_q.put((chat_id, text, sender_name, message_id or ""))
        return True

    def _worker_loop(self):
        """The single consumer. See the __init__ comment before adding a second."""
        while True:
            chat_id, text, sender_name, message_id = self._work_q.get()
            try:
                if not self._answer(chat_id, text, sender_name=sender_name):
                    # The answer was produced but the reply POST failed even after
                    # retries. Nothing re-sends it — the user has an ack and then
                    # silence — so say so loudly instead of dropping it quietly.
                    print(
                        f"ERROR: answer not delivered to chat {chat_id} "
                        f"(message_id={message_id or 'unknown'}, question={text[:80]!r})"
                    )
            except Exception as e:
                # Log the detail; never send raw exception text to a user — the
                # renderer does not HTML-escape, and today these never reach the chat.
                print(f"Worker error answering in {chat_id}: {e!r}")
                self._send_message(
                    chat_id,
                    render_error(text, "Something went wrong while looking this up."),
                )
            finally:
                if message_id:
                    with self._inflight_lock:
                        self._inflight.pop(message_id, None)
                self._work_q.task_done()

    def _ensure_worker(self):
        """Start the single worker, or restart it if it has died.

        Called once at startup and once per poll cycle. A dead worker would
        otherwise be silent: the poll thread keeps acknowledging and nobody is
        ever answered.
        """
        if self._worker is not None and self._worker.is_alive():
            return
        if self._worker is not None:
            print("ERROR: rag-worker thread died; restarting it")
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="rag-worker")
        self._worker.start()

    def _answer(self, chat_id, text, sender_name="Unknown"):
        """Worker thread: route, run RAG, reply. Never called from the poll loop."""
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
                return bool(self._send_message(chat_id, WELCOME_HTML))
            if category == Category.OUT_OF_SCOPE:
                return bool(self._send_message(chat_id, render_out_of_scope()))
            if category == Category.UNINTELLIGIBLE:
                return bool(self._send_message(chat_id, render_unintelligible()))
            # Category.IN_SCOPE falls through to the RAG pipeline below.

        result = _run_rag(text)

        # Transient backend failure — not an answer, not an escalation; no rating prompt.
        if result.get("status") == "unavailable":
            sent = self._send_message(chat_id, render_unavailable(), retry=True)
            if sent:
                print("Unavailable notice sent")
            return bool(sent)

        # Render response
        escalation = result.get("escalation", {})
        if escalation.get("needed"):
            html = render_escalation(text, result)
        elif result.get("answer"):
            html = render_answer(result)
        else:
            html = render_error(text, "No answer returned from the pipeline.")

        # Worth retrying: the pipeline already spent ~16s producing this.
        sent = self._send_message(chat_id, html, retry=True)
        if sent:
            print("Reply sent")
            # Send rating prompt and store pending context
            self._send_message(chat_id, RATING_PROMPT_HTML)
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

    def _should_process_message(self, message, my_user_id):
        message_id = message.get("id")
        if not message_id:
            return False, "no_id"

        if message_id in self.processed_messages:
            return False, "already_processed"

        if message.get("messageType") != "message":
            self._mark_processed(message_id)
            return False, "system_message"

        sender_id = safe_get_nested(message, "from", "user", "id")
        if sender_id == my_user_id:
            self._mark_processed(message_id)
            return False, "self_message"

        created_datetime = message.get("createdDateTime")
        if not created_datetime:
            self._mark_processed(message_id)
            return False, "no_timestamp"

        try:
            created_time = datetime.fromisoformat(created_datetime.replace("Z", "+00:00"))
            if created_time <= self.last_check:
                self._mark_processed(message_id)
                return False, "old_message"
        except ValueError:
            self._mark_processed(message_id)
            return False, "invalid_timestamp"

        return True, None

    def _mark_processed(self, message_id):
        """Record an id as seen.

        processed_messages is a dict (values unused) rather than a set purely for
        insertion order: a set iterates by hash, so evicting a slice of it drops
        near-random ids instead of the oldest ones.
        """
        self.processed_messages[message_id] = None

    def _cleanup_processed_messages(self):
        if len(self.processed_messages) > settings.teams_max_processed_messages:
            remove_count = len(self.processed_messages) // 5
            self.processed_messages = dict.fromkeys(list(self.processed_messages)[remove_count:])

    def process_new_messages(self):
        my_user_id = self._get_my_user_id()
        if not my_user_id:
            return

        url = f"{GRAPH_API}/me/chats"
        chats_data = self._api_request(url)
        chats = chats_data.get("value", []) if chats_data else []

        newest_message_time = self.last_check

        for chat in chats:
            if not chat:
                continue
            chat_id = chat.get("id")
            if not chat_id:
                continue

            messages_url = f"{GRAPH_API}/me/chats/{chat_id}/messages"
            messages_data = self._api_request(messages_url)
            messages = messages_data.get("value", []) if messages_data else []

            for message in messages:
                if not message:
                    continue

                should_process, _ = self._should_process_message(message, my_user_id)
                if not should_process:
                    continue

                message_id = message.get("id")
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

                self._mark_processed(message_id)
                self._handle_inbound(
                    chat_id, clean_message,
                    sender_name=sender_name,
                    message_id=message_id,
                    created_time=created_time,
                )

        if newest_message_time > self.last_check:
            self.last_check = newest_message_time
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
            print(f"Polling every {settings.teams_poll_interval}s")
            print("Workers: 1 (single-threaded by design — see _work_q comment)")
            print("=" * 50)
            print("\nWaiting for messages...\n")

            consecutive_errors = 0

            while True:
                try:
                    self._ensure_worker()  # restarts the worker if it ever died
                    self.process_new_messages()
                    consecutive_errors = 0
                    time.sleep(settings.teams_poll_interval)
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
