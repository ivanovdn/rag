"""Teams bot — polls Microsoft Graph API for new messages, runs RAG pipeline directly."""

import base64
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from config import settings
from channels.teams.utils import safe_get_nested, strip_html
from channels.teams.renderer import (
    LOADING_HTML,
    WELCOME_HTML,
    render_answer,
    render_escalation,
    render_error,
)

GRAPH_API = "https://graph.microsoft.com/v1.0"
STATE_FILE = Path("channels/teams/data/bot_state.json")
PID_FILE = Path("channels/teams/data/bot.pid")


def _run_rag(question: str) -> dict:
    """Run the RAG pipeline directly (no HTTP). Returns ComplianceAnswer dict."""
    try:
        if settings.pipeline_mode == "vanilla":
            from rag.pipeline import run_query
            return run_query(question)
        else:
            import asyncio
            from rag.agent import build_agent
            from eval.agent_wrapper import parse_agent_response

            async def _run():
                agent = build_agent()
                return await agent.run(user_msg=question)

            response = asyncio.run(_run())
            return parse_agent_response(str(response))
    except Exception as e:
        print(f"RAG pipeline error: {e}")
        return {
            "answer": "",
            "citations": [],
            "escalation": {"needed": True, "reason": str(e)},
        }


class TeamsBot:
    def __init__(self, token_refresher):
        self.token_refresher = token_refresher
        self._my_user_id = None
        state = self._load_state()
        self.last_check = state["last_check"]
        self.processed_messages = state["processed_messages"]

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        default = {
            "last_check": datetime.now(timezone.utc) - timedelta(minutes=settings.teams_initial_lookback_minutes),
            "processed_messages": set(),
        }
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            last_check = datetime.fromisoformat(data["last_check"])
            processed = set(data.get("processed_messages", []))
            return {"last_check": last_check, "processed_messages": processed}
        except (FileNotFoundError, KeyError, ValueError):
            return default

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ids = list(self.processed_messages)[-settings.teams_max_processed_messages:]
        with open(STATE_FILE, "w") as f:
            json.dump(
                {
                    "last_check": self.last_check.isoformat(),
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
                if self._pid_is_running(existing_pid):
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

    def _api_request(self, url, method="GET", json_data=None):
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
        except requests.exceptions.Timeout:
            print(f"Request timeout for {url}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error for {url}: {e}")
            return None
        except Exception as e:
            print(f"Request failed for {url}: {e}")
            return None

    def _send_message(self, chat_id, text, content_type="html"):
        url = f"{GRAPH_API}/me/chats/{chat_id}/messages"
        payload = {"body": {"contentType": content_type, "content": text}}
        return self._api_request(url, method="POST", json_data=payload)

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _send_reply(self, chat_id, message_text):
        if not message_text or not message_text.strip():
            return False

        text = message_text.strip()

        if text.lower() in ("start", "/start", "help", "/help"):
            self._send_message(chat_id, WELCOME_HTML)
            return True

        if text == "[media/emoji]":
            return True

        # Show loading indicator
        self._send_message(chat_id, LOADING_HTML)

        # Run RAG directly
        result = _run_rag(text)

        # Render response
        escalation = result.get("escalation", {})
        if escalation.get("needed"):
            html = render_escalation(text, result)
        elif result.get("answer"):
            html = render_answer(result)
        else:
            html = render_error(text, "No answer returned from the pipeline.")

        sent = self._send_message(chat_id, html)
        if sent:
            print("Reply sent")
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
            self.processed_messages.add(message_id)
            return False, "system_message"

        sender_id = safe_get_nested(message, "from", "user", "id")
        if sender_id == my_user_id:
            self.processed_messages.add(message_id)
            return False, "self_message"

        created_datetime = message.get("createdDateTime")
        if not created_datetime:
            self.processed_messages.add(message_id)
            return False, "no_timestamp"

        try:
            created_time = datetime.fromisoformat(created_datetime.replace("Z", "+00:00"))
            if created_time <= self.last_check:
                self.processed_messages.add(message_id)
                return False, "old_message"
        except ValueError:
            self.processed_messages.add(message_id)
            return False, "invalid_timestamp"

        return True, None

    def _cleanup_processed_messages(self):
        if len(self.processed_messages) > settings.teams_max_processed_messages:
            remove_count = len(self.processed_messages) // 5
            self.processed_messages = set(list(self.processed_messages)[remove_count:])

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

                self.processed_messages.add(message_id)
                self._send_reply(chat_id, clean_message)

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
            print("Starting Compliance Teams Bot...")
            print("=" * 50)
            print(f"Pipeline: {settings.pipeline_mode}")
            print(f"LLM: {settings.llm_model} ({settings.active_ollama_url})")
            print(f"Polling every {settings.teams_poll_interval}s")
            print("=" * 50)
            print("\nWaiting for messages...\n")

            consecutive_errors = 0

            while True:
                try:
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
