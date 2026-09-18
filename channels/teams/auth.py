"""Microsoft Graph API token management."""

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from config import settings

_TOKEN_ENDPOINT = f"https://login.microsoftonline.com/{settings.teams_tenant_id}/oauth2/v2.0/token"
_SCOPE = "https://graph.microsoft.com/.default"
_TOKEN_REFRESH_BUFFER = 300  # seconds before expiry to refresh
_TOKEN_REFRESH_COOLDOWN = 30  # seconds to wait before retrying a refresh that failed
TOKEN_FILE = Path("channels/teams/data/refresh_token.json")


class TokenRefresher:
    def __init__(self):
        # Prefer saved file (rotated token), fall back to .env (initial seed)
        try:
            with open(TOKEN_FILE, "r") as f:
                data = json.load(f)
                self.refresh_token = data["refresh_token"]
                print("Using refresh token from file")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            if settings.teams_refresh_token:
                self.refresh_token = settings.teams_refresh_token
                print("Using refresh token from .env")
            else:
                raise RuntimeError("No refresh token found. Set TEAMS_REFRESH_TOKEN in .env or run get_refresh_token.py")
        self.access_token = None
        self.token_expires_at = None
        # Set after a failed refresh; blocks further attempts until it passes.
        self._retry_refresh_after = None
        # Both the poll thread and the RAG worker call get_access_token().
        self._lock = threading.Lock()

    def _save_refresh_token(self):
        # Atomic: once Azure rotates past the .env seed, this file is the only copy of
        # the live credential — a plain open("w") truncates before writing, so a
        # container kill mid-write would leave an empty file and strand the bot with no
        # unattended way back (recovery is an interactive sign-in, scripts/get_refresh_token.py).
        # Stage in a same-directory temp file and rename over the target instead (atomic
        # on POSIX; os.replace across filesystems is not, so the temp file must sit next
        # to it, not under a different mount such as /tmp). Mirrors bot.py's _save_state.
        try:
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = TOKEN_FILE.with_name(TOKEN_FILE.name + ".tmp")
            with open(tmp_file, "w") as f:
                json.dump({"refresh_token": self.refresh_token}, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, TOKEN_FILE)
        except OSError:
            pass

    def _is_token_expired(self):
        if not self.access_token or not self.token_expires_at:
            return True
        remaining = self.token_expires_at - datetime.now(timezone.utc)
        return remaining.total_seconds() < _TOKEN_REFRESH_BUFFER

    def _refresh_access_token(self):
        data = {
            "client_id": settings.teams_client_id,
            "client_secret": settings.teams_client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
            "scope": _SCOPE,
        }
        try:
            # Timeout is required: get_access_token() holds self._lock across this
            # call, so a silent socket would stall the poll thread (no detection,
            # no acks) for as long as the OS lets the read hang.
            response = requests.post(_TOKEN_ENDPOINT, data=data, timeout=settings.teams_api_timeout)
            response.raise_for_status()
            token_data = response.json()

            self.access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            self._retry_refresh_after = None

            if "refresh_token" in token_data:
                self.refresh_token = token_data["refresh_token"]
                self._save_refresh_token()
                print("Refresh token updated")

            print(f"Access token refreshed (expires in {expires_in // 60} minutes)")
            return self.access_token

        except requests.exceptions.RequestException as e:
            print(f"Error refreshing token: {e}")
            if hasattr(e, "response") and hasattr(e.response, "text"):
                print(f"Response: {e.response.text}")
            # Cool off. Without this the token stays "expired", so every _get_headers()
            # on every Graph call retries the refresh under the lock at ~10s a time —
            # at 30 chats that is minutes of poll-thread stall per cycle during an AAD
            # outage. A dead credential is now retried every 30s instead.
            self._retry_refresh_after = datetime.now(timezone.utc) + timedelta(seconds=_TOKEN_REFRESH_COOLDOWN)
            return None

    def get_access_token(self):
        """Get access token, refreshing only if expired.

        Thread-safe: the poll thread (chat list, acks, ratings) and the RAG worker
        (answers) both call this. Without the lock, two threads that see an expired
        token refresh twice and can interleave the refresh_token.json rewrite.
        """
        with self._lock:
            if self._is_token_expired():
                # A failed refresh cools off rather than retrying on every Graph call.
                # _is_token_expired() alone can't throttle it: with no access_token yet
                # it is unconditionally True, so the cool-off is tracked separately.
                if self._retry_refresh_after and datetime.now(timezone.utc) < self._retry_refresh_after:
                    return self.access_token
                self._refresh_access_token()
            return self.access_token
