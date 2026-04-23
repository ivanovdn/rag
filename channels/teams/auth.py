"""Microsoft Graph API token management."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from config import settings

_TOKEN_ENDPOINT = f"https://login.microsoftonline.com/{settings.teams_tenant_id}/oauth2/v2.0/token"
_SCOPE = "https://graph.microsoft.com/.default"
_TOKEN_REFRESH_BUFFER = 300  # seconds before expiry to refresh
_TOKEN_FILE = Path("channels/teams/data/refresh_token.json")


class TokenRefresher:
    def __init__(self):
        # Prefer saved file (rotated token), fall back to .env (initial seed)
        try:
            with open(_TOKEN_FILE, "r") as f:
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

    def _save_refresh_token(self):
        try:
            _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_TOKEN_FILE, "w") as f:
                json.dump({"refresh_token": self.refresh_token}, f, indent=4)
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
            response = requests.post(_TOKEN_ENDPOINT, data=data)
            response.raise_for_status()
            token_data = response.json()

            self.access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

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
            return None

    def get_access_token(self):
        """Get access token, refreshing only if expired."""
        if self._is_token_expired():
            self._refresh_access_token()
        return self.access_token
