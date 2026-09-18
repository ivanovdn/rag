"""One-time interactive sign-in to obtain a Microsoft Graph refresh token via the
device-code flow.

This is also the RECOVERY procedure if channels/teams/data/refresh_token.json is
ever lost or corrupted. Azure invalidates a refresh token every time it is used, so
the TEAMS_REFRESH_TOKEN seed in .env is only good for the bot's first refresh —
after that, this file is the only surviving copy of the live credential, and
restoring an old copy of it does not work either (it too has already been rotated
past). A fresh interactive sign-in is the only way back. See SETUP.md.

Run this, sign in with the bot's Teams account, and the resulting refresh token is
saved to the same file the bot itself reads and rotates (TOKEN_FILE in
channels/teams/auth.py — imported from there, not duplicated here). The bot then
reuses it indefinitely, rotating it as Microsoft issues new ones (about 14x/day in
production).

Prerequisites on the Azure app registration:
  * "Allow public client flows" = Yes (Authentication blade)
  * Delegated API permissions: Chat.Read, Chat.ReadWrite,
    offline_access, User.Read  (admin-consented if required by tenant)

Usage:
    PYTHONPATH=. python scripts/get_refresh_token.py
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from channels.teams.auth import TOKEN_FILE

# Different grant type than auth.py's _SCOPE ("https://graph.microsoft.com/.default"):
# a device-code sign-in requests explicit delegated scopes, not .default. Not a bug.
SCOPES = "Chat.Read Chat.ReadWrite User.Read offline_access"


def _save_token_atomically(refresh_token: str) -> None:
    """Stage in a same-directory temp file and rename onto TOKEN_FILE — the same
    atomic pattern as TokenRefresher._save_refresh_token (channels/teams/auth.py),
    so a kill mid-write can't leave a truncated or empty credential file. Unlike
    that method, a failure here is NOT swallowed: this is a one-shot interactive
    sign-in with no in-memory fallback, so the caller must find out immediately if
    the token they just granted was not actually persisted.
    """
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = TOKEN_FILE.with_name(TOKEN_FILE.name + ".tmp")
    with open(tmp_file, "w") as f:
        json.dump({"refresh_token": refresh_token}, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, TOKEN_FILE)


def main() -> int:
    tenant_id = settings.teams_tenant_id
    client_id = settings.teams_client_id

    if not tenant_id or not client_id:
        print("ERROR: TEAMS_TENANT_ID and TEAMS_CLIENT_ID must be set in .env")
        return 1

    devicecode_url = (
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/devicecode"
    )
    token_url = (
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    )

    # 1) Request a device code
    resp = requests.post(
        devicecode_url,
        data={"client_id": client_id, "scope": SCOPES},
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"ERROR requesting device code: {resp.status_code}")
        print(resp.text)
        return 1

    payload = resp.json()
    device_code = payload["device_code"]
    user_code = payload["user_code"]
    verification_uri = payload["verification_uri"]
    interval = int(payload.get("interval", 5))
    expires_in = int(payload.get("expires_in", 900))

    print()
    print("=" * 70)
    print("  Sign in to grant the compliance bot access to Teams")
    print("=" * 70)
    print(f"  1. Open this URL in your browser:  {verification_uri}")
    print(f"  2. Enter this code:                {user_code}")
    print(f"  3. Sign in with the bot's Teams account.")
    print("=" * 70)
    print(f"  Waiting for sign-in (expires in {expires_in // 60} min)...")
    print()

    # 2) Poll the token endpoint until the user completes sign-in
    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(interval)

        token_resp = requests.post(
            token_url,
            data={
                "client_id": client_id,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
            },
            timeout=10,
        )
        token_data = token_resp.json()

        if token_resp.status_code == 200:
            refresh_token = token_data.get("refresh_token")
            if not refresh_token:
                print("ERROR: Sign-in succeeded but no refresh_token returned.")
                print("Make sure 'offline_access' is in the requested scopes")
                print("and granted on the app registration.")
                return 1

            try:
                _save_token_atomically(refresh_token)
            except OSError as e:
                print(f"ERROR: sign-in succeeded but the token could not be saved: {e}")
                print(f"(target file: {TOKEN_FILE})")
                return 1

            print("Sign-in complete.")
            print(f"Refresh token saved to: {TOKEN_FILE}")
            print()
            print("You can now start the bot:")
            print("    PYTHONPATH=. python scripts/start_teams_bot.py")
            return 0

        error = token_data.get("error")
        if error == "authorization_pending":
            # User has not finished signing in yet — keep polling
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "authorization_declined":
            print("Sign-in was declined.")
            return 1
        if error == "expired_token":
            print("Device code expired before sign-in completed.")
            return 1

        # Unexpected error — show it and stop
        print(f"ERROR from token endpoint: {error}")
        print(json.dumps(token_data, indent=2))
        return 1

    print("Timed out waiting for sign-in.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
