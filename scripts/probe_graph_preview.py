"""Does /me/chats?$expand=lastMessagePreview carry a usable message body?

Run ON THE BOT HOST WITH THE BOT STOPPED — refresh tokens rotate on use, and
this probe consumes and rotates the ONLY copy of the one the production bot
holds (channels/teams/data/refresh_token.json). Run it anywhere else, or
with the bot still running, and Azure invalidates the token the bot is
using — taking it offline with no unattended way back. Recovery is an
interactive sign-in (see scripts/get_refresh_token.py), not a restart.

    PYTHONPATH=. python scripts/probe_graph_preview.py
"""

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from channels.teams.auth import TokenRefresher

GRAPH_API = "https://graph.microsoft.com/v1.0"
# The exact URL Task 5 Step 5 will use in production — probe the combination, not the parts.
CHATS_URL = (
    f"{GRAPH_API}/me/chats"
    "?$expand=lastMessagePreview"
    "&$orderby=lastMessagePreview/createdDateTime desc"
    "&$top=50"
)


def main():
    token = TokenRefresher().get_access_token()
    if not token:
        print("No access token — aborting.")
        return 1

    t0 = time.perf_counter()
    resp = requests.get(CHATS_URL, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"HTTP {resp.status_code} in {elapsed_ms:.0f} ms")
    if resp.status_code != 200:
        print(resp.text[:600])
        return 1

    payload = resp.json()
    chats = payload.get("value", [])
    print(f"chats on page 1: {len(chats)} | nextLink present: {'@odata.nextLink' in payload}")

    go = True
    for chat in chats[:10]:
        preview = chat.get("lastMessagePreview") or {}
        body = (preview.get("body") or {}).get("content", "")
        sender = preview.get("from") or {}
        from_user_id = (sender.get("user") or {}).get("id")
        from_app_id = (sender.get("application") or {}).get("id")
        print(json.dumps({
            "chat_id": (chat.get("id") or "")[:24],
            "preview_id": preview.get("id"),
            "createdDateTime": preview.get("createdDateTime"),
            "messageType": preview.get("messageType"),
            "from_user_id": from_user_id,
            "from_app_id": from_app_id,
            "body_len": len(body),
            "body_head": body[:120],
        }, indent=2))
        if not preview.get("id") or not preview.get("createdDateTime"):
            go = False
        # systemEventMessage previews have from=null by design; only real messages need a sender.
        if preview.get("messageType") == "message" and not (from_user_id or from_app_id):
            go = False

    print("\nGO if: every preview has id + createdDateTime, and every messageType=='message'")
    print("preview has a sender (from.user.id, or from.application.id for bot-sent messages).")
    print("Body completeness is irrelevant: the skip is keyed on id, not text.")
    print("Record the elapsed ms above — Graph latency was assumed (~250 ms) in the audit, never measured.")
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'}")
    return 0 if go else 2


if __name__ == "__main__":
    raise SystemExit(main())
