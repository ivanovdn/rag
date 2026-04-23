#!/usr/bin/env python
"""
Start the Compliance Teams Bot.

Runs the RAG pipeline directly.

Usage:
    PYTHONPATH=. python scripts/start_teams_bot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()

from channels.teams.auth import TokenRefresher
from channels.teams.bot import TeamsBot

if __name__ == "__main__":
    token_refresher = TokenRefresher()
    bot = TeamsBot(token_refresher)
    bot.run()
