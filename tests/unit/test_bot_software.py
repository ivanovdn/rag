from channels.teams.bot import TeamsBot


class _Bot(TeamsBot):
    def __init__(self):
        # bypass TeamsBot.__init__ (no token refresher / state needed for this test)
        self.sent = []

    def _send_message(self, chat_id, text, content_type="html"):
        self.sent.append(text)
        return True


def test_software_not_found_renders_not_listed_and_no_rating(monkeypatch):
    import channels.teams.bot as botmod

    monkeypatch.setattr(botmod.settings, "router_enabled", False)
    monkeypatch.setattr(
        botmod, "_run_rag",
        lambda q: {"status": "software_not_found", "name": "ZzzTool",
                   "suggestion": {"name": "Docker", "status": "allowed"}},
    )

    bot = _Bot()
    bot._send_reply("chat1", "is ZzzTool allowed?", sender_name="Alice")

    joined = "\n".join(bot.sent)
    assert "isn't on the approved or forbidden software list" in joined
    assert "Was this helpful?" not in joined  # no rating prompt
    assert "chat1" not in botmod._pending_ratings  # no pending feedback row
