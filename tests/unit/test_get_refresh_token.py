"""scripts/get_refresh_token.py — the in-repo device-code re-auth / recovery script.

Not exercised end-to-end here: main() runs an interactive Microsoft sign-in flow
that would rotate the live production refresh token. This only checks import-time
wiring — that the script targets the exact file the bot reads, not a second literal
path that could drift from it.
"""
import channels.teams.auth as auth
import scripts.get_refresh_token as get_refresh_token


def test_targets_the_same_token_file_the_bot_reads():
    """TOKEN_FILE must be defined once (channels/teams/auth.py) and imported here —
    never duplicated as a second `Path("channels/teams/data/refresh_token.json")`
    literal that the two modules could drift apart on."""
    assert get_refresh_token.TOKEN_FILE is auth.TOKEN_FILE


def test_module_exposes_a_callable_main():
    """Importing the module must be inert: no network I/O, no prompt, no sign-in.
    (The import itself, succeeding at all, is also the "does it compile" check.)"""
    assert callable(get_refresh_token.main)
