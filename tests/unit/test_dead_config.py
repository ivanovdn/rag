"""Settings that nothing reads are deleted, and orphaned .env keys announce themselves.

Two halves of one problem.

The first half is the config itself: nine fields were read by no module and by no
computed property -- SMTP and compliance_team_email for an escalation email that
was never built, api_secret_key/admin_api_key for the removed HTTP API,
database_url for the db/ stub, eval_confidence_threshold for nothing at all.
Eight were advertised in .env.example, so setup filled in values that went
nowhere. smtp_password was also one of the four plain-str secrets that
repr(settings) carries into a pytest AttributeError.

The second half is why they survived. model_config sets extra="ignore", which is
load-bearing -- without it a stale key in the deployed .env is a hard
ValidationError at import and the bot will not start. But it also means deleting
a field silently demotes its .env key to decoration, with nothing anywhere
saying so. That is how MIN_CONFIDENCE_SCORE, RERANKER_QUERY_TEMPLATE,
BM25_AVG_LEN and RETRIEVAL_TOP_K each survived long enough to be tuned by hand.
unknown_env_keys() is the counterweight: ignore the key, but say its name.
"""

from pathlib import Path

import pytest

from config import Settings, unknown_env_keys


DELETED = (
    "smtp_host", "smtp_port", "smtp_user", "smtp_password",
    "compliance_team_email", "api_secret_key", "admin_api_key",
    "database_url", "eval_confidence_threshold",
)


def test_the_unread_settings_are_gone():
    present = [f for f in DELETED if f in Settings.model_fields]
    assert present == [], f"settings nothing reads are back: {present}"


def test_nothing_reads_them():
    offenders = []
    for path in Path(".").rglob("*.py"):
        if any(p.startswith(".") or p in {"__pycache__", "tests"} for p in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        for name in DELETED:
            if f"settings.{name}" in source or f"self.{name}" in source:
                offenders.append(f"{path}: {name}")
    assert offenders == [], f"still reading deleted settings: {offenders}"


def test_they_are_gone_from_the_example_env():
    text = Path(".env.example").read_text(encoding="utf-8")
    still = [n for n in DELETED if f"\n{n.upper()}=" in text]
    assert still == [], f"still advertised in .env.example: {still}"


def test_smtp_password_no_longer_rides_along_in_the_settings_repr():
    """One of the four plain-str secrets from the native-sparse follow-up list.

    monkeypatch.setattr interpolates repr(target) into its AttributeError, and
    repr(settings) prints every str field's value. Deleting an unused secret is
    the cheapest quarter of that fix.
    """
    assert "smtp_password" not in Settings.model_fields


# --- unknown_env_keys: the guard against the next silent orphan ---

def test_it_names_a_key_that_matches_no_field():
    assert unknown_env_keys("RETRIEVAL_TOP_K=20\n") == ["RETRIEVAL_TOP_K"]


def test_it_stays_quiet_when_every_key_is_a_real_field():
    assert unknown_env_keys("BM25_ENABLED=true\nRERANKER_CANDIDATES=25\n") == []


def test_it_ignores_comments_and_blank_lines():
    assert unknown_env_keys("# RETRIEVAL_TOP_K=20\n\n   \nBM25_ENABLED=true\n") == []


def test_it_matches_field_names_case_insensitively():
    """pydantic-settings resolves env keys case-insensitively, so a lowercase
    key in .env is live. Flagging it would send someone deleting a working line."""
    assert unknown_env_keys("bm25_enabled=true\n") == []


def test_a_value_containing_an_equals_sign_does_not_confuse_the_split():
    assert unknown_env_keys("RERANKER_QUERY_TEMPLATE=<Q>: {q}={x}\n") == []


def test_it_reports_every_orphan_in_order():
    text = "BM25_ENABLED=true\nHYBRID_BM25_CANDIDATES=20\nAGENT_MAX_ITERATIONS=5\n"
    assert unknown_env_keys(text) == ["HYBRID_BM25_CANDIDATES", "AGENT_MAX_ITERATIONS"]
