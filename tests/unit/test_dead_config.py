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

from config import Settings, inert_env_keys, unknown_env_keys


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


# --- set, but inert on this configuration ------------------------------------
#
# The third shape, after "field nobody reads" and "key matching no field": the
# field exists, something reads it, and the branch that reads it is not the one
# this deployment runs. unknown_env_keys() cannot see it, because the key does
# match a real field. No test of the code can see it either, because nothing
# about the code is wrong. Only the pairing of a value with a configuration is.


def test_a_query_template_set_against_a_vllm_backend_is_named():
    cfg = Settings(_env_file=None, reranker_backend="vllm-score")

    reported = inert_env_keys(cfg, env_text="RERANKER_QUERY_TEMPLATE=x\n", environ={})

    assert any("RERANKER_QUERY_TEMPLATE" in r for r in reported), reported


def test_the_same_template_on_llama_server_is_left_alone():
    """Inert is a property of the pairing, not of the key."""
    cfg = Settings(_env_file=None, reranker_backend="llama-server")

    reported = inert_env_keys(cfg, env_text="RERANKER_QUERY_TEMPLATE=x\n", environ={})

    assert reported == []


def test_a_key_the_operator_never_set_is_never_named():
    """Only what someone actually typed. A field sitting at its code default is
    not a mistake, and warning about it on every start is precisely how a
    warning becomes something people stop reading."""
    cfg = Settings(_env_file=None, reranker_backend="vllm-score")

    assert inert_env_keys(cfg, env_text="", environ={}) == []


def test_it_sees_a_key_that_only_the_container_has():
    """The case unknown_env_keys() structurally cannot cover.

    Its docstring is right that in a container there is no set of "keys the
    operator meant as settings" to enumerate -- they arrive indistinguishable
    from PATH. But this check never enumerates: it asks after two keys it already
    knows the names of, and nothing but this project sets either. So the one
    place the orphan warning is deaf, this one hears, which matters because the
    deployed bot is exactly that place."""
    cfg = Settings(_env_file=None, reranker_backend="vllm-score")

    reported = inert_env_keys(cfg, env_text="", environ={"RERANKER_QUERY_TEMPLATE": "x"})

    assert any("RERANKER_QUERY_TEMPLATE" in r for r in reported), reported


def test_the_cosine_floor_is_named_when_it_cannot_fire():
    """MIN_CONFIDENCE_SCORE is the oldest member of this family and the reason
    the family has a name -- it was tuned by hand while being unreachable."""
    cfg = Settings(_env_file=None, reranker_enabled=True)

    reported = inert_env_keys(cfg, env_text="MIN_CONFIDENCE_SCORE=0.45\n", environ={})

    assert any("MIN_CONFIDENCE_SCORE" in r for r in reported), reported


def test_every_report_names_the_setting_that_does_apply():
    """A warning that says only "this does nothing" sends the reader hunting,
    and the hunt is the expensive part -- both of these keys have a live
    counterpart that is easy to miss and easy to confuse with them."""
    cfg = Settings(_env_file=None, reranker_backend="vllm-score", reranker_enabled=True)

    reported = inert_env_keys(
        cfg,
        env_text="RERANKER_QUERY_TEMPLATE=x\nMIN_CONFIDENCE_SCORE=0.45\n",
        environ={},
    )

    joined = " ".join(reported)
    assert len(reported) == 2, reported
    assert "RERANKER_INSTRUCTION" in joined
    assert "RERANKER_MIN_SCORE" in joined


def test_startup_reports_inert_keys_beside_the_orphans():
    """Wired into get_settings, or it is a function nobody calls -- which is the
    failure mode this whole file exists to catch."""
    source = Path("config.py").read_text(encoding="utf-8")
    getter = source[source.index("def get_settings()"):]

    assert "inert" in getter, "get_settings() does not report inert keys"
