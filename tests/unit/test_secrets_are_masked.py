"""The three live credentials in Settings must never render in plain text.

pydantic builds `repr(settings)` out of every field, and pytest's assertion
rewriting reprs the subexpressions of a failing assert into the failure message
-- which goes to the terminal, CI logs, pasted tracebacks and bug reports. So
`assert settings.ollama_num_ctx < 8192` could print three live credentials
because a NUMBER was wrong. `monkeypatch.setattr(settings, ...)` does the same
through AttributeError, which interpolates `repr(target)` when the attribute
name is mistyped.

Until this file, the defence was a hand-held convention -- bind the value to a
local so `settings` is never a bare name on an assert line -- documented in
three comments in tests/unit/test_llm_config.py and nowhere enforced. SecretStr
makes it a property of the type instead.

This file does not put a Settings repr on an assert line either: it binds a
bool first, and builds its Settings with `_env_file=None` so no real credential
is ever loaded into the object under test. A test that leaked the thing it
tests for would be its own counterexample.
"""

import json

import pytest
from pydantic import SecretStr

import channels.teams.auth as auth
from config import Settings

SECRET_FIELDS = ("hf_token", "teams_client_secret", "teams_refresh_token")

# Not a credential, and distinctive enough that grepping it finds only this file.
SYNTHETIC = "synthetic-value-not-a-real-credential"


def _hermetic(**overrides) -> Settings:
    """A Settings that never reads the real .env -- so the object under test holds
    no live secret even transiently, and the test behaves the same on a dev Mac
    (where .env exists) as in a container (where it does not)."""
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_a_secret_field_never_renders_its_value(field):
    cfg = _hermetic(**{field: SYNTHETIC})

    # repr() and str() differ on a pydantic model and both reach logs; the bare
    # getattr covers an f-string interpolating the field on its own.
    rendered = f"{cfg!r} {cfg} {getattr(cfg, field)}"
    leaked = SYNTHETIC in rendered

    # Bind first (see the module docstring) -- never `assert SYNTHETIC not in
    # f"{cfg!r}"`, which would repr the whole object into the failure message.
    assert not leaked, f"{field} renders its value in plain text"


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_a_secret_field_still_carries_its_value(field):
    """Masking is only safe if the value survives. A field that rendered as
    '**********' and also returned it would authenticate nothing, and would fail
    at the Azure or HuggingFace boundary rather than here."""
    cfg = _hermetic(**{field: SYNTHETIC})

    value = getattr(cfg, field)
    assert isinstance(value, SecretStr)
    assert value.get_secret_value() == SYNTHETIC


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_an_unset_secret_is_still_falsy(field):
    """Three call sites gate on truthiness (`if settings.hf_token:`,
    `if settings.teams_refresh_token:`). pydantic 2.12's SecretStr defines
    __len__, so SecretStr("") is False and those guards keep working -- but a
    pydantic that dropped __len__ would make every unset secret truthy and, for
    the refresh token, replace the actionable "No refresh token found" startup
    error with a silent seed of "". Measured on pydantic 2.12.5."""
    cfg = _hermetic(**{field: ""})

    assert not getattr(cfg, field)


def test_the_client_secret_posted_to_azure_is_a_real_string(tmp_path, monkeypatch):
    """channels/teams/auth.py:69 is the call site where a missed unwrap costs most.

    It fails loudly rather than silently -- requests' urlencode raises
    "TypeError: 'SecretStr' object is not iterable" (verified on requests 2.x /
    pydantic 2.12.5) instead of POSTing the mask -- but it fails inside the token
    refresh, whose symptom is the bot not authenticating. That is exactly the
    2026-10-06 outage signature (an expired client secret), so a type bug here
    would imitate a credential problem and send the next reader to the Azure
    portal for nothing.
    """
    token_file = tmp_path / "refresh_token.json"
    token_file.write_text(json.dumps({"refresh_token": "seed"}))
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)
    monkeypatch.setattr(auth.settings, "teams_client_secret", SecretStr(SYNTHETIC))
    refresher = auth.TokenRefresher()

    posted = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "tok", "expires_in": 3600}

    def _capture(url, data=None, timeout=None):
        posted.update(data)
        return _Response()

    monkeypatch.setattr(auth.requests, "post", _capture)
    refresher._refresh_access_token()

    secret = posted["client_secret"]
    assert isinstance(secret, str), "a SecretStr here breaks urlencode inside the refresh"
    assert secret == SYNTHETIC


def test_the_env_seed_refresh_token_is_unwrapped_to_a_plain_string(tmp_path, monkeypatch):
    """The seed path (no token file yet) feeds self.refresh_token, which is both
    POSTed to Azure and json.dump()ed back to refresh_token.json. A SecretStr
    there breaks the save with "not JSON serializable" -- and the save is what
    persists the rotated credential, so losing it costs an interactive
    device-code sign-in (SETUP.md Step 10)."""
    monkeypatch.setattr(auth, "TOKEN_FILE", tmp_path / "does-not-exist.json")
    monkeypatch.setattr(auth.settings, "teams_refresh_token", SecretStr(SYNTHETIC))

    refresher = auth.TokenRefresher()

    assert isinstance(refresher.refresh_token, str)
    assert refresher.refresh_token == SYNTHETIC
    # The value has to survive a round trip through the atomic save, which is the
    # operation that would raise on a SecretStr.
    refresher._save_refresh_token()
    assert json.loads(auth.TOKEN_FILE.read_text())["refresh_token"] == SYNTHETIC


def test_every_read_of_a_secret_unwraps_it():
    """Covers the call sites no behavioural test reaches.

    rag/embeddings.py only runs on EMBEDDING_SOURCE=huggingface, which
    production does not use -- so a missed unwrap there would surface first on
    whichever box did, as a TypeError inside model construction. This is the
    dependency-free net, in the shape of test_no_undefined_names.py.

    The rule: a `settings.<secret>` read is either immediately unwrapped with
    .get_secret_value(), or is the bare test of an `if`, which is safe because
    SecretStr defines __len__. Anything else -- into a dict, an f-string,
    os.environ, a log call -- is either a leak or a TypeError waiting for the
    configuration that reaches it.
    """
    import ast
    from pathlib import Path

    offenders = []
    for path in sorted(Path(".").rglob("*.py")):
        if any(part.startswith(".") or part in {"__pycache__", "tests"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }
        guards = {n.test for n in ast.walk(tree) if isinstance(n, (ast.If, ast.While))}

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr in SECRET_FIELDS):
                continue
            if not (isinstance(node.value, ast.Name) and node.value.id == "settings"):
                continue
            if node in guards:
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Attribute) and parent.attr == "get_secret_value":
                continue
            offenders.append(
                f"{path}:{node.lineno} reads settings.{node.attr} without .get_secret_value()"
            )

    assert offenders == [], "\n" + "\n".join(offenders)
