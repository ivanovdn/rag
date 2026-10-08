"""The startup banner must say which code and which model this process runs.

The 2026-10-08 deploy audit could not answer its own first question from the
log. Yesterday's merge was on both remotes, yet the container was still running
the image built the day before: the follow-ups had never been deployed, and the
only evidence was a warning that failed to appear. Working that out meant
reasoning about what was ABSENT from a log. A banner that names its commit turns
it into a glance.

The model half is the same question for the other moving part. Production's
LLM_MODEL is `qwen3.6:latest`, the only tag on a shared host this project does
not own. Anyone who re-pulls it there changes the weights under a running bot,
with no deploy and no log line, and every measurement taken here -- the
96-citation provenance check, the 571-token prompt budget, the MoE crash matrix
-- silently stops describing production. A tag cannot be pinned from this side.
Its digest can at least be read, printed and compared.
"""

import inspect
from pathlib import Path

import pytest
import requests

import channels.teams.bot as bot
import rag.model_digest as md

DIGEST = "07d35212591f" + "0" * 52  # 64 hex characters, shaped like the real one
TAGS = {
    "models": [
        {"name": "qwen3.6:latest", "model": "qwen3.6:latest", "digest": DIGEST},
        {"name": "embeddinggemma:latest", "model": "embeddinggemma:latest", "digest": "e" * 64},
    ]
}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


@pytest.fixture
def tags(monkeypatch):
    """Serve TAGS from /api/tags, and record every URL asked for."""
    calls = []

    def _get(url, timeout=None):
        calls.append(url)
        return _Resp(TAGS)

    monkeypatch.setattr(md.requests, "get", _get)
    return calls


def _fail_with(monkeypatch, exc):
    def _get(url, timeout=None):
        raise exc

    monkeypatch.setattr(md.requests, "get", _get)


@pytest.fixture
def ollama(monkeypatch):
    """Production's shape: Ollama backend, a known commit, no expected digest."""
    monkeypatch.setattr(bot.settings, "llm_backend", "ollama")
    monkeypatch.setattr(bot.settings, "llm_model", "qwen3.6:latest")
    monkeypatch.setattr(bot.settings, "use_remote_ollama", False)
    monkeypatch.setattr(bot.settings, "ollama_base_url", "http://h:11434")
    monkeypatch.setattr(bot.settings, "llm_model_digest", "")
    monkeypatch.setattr(bot.settings, "git_commit", "9e7c76e")


# --- reading a digest ----------------------------------------------------------


def test_the_digest_is_read_for_the_exact_tag(tags):
    assert md.ollama_model_digest("http://h:11434", "qwen3.6:latest") == DIGEST
    assert tags == ["http://h:11434/api/tags"]


def test_an_untagged_name_resolves_to_latest_as_ollama_does(tags):
    assert md.ollama_model_digest("http://h:11434", "qwen3.6") == DIGEST


def test_a_model_missing_from_the_host_is_named_not_guessed(tags):
    """No fallback to a similar name. A digest for the wrong model is worse than
    none, because it would be printed with the same confidence. The example is
    the tag CLAUDE.md documented as production's, which does not exist there."""
    with pytest.raises(LookupError, match="qwen3.6:35b"):
        md.ollama_model_digest("http://h:11434", "qwen3.6:35b")


# --- the banner ----------------------------------------------------------------


def test_the_banner_names_the_commit(ollama, tags):
    assert bot._identity_lines()[0] == "Build: 9e7c76e"


def test_an_image_built_without_the_commit_says_so_and_how_to_fix_it(ollama, tags, monkeypatch):
    monkeypatch.setattr(bot.settings, "git_commit", "unknown")

    first = bot._identity_lines()[0]

    assert first.startswith("Build: unknown")
    assert "GIT_COMMIT" in first


def test_the_banner_names_the_model_by_digest(ollama, tags):
    assert bot._identity_lines()[1] == "LLM: qwen3.6:latest @ 07d35212591f (http://h:11434)"


def test_a_matching_expected_digest_says_so_and_warns_nothing(ollama, tags, monkeypatch):
    monkeypatch.setattr(bot.settings, "llm_model_digest", "07D35212591F")  # case-insensitive

    lines = bot._identity_lines()

    assert "as expected" in lines[1]
    assert not [line for line in lines if line.startswith("WARNING")]


def test_a_moved_tag_is_a_warning_naming_both_digests(ollama, tags, monkeypatch):
    monkeypatch.setattr(bot.settings, "llm_model_digest", "aaaaaaaaaaaa")

    warnings = [line for line in bot._identity_lines() if line.startswith("WARNING")]

    assert len(warnings) == 1
    assert "07d35212591f" in warnings[0] and "aaaaaaaaaaaa" in warnings[0]


def test_an_unreachable_model_host_never_stops_the_bot_starting(ollama, monkeypatch):
    """The banner is diagnostic. The Spark box being slow at the moment of a
    deploy has to cost a line of text, not the deploy."""
    _fail_with(monkeypatch, requests.ConnectionError("no route to host"))

    llm = bot._identity_lines()[1]

    assert llm.startswith("LLM: qwen3.6:latest")
    assert "digest unavailable" in llm


def test_an_expected_digest_that_could_not_be_checked_is_not_reported_as_fine(ollama, monkeypatch):
    """Silence would read as a pass. If the operator asked for the check and it
    could not run, the banner has to say so rather than just omit it."""
    _fail_with(monkeypatch, requests.Timeout())
    monkeypatch.setattr(bot.settings, "llm_model_digest", "07d35212591f")

    warnings = [line for line in bot._identity_lines() if line.startswith("WARNING")]

    assert len(warnings) == 1 and "07d35212591f" in warnings[0]


def test_the_openai_compatible_backend_names_its_own_model_and_asks_ollama_nothing(
    ollama, tags, monkeypatch
):
    """That backend reads OPENAI_MODEL and OPENAI_API_BASE (rag/agent.py get_llm).
    The banner used to print LLM_MODEL and the Ollama URL regardless, naming a
    model the process was not using."""
    monkeypatch.setattr(bot.settings, "llm_backend", "openai-compatible")
    monkeypatch.setattr(bot.settings, "openai_model", "qwen2.5-32b")
    monkeypatch.setattr(bot.settings, "openai_api_base", "http://v:8082/v1")

    llm = bot._identity_lines()[1]

    assert "qwen2.5-32b" in llm and "http://v:8082/v1" in llm
    assert "qwen3.6" not in llm
    assert tags == []  # no /api/tags call: that endpoint does not exist there


def test_run_prints_the_identity_lines():
    assert "_identity_lines()" in inspect.getsource(bot.TeamsBot.run)


# --- getting the commit into the image -----------------------------------------


def test_the_commit_arg_sits_below_the_dependency_layer():
    """Every RUN below an ARG sees it as an environment variable, so a changing
    value busts their cache. Declared above `pip install`, GIT_COMMIT would
    reinstall every dependency on every deploy -- and with requirements-bot.txt's
    >= ranges, re-resolve them to whatever is newest that day. The line that
    reports the build would become the thing that changes it."""
    src = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ARG GIT_COMMIT" in src and "ENV GIT_COMMIT" in src
    assert src.index("ARG GIT_COMMIT") > src.index("RUN pip install")


def test_compose_passes_the_commit_through():
    src = Path("docker-compose-remote.yml").read_text(encoding="utf-8")

    assert "GIT_COMMIT: ${GIT_COMMIT:-unknown}" in src
