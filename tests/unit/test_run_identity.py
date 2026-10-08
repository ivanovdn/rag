"""An experiment and a production trace must be matchable by what they ran.

Before this, experiments recorded a hash of SYSTEM_PROMPT and nothing else of
what the model reads, and production traces recorded nothing at all. So a
change to how sources are laid out for the model -- the likely shape of the
citation-selection fix (#10) -- would have produced two experiments with the
same prompt hash and different results, and no trace could say which of them
production was running.
"""

import inspect

import pytest

import channels.teams.bot as bot
import eval.run_experiment as rx
import rag.agent as agent
import rag.model_digest as md
import rag.router as router
import rag.run_identity as ri
import rag.search_first as search_first
import rag.tools.search_policies as sp


# --- the hashes ------------------------------------------------------------------


def test_the_system_prompt_hash_is_of_the_prompt_in_use():
    assert ri.prompt_identity()["system_prompt_sha12"] == ri.sha12(agent.SYSTEM_PROMPT)


def test_a_change_to_how_sources_are_laid_out_changes_the_input_hash(monkeypatch):
    """The gap this module exists for: the model reads the [Source N] layout as
    closely as it reads the prompt, and the prompt hash cannot see it."""
    before = ri.prompt_identity()
    original = sp.format_sources
    monkeypatch.setattr(
        sp, "format_sources", lambda results: original(results).replace("Clause Name:", "Clause:")
    )
    after = ri.prompt_identity()

    assert after["agent_input_sha12"] != before["agent_input_sha12"]
    assert after["system_prompt_sha12"] == before["system_prompt_sha12"]


def test_a_change_to_how_the_question_is_framed_changes_the_input_hash(monkeypatch):
    before = ri.prompt_identity()["agent_input_sha12"]
    monkeypatch.setattr(
        search_first, "compose_agent_input", lambda q, s: f"QUESTION: {q}\n\n{s}"
    )
    assert ri.prompt_identity()["agent_input_sha12"] != before


def test_the_sample_exercises_every_line_format_sources_can_emit():
    """A sample without a clause number would hash a layout that never shows
    one, and a change to that line would go unrecorded."""
    rendered = sp.format_sources(ri._SAMPLE_RESULTS)

    for label in ("[Source 1]", "[Source 2]", "Section:", "Clause Number:", "Clause Name:", "---"):
        assert label in rendered, label


def test_the_router_prompt_has_its_own_hash(monkeypatch):
    before = ri.router_prompt_sha12()
    monkeypatch.setattr(router, "ROUTER_SYSTEM_PROMPT", router.ROUTER_SYSTEM_PROMPT + "x")
    assert ri.router_prompt_sha12() != before
    assert "router_prompt_sha12" not in ri.prompt_identity()  # eval never runs the router


# --- the model digest ------------------------------------------------------------


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def ollama(monkeypatch):
    monkeypatch.setattr(ri.settings, "llm_backend", "ollama")
    monkeypatch.setattr(ri.settings, "llm_model", "qwen3.6:latest")
    monkeypatch.setattr(ri.settings, "use_remote_ollama", False)
    monkeypatch.setattr(ri.settings, "ollama_base_url", "http://h:11434")


def test_the_digest_is_shortened_to_what_the_banner_prints(ollama, monkeypatch):
    payload = {"models": [{"name": "qwen3.6:latest", "digest": "07d35212591f" + "a" * 52}]}
    monkeypatch.setattr(md.requests, "get", lambda url, timeout=None: _Resp(payload))
    assert ri.llm_digest12() == "07d35212591f"


def test_an_unreachable_host_costs_the_digest_not_the_run(ollama, monkeypatch):
    def _boom(url, timeout=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(md.requests, "get", _boom)
    assert ri.llm_digest12() == ""


def test_the_openai_compatible_backend_asks_nothing(monkeypatch):
    monkeypatch.setattr(ri.settings, "llm_backend", "openai-compatible")

    def _never(url, timeout=None):
        raise AssertionError("no /api/tags on that backend")

    monkeypatch.setattr(md.requests, "get", _never)
    assert ri.llm_digest12() == ""


# --- eval and production record the same names -------------------------------------


def test_experiments_record_every_answering_model_hash():
    meta = rx._prompt_meta()
    for key, value in ri.prompt_identity().items():
        assert meta[key] == value


def test_production_spans_carry_the_same_hashes_under_the_same_names(monkeypatch):
    monkeypatch.setattr(bot.settings, "git_commit", "22244b6")
    b = object.__new__(bot.TeamsBot)  # skip __init__: it loads state from disk
    b._llm_digest_at_startup = "07d35212591f"

    attrs = b._identity_attributes()

    assert attrs["identity.git_commit"] == "22244b6"
    assert attrs["identity.llm_digest_at_startup"] == "07d35212591f"
    assert attrs["identity.router_prompt_sha12"] == ri.router_prompt_sha12()
    for key, value in ri.prompt_identity().items():
        assert attrs[f"identity.{key}"] == value


def test_the_identity_is_on_the_root_span_of_every_request():
    src = inspect.getsource(bot.TeamsBot._answer)
    assert "**self._identity_attributes()" in src


def test_the_digest_starts_empty_until_the_banner_reads_it():
    src = inspect.getsource(bot.TeamsBot.__init__)
    assert 'self._llm_digest_at_startup = ""' in src
    assert "self._llm_digest_at_startup = _startup_identity()" in inspect.getsource(bot.TeamsBot.run)


# --- eval's commit and digest ------------------------------------------------------


def test_eval_falls_back_to_GIT_COMMIT_where_there_is_no_git(monkeypatch):
    def _no_git(*a, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(rx.subprocess, "run", _no_git)
    monkeypatch.setattr(rx.settings, "git_commit", "abc1234")
    assert rx._git_commit() == "abc1234"


def test_eval_prefers_git_over_the_image_commit(monkeypatch):
    """On the VM the image's GIT_COMMIT names the DEPLOYED code, not the mounted
    worktree being measured. Where git can answer, it wins."""

    class _Done:
        stdout = "f00ba12-dirty\n"

    monkeypatch.setattr(rx.subprocess, "run", lambda *a, **kw: _Done())
    monkeypatch.setattr(rx.settings, "git_commit", "22244b6")
    assert rx._git_commit() == "f00ba12-dirty"


def test_eval_warns_when_the_model_is_not_the_expected_one(monkeypatch, capsys):
    monkeypatch.setattr(ri, "llm_digest12", lambda: "aaaaaaaaaaaa")
    monkeypatch.setattr(rx.settings, "llm_model_digest", "07d35212591f")

    assert rx._llm_digest_meta() == {"llm_digest": "aaaaaaaaaaaa"}
    assert "WARNING" in capsys.readouterr().out


def test_eval_says_an_unchecked_digest_is_not_a_pass(monkeypatch, capsys):
    monkeypatch.setattr(ri, "llm_digest12", lambda: "")
    monkeypatch.setattr(rx.settings, "llm_model_digest", "07d35212591f")

    assert rx._llm_digest_meta() == {"llm_digest": ""}
    assert "did not run" in capsys.readouterr().out


def test_eval_is_quiet_when_the_digest_matches(monkeypatch, capsys):
    monkeypatch.setattr(ri, "llm_digest12", lambda: "07d35212591f")
    monkeypatch.setattr(rx.settings, "llm_model_digest", "07d35212591f")

    rx._llm_digest_meta()
    assert "WARNING" not in capsys.readouterr().out
