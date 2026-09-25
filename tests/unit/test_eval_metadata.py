"""Experiment metadata must identify the run's parameters.

An experiment whose configuration is only recoverable from its --name cannot be
compared against another one weeks later. The prompt is the parameter most likely
to be edited between runs and the easiest to lose track of, so its identity is
pinned here.
"""

import hashlib

from eval.run_experiment import _prompt_meta


def test_prompt_meta_identifies_the_prompt_actually_in_use():
    from rag.agent import FIXED_OVERHEAD_TOKENS, SYSTEM_PROMPT

    meta = _prompt_meta()
    assert meta["system_prompt_chars"] == len(SYSTEM_PROMPT)
    assert meta["fixed_overhead_tokens"] == FIXED_OVERHEAD_TOKENS
    assert meta["system_prompt_sha12"] == hashlib.sha256(
        SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()[:12]


def test_the_hash_actually_discriminates_between_prompts():
    """The whole point is that two runs with the same hash used the same prompt.

    A digest that did not change with the text would make every prompt experiment
    look identical — worse than recording nothing, because it would look right.
    """
    import rag.agent as agent_mod

    original = agent_mod.SYSTEM_PROMPT
    before = _prompt_meta()
    try:
        agent_mod.SYSTEM_PROMPT = original + "\nAn extra instruction.\n"
        after = _prompt_meta()
    finally:
        agent_mod.SYSTEM_PROMPT = original

    assert after["system_prompt_sha12"] != before["system_prompt_sha12"]
    assert after["system_prompt_chars"] > before["system_prompt_chars"]
    # and it must return to the original once the prompt is restored
    assert _prompt_meta()["system_prompt_sha12"] == before["system_prompt_sha12"]


def test_prompt_meta_does_not_drag_llamaindex_into_module_import():
    """run_experiment imports rag/ lazily; _prompt_meta must not break that.

    init_observability() has to run before LlamaIndex loads in the bot's entry
    points, and this module follows the same discipline.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("eval/run_experiment.py").read_text())
    top_level = [
        n.module
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module
    ]
    assert not [m for m in top_level if m.startswith("rag") or m.startswith("llama")]


# --- prompt registry mirror -------------------------------------------------
#
# The registry is a MIRROR of rag/agent.py, never a source. These pin the two
# properties that make that safe: it cannot break a run, and it cannot mangle the
# prompt on the way in.


class _FakePrompts:
    def __init__(self, existing=None, boom=False):
        self.existing, self.boom, self.created = existing, boom, []
        self.tags = self

    def get(self, **kw):
        if self.existing is None:
            raise LookupError("not registered")
        return self.existing

    def create(self, *, version, name, prompt_description=None, prompt_metadata=None):
        if self.boom:
            raise RuntimeError("registry write failed")
        self.created.append({"version": version, "name": name})
        return type("V", (), {"id": "version-123"})()


class _FakeClient:
    def __init__(self, prompts):
        self.prompts = prompts


def test_mirror_never_raises_and_returns_empty_when_the_registry_fails():
    """An eval run costs real GPU time on a shared host. A registry hiccup must
    cost the version id, not the run."""
    from eval.run_experiment import _mirror_prompt_to_registry

    result = _mirror_prompt_to_registry(_FakeClient(_FakePrompts(existing=None, boom=True)))
    assert result == {}


def test_mirror_reuses_the_existing_version_for_an_unchanged_prompt():
    """Idempotent by sha12 tag — re-running a sweep must not pile up duplicate
    versions of a prompt that never changed."""
    from eval.run_experiment import _mirror_prompt_to_registry

    prompts = _FakePrompts(existing=type("V", (), {"id": "already-there"})())
    result = _mirror_prompt_to_registry(_FakeClient(prompts))
    assert result == {"system_prompt_version_id": "already-there"}
    assert prompts.created == []


def test_mirror_stores_the_prompt_with_no_templating():
    """SYSTEM_PROMPT embeds a literal JSON block with { } braces — the output
    contract. F_STRING or MUSTACHE would try to interpolate them and mangle it,
    so the stored version must declare template_format NONE."""
    from eval.run_experiment import _mirror_prompt_to_registry

    prompts = _FakePrompts(existing=None)
    _mirror_prompt_to_registry(_FakeClient(prompts))
    assert len(prompts.created) == 1
    version = prompts.created[0]["version"]
    assert version._template_format == "NONE", version._template_format
