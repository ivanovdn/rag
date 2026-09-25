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
