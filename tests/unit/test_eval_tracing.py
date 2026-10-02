"""Eval runs emit Phoenix traces.

Before this, `eval/run_experiment.py` never called `init_observability()` at all,
so a gate run produced no spans: no retrieval timings, no token counts, nothing to
inspect afterwards. Measured 2026-10-02 on a real 61-question run: with the call,
the experiment carries 36 each of `embed_query`, `search_vectors` and `rerank`
plus the LlamaIndex agent spans. Without it, `auto_instrument` never installs the
instrumentor and `get_tracer()` hands back a no-op, so none of them exist.

What this deliberately does NOT do is choose the project. Phoenix's
`run_experiment` files task spans under its own per-experiment project and
overrides whatever was registered, so a project override here is inert — see
`test_the_run_does_not_claim_to_choose_a_project`.

Nothing here registers a real tracer or touches the network: `register` is
replaced, and `_initialized` is reset so each test exercises a fresh call.
"""

import argparse

import pytest

import rag.observability as obs
from eval.run_experiment import _tracing_endpoint


# --- init_observability's endpoint override ------------------------------------


@pytest.fixture
def captured_register(monkeypatch):
    """Replace phoenix.otel.register and hand back what init_observability passed it."""
    captured = {}

    def _fake_register(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(obs, "_initialized", False)
    monkeypatch.setattr(obs.settings, "phoenix_enabled", True)
    # init_observability imports `register` inside the function, so the patch has
    # to land on the module it is imported FROM, not on rag.observability.
    monkeypatch.setattr("phoenix.otel.register", _fake_register)
    return captured


def test_an_explicit_endpoint_reaches_phoenix(captured_register):
    obs.init_observability(endpoint="http://172.20.1.10:6006/v1/traces")

    assert captured_register["endpoint"] == "http://172.20.1.10:6006/v1/traces"


def test_omitting_it_falls_back_to_settings(monkeypatch, captured_register):
    monkeypatch.setattr(obs.settings, "phoenix_endpoint", "http://localhost:6006/v1/traces")
    monkeypatch.setattr(obs.settings, "phoenix_project_name", "compliance-bot")

    obs.init_observability()

    assert captured_register["endpoint"] == "http://localhost:6006/v1/traces"
    assert captured_register["project_name"] == "compliance-bot"


def test_phoenix_disabled_still_registers_nothing(monkeypatch, captured_register):
    """The kill switch outranks an explicit endpoint — otherwise PHOENIX_ENABLED=false
    would stop meaning what it says the moment a caller passed an argument."""
    monkeypatch.setattr(obs.settings, "phoenix_enabled", False)

    obs.init_observability(endpoint="http://172.20.1.10:6006/v1/traces")

    assert captured_register == {}


# --- which endpoint an eval run resolves to ------------------------------------


def _args(**overrides):
    """The argparse Namespace main() builds, with only what _tracing_endpoint reads."""
    defaults = {"no_trace": False, "phoenix_url": None}
    return argparse.Namespace(**{**defaults, **overrides})


def test_tracing_is_on_by_default(monkeypatch):
    monkeypatch.setattr(
        "eval.run_experiment.settings.phoenix_endpoint", "http://localhost:6006/v1/traces"
    )

    assert _tracing_endpoint(_args()) == "http://localhost:6006/v1/traces"


def test_no_trace_disables_it():
    assert _tracing_endpoint(_args(no_trace=True)) is None


def test_the_endpoint_follows_phoenix_url_when_given(monkeypatch):
    """--phoenix-url moves the client that writes the experiment. If traces did not
    follow, the experiment would land on one Phoenix and its spans on another, with
    nothing reporting the split."""
    monkeypatch.setattr(
        "eval.run_experiment.settings.phoenix_endpoint", "http://localhost:6006/v1/traces"
    )

    assert (
        _tracing_endpoint(_args(phoenix_url="http://172.20.1.10:6006"))
        == "http://172.20.1.10:6006/v1/traces"
    )


def test_a_phoenix_url_with_a_trailing_slash_does_not_double_up():
    assert (
        _tracing_endpoint(_args(phoenix_url="http://172.20.1.10:6006/"))
        == "http://172.20.1.10:6006/v1/traces"
    )


# --- what this entry point must NOT try to do ---------------------------------


def test_the_run_does_not_claim_to_choose_a_project():
    """Pins a deliberate absence, because the reason for it is not obvious.

    An earlier version of this file registered a dedicated `compliance-bot-eval`
    project and recorded it in experiment metadata. Measured 2026-10-02: Phoenix's
    run_experiment files task spans under its OWN per-experiment project and
    overrides the registered one, so `compliance-bot-eval` was never created (the
    API returned 404 for it) while the spans sat in `Experiment-<hash>`. The
    metadata key therefore pointed at a project that did not exist — worse than
    no key, since its stated purpose was to stop you guessing where traces went.

    Phoenix already records the real project on the experiment itself, and already
    isolates experiments from the bot's project without help.
    """
    from pathlib import Path

    source = Path("eval/run_experiment.py").read_text(encoding="utf-8")

    assert "phoenix_project" not in source
    assert "EVAL_PROJECT_NAME" not in source


def test_tracing_is_initialised_before_llamaindex_is_imported():
    """eval/agent_wrapper.py imports llama_index at module level and is pulled in by
    make_agent_task(). Phoenix's instrumentors have to be installed first or the
    agent's own spans are lost — the exact failure this change exists to fix.

    Compares real Call nodes inside main(), not source offsets: a substring search
    for "init_observability(" also matches the comments that explain this ordering,
    which sit near the top of the file and would make the check pass whatever the
    code did.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("eval/run_experiment.py").read_text(encoding="utf-8"))
    main_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    init_lines = [line for line, name in calls if name == "init_observability"]
    setup_lines = [line for line, name in calls if name == "setup_async"]

    assert init_lines, "main() never calls init_observability"
    assert setup_lines, "main() never calls setup_async"
    assert min(init_lines) < min(setup_lines), (
        "init_observability is called after setup_async, so LlamaIndex loads "
        "before Phoenix instruments it"
    )
