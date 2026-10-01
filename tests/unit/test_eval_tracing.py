"""Eval runs emit Phoenix traces, into their own project.

Before this, `eval/run_experiment.py` never called `init_observability()` at all,
so a gate run produced no spans: no retrieval timings, no token counts, nothing
to inspect afterwards. The fix has to keep those traces out of the production
project, and — less obviously — has to keep the trace endpoint pointing at the
same Phoenix the experiment itself is written to. `--phoenix-url` moves only the
client; a trace endpoint left behind would send spans somewhere else entirely and
report nothing wrong.

Nothing here registers a real tracer or touches the network: `register` is
replaced, and `_initialized` is reset so each test exercises a fresh call.
"""

import argparse

import pytest

import rag.observability as obs
from eval.run_experiment import EVAL_PROJECT_NAME, _tracing_target


# --- init_observability's new parameters ---------------------------------------


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


def test_an_explicit_project_and_endpoint_reach_phoenix(captured_register):
    obs.init_observability(
        project_name="compliance-bot-eval", endpoint="http://172.20.1.10:6006/v1/traces"
    )

    assert captured_register["project_name"] == "compliance-bot-eval"
    assert captured_register["endpoint"] == "http://172.20.1.10:6006/v1/traces"


def test_omitting_them_falls_back_to_settings(monkeypatch, captured_register):
    monkeypatch.setattr(obs.settings, "phoenix_project_name", "compliance-bot")
    monkeypatch.setattr(obs.settings, "phoenix_endpoint", "http://localhost:6006/v1/traces")

    obs.init_observability()

    assert captured_register["project_name"] == "compliance-bot"
    assert captured_register["endpoint"] == "http://localhost:6006/v1/traces"


def test_phoenix_disabled_still_registers_nothing(monkeypatch, captured_register):
    """The kill switch outranks an explicit project — otherwise PHOENIX_ENABLED=false
    would stop meaning what it says the moment a caller passed an argument."""
    monkeypatch.setattr(obs.settings, "phoenix_enabled", False)

    obs.init_observability(project_name="compliance-bot-eval")

    assert captured_register == {}


# --- which project and endpoint an eval run resolves to ------------------------


def _args(**overrides):
    """The argparse Namespace main() builds, with only what _tracing_target reads."""
    defaults = {"no_trace": False, "phoenix_project": None, "phoenix_url": None}
    return argparse.Namespace(**{**defaults, **overrides})


def test_tracing_is_on_by_default_and_uses_its_own_project(monkeypatch):
    """Separate from the bot's project: a gate run's spans must not be mistaken for
    production traffic when someone reads the production project later."""
    monkeypatch.setattr("eval.run_experiment.settings.phoenix_project_name", "compliance-bot")

    target = _tracing_target(_args())

    assert target is not None
    project, _ = target
    assert project == EVAL_PROJECT_NAME
    assert project != "compliance-bot"


def test_no_trace_disables_it(monkeypatch):
    assert _tracing_target(_args(no_trace=True)) is None


def test_phoenix_project_overrides_the_default(monkeypatch):
    project, _ = _tracing_target(_args(phoenix_project="rrf-k-sweep"))

    assert project == "rrf-k-sweep"


def test_the_endpoint_defaults_to_the_configured_one(monkeypatch):
    monkeypatch.setattr(
        "eval.run_experiment.settings.phoenix_endpoint", "http://localhost:6006/v1/traces"
    )

    _, endpoint = _tracing_target(_args())

    assert endpoint == "http://localhost:6006/v1/traces"


def test_the_endpoint_follows_phoenix_url_when_given(monkeypatch):
    """--phoenix-url moves the client that writes the experiment. If traces did not
    follow, the experiment would land on one Phoenix and its spans on another, with
    nothing reporting the split."""
    monkeypatch.setattr(
        "eval.run_experiment.settings.phoenix_endpoint", "http://localhost:6006/v1/traces"
    )

    _, endpoint = _tracing_target(_args(phoenix_url="http://172.20.1.10:6006"))

    assert endpoint == "http://172.20.1.10:6006/v1/traces"


def test_a_phoenix_url_with_a_trailing_slash_does_not_double_up(monkeypatch):
    _, endpoint = _tracing_target(_args(phoenix_url="http://172.20.1.10:6006/"))

    assert endpoint == "http://172.20.1.10:6006/v1/traces"


# --- the run records where its traces went ------------------------------------


def test_experiment_metadata_names_the_project_holding_the_traces():
    """Without this, finding a six-week-old run's spans means guessing which
    project they landed in."""
    from pathlib import Path

    source = Path("eval/run_experiment.py").read_text(encoding="utf-8")

    assert '"phoenix_project": trace_target' in source


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
