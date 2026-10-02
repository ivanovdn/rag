"""Retrieval has ONE candidate knob: RERANKER_CANDIDATES.

Three settings were deleted, for two different reasons.

RETRIEVAL_TOP_K was inert. `search_chunks` read it only through
`limit = top_k or settings.retrieval_top_k`, and every caller passed a truthy
top_k -- reranker_candidates when the reranker is on, the caller's own top_k
when it is off. Its last reader outside that dead branch was
eval/evaluators.py's `cfg_top_k`, in a build_config_evaluators() that nothing
calls and that referenced a `settings.reranker_top_k` which has never existed --
proof it had never run. That block went with it. Fourth member of the family
with MIN_CONFIDENCE_SCORE, RERANKER_QUERY_TEMPLATE and BM25_AVG_LEN.

HYBRID_VECTOR_CANDIDATES and HYBRID_BM25_CANDIDATES were worse than inert: at
20 each against a fused limit of 25 they capped the union the fusion drew from,
so the pool shrank exactly when the two branches agreed. See
test_each_prefetch_branch_supplies_at_least_the_fused_limit in
tests/unit/test_fused_retrieval.py for the mechanism.

Re-adding any of them re-opens one of those two failures, so this guards the
names rather than the behaviour.
"""

from pathlib import Path

from config import Settings


DELETED = (
    "retrieval_top_k",
    "hybrid_vector_candidates",
    "hybrid_bm25_candidates",
)


def test_the_settings_are_gone_from_the_model():
    present = [name for name in DELETED if name in Settings.model_fields]
    assert present == [], f"deleted retrieval settings are back: {present}"


def test_nothing_reads_them():
    """Source files only -- this test names them in its own docstring."""
    offenders = []
    for path in Path(".").rglob("*.py"):
        if any(part.startswith(".") or part in {"__pycache__", "tests"} for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        for name in DELETED:
            if f"settings.{name}" in source:
                offenders.append(f"{path}: settings.{name}")
    assert offenders == [], f"still reading deleted settings: {offenders}"


def test_the_stale_env_entries_are_gone():
    """An env var documented for a setting that no longer exists is how someone
    spends an afternoon tuning a number that cannot do anything."""
    text = Path(".env.example").read_text(encoding="utf-8")
    for name in ("RETRIEVAL_TOP_K", "HYBRID_VECTOR_CANDIDATES", "HYBRID_BM25_CANDIDATES"):
        assert name not in text, f"{name} still in .env.example"
