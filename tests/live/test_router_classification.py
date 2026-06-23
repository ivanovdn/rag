import pytest

from tests._llm import llm_reachable

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(not llm_reachable(), reason="no reachable LLM (local-only)"),
]

CASES = [
    ("hi", "greeting"),
    ("hello there", "greeting"),
    ("thanks!", "greeting"),
    ("Can I install software on my work laptop?", "in_scope"),
    ("What is our policy on remote work?", "in_scope"),
    ("How many vacation days do I get?", "in_scope"),
    ("order me a pizza", "out_of_scope"),
    ("who is Sarah Connor", "out_of_scope"),
    ("what is the weather today", "out_of_scope"),
    ("црфе ші ърщдшсн", "unintelligible"),
    ("asdkj qweoiu zxcmnv", "unintelligible"),
]


def test_classifier_accuracy_on_labeled_set():
    """Tuning signal (not a hard gate): a live model is non-deterministic, so we assert
    aggregate accuracy and report misses rather than failing per case."""
    # Imported lazily (not at module top, despite CLAUDE.md): a module-top import of
    # rag.router pulls the llama-index chain at pytest collection time on EVERY offline
    # run, since collection imports this module before the live_llm marker deselects it.
    # This test auto-skips without an LLM, so we defer the heavy import to execution.
    from rag.router import classify_message

    misses = []
    for text, expected in CASES:
        got = classify_message(text).category.value
        if got != expected:
            misses.append((text, expected, got))
    accuracy = (len(CASES) - len(misses)) / len(CASES)
    assert accuracy >= 0.8, f"classifier accuracy {accuracy:.0%} < 80%; misses={misses}"
