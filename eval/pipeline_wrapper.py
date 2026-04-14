"""
Eval wrapper for the vanilla (non-agentic) RAG pipeline.
Produces the same output format as agent_wrapper.py so all
evaluators work unchanged.

Usage by run_experiment.py:
    from eval.pipeline_wrapper import run_pipeline_task
    output = await run_pipeline_task(example)
"""

import rag.tools.search_policies as _sp_module
from rag.pipeline import run_query


def run_pipeline_task(example: dict) -> dict:
    """
    Task function for Phoenix experiments.
    Takes a dataset example, runs the vanilla pipeline, returns
    output in the same format as the agentic wrapper.
    """
    question = example.get("input", {}).get("question", "")

    if not question:
        return _empty_output("No question provided")

    # Run the vanilla pipeline
    result = run_query(question)

    # Capture search results for retrieval evaluators
    search_results = list(_sp_module._last_search_results)

    # Build output matching agent_wrapper format
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "escalation": result.get("escalation", {"needed": False, "reason": ""}),
        "parse_success": _check_parse_success(result),
        "raw_response": result,
        "search_results": search_results,
        "agent_metadata": {
            "search_queries": [question],
            "num_searches": 1,
            "num_section_fetches": 0,
            "section_fetches": [],
            "escalated": result.get("escalation", {}).get("needed", False),
            "escalation_reason": result.get("escalation", {}).get("reason", ""),
        },
    }


def _check_parse_success(result: dict) -> bool:
    """Check if the pipeline produced a valid structured response."""
    if not result.get("answer") and not result.get("escalation", {}).get("needed"):
        return False
    if result.get("escalation", {}).get("reason") == "Failed to parse structured response.":
        return False
    return True


def _empty_output(reason: str) -> dict:
    """Return empty output for error cases."""
    return {
        "answer": "",
        "citations": [],
        "escalation": {"needed": True, "reason": reason},
        "parse_success": False,
        "raw_response": "",
        "search_results": [],
        "agent_metadata": {
            "search_queries": [],
            "num_searches": 0,
            "num_section_fetches": 0,
            "section_fetches": [],
            "escalated": True,
            "escalation_reason": reason,
        },
    }
