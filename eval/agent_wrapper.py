"""
Instrumented agent wrapper for evaluation.
Wraps tools with logging. Parses structured JSON responses.
"""

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

from rag.agent import SYSTEM_PROMPT, get_llm
from rag.response import parse_agent_response  # re-exported for backwards compat
from rag.tools.escalate import escalate_to_compliance as _original_escalate
from rag.tools.get_section import get_section as _original_get_section
from rag.tools.search_policies import search_policies as _original_search

__all__ = ["build_instrumented_agent", "get_log", "clear_log", "parse_agent_response"]

# ============================================================
# Global tool call log
# ============================================================

_tool_call_log: list[dict] = []


def get_log() -> list[dict]:
    return _tool_call_log


def clear_log() -> None:
    _tool_call_log.clear()


# ============================================================
# Wrapped tool functions
# ============================================================


def _logged_search_policies(query: str, top_k: int = 6) -> str:
    """Logged version — calls the real search pipeline (with reranker) and captures results."""
    import rag.tools.search_policies as _sp_module

    result_text = _original_search(query, top_k)

    _tool_call_log.append(
        {
            "tool": "search_policies",
            "query": query,
            "top_k": top_k,
            "results": list(_sp_module._last_search_results),
        }
    )

    return result_text


def _logged_get_section(doc_id: str, section_name: str) -> str:
    result = _original_get_section(doc_id, section_name)
    _tool_call_log.append(
        {
            "tool": "get_section",
            "doc_id": doc_id,
            "section_name": section_name,
            "result_length": len(result),
            "found": "No section found" not in result,
        }
    )
    return result


def _logged_escalate(
    reason: str, unanswered_question: str, search_attempted: bool = True
) -> str:
    result = _original_escalate(reason, unanswered_question, search_attempted)
    _tool_call_log.append(
        {
            "tool": "escalate_to_compliance",
            "reason": reason,
            "question": unanswered_question,
        }
    )
    return result


# Copy docstrings from originals
_logged_search_policies.__doc__ = _original_search.__doc__
_logged_get_section.__doc__ = _original_get_section.__doc__
_logged_escalate.__doc__ = _original_escalate.__doc__


# ============================================================
# Agent builder
# ============================================================


def build_instrumented_agent(verbose: bool = False) -> AgentWorkflow:
    """Build agent with logged tools.
    CRITICAL: name= MUST match system prompt references.
    Fresh agent per call — no state leakage.
    """
    llm = get_llm()
    instrumented_tools = [
        FunctionTool.from_defaults(fn=_logged_search_policies, name="search_policies"),
        FunctionTool.from_defaults(fn=_logged_get_section, name="get_section"),
        FunctionTool.from_defaults(fn=_logged_escalate, name="escalate_to_compliance"),
    ]
    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=instrumented_tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=verbose,
    )


# parse_agent_response is now imported from rag.response (see top of file).
