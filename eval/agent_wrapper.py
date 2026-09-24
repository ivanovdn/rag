"""Instrumented agent for evaluation.

Mirrors production exactly: retrieval runs in rag/search_first.py before a
tool-free agent. The only difference is the tool-call log, which eval needs for
its agent_metadata. Do not give this agent tools — a divergence here means eval
stops measuring the thing that ships.
"""

from llama_index.core.agent.workflow import AgentWorkflow

from config import settings
from rag.agent import ALL_TOOLS, SYSTEM_PROMPT, get_llm
from rag.response import parse_agent_response  # re-exported for backwards compat
from rag.search_first import compose_agent_input, prefetch
import rag.tools.search_policies as sp

__all__ = [
    "build_instrumented_agent",
    "compose_agent_input",
    "get_log",
    "clear_log",
    "parse_agent_response",
    "prefetch_logged",
]

_tool_call_log: list[dict] = []


def get_log() -> list[dict]:
    return _tool_call_log


def clear_log() -> None:
    _tool_call_log.clear()


def prefetch_logged(question: str):
    """prefetch(), plus the log entry eval's agent_metadata is built from."""
    result = prefetch(question)
    _tool_call_log.append(
        {
            "tool": "search_policies",
            "query": question,
            "status": result.status,
            "results": list(sp._last_search_results),
        }
    )
    return result


def build_instrumented_agent(verbose: bool = False) -> AgentWorkflow:
    """Tool-free, like production. Fresh agent per call — no state leakage."""
    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=ALL_TOOLS,
        llm=get_llm(),
        system_prompt=SYSTEM_PROMPT,
        timeout=float(settings.agent_timeout),
        verbose=verbose,
    )
