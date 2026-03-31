"""
Instrumented agent wrapper for evaluation.
Wraps tools with logging. Parses structured JSON responses.
"""

import json
import re

from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama

from config import settings
from rag.hybrid_search import hybrid_search
from rag.tools.search_policies import search_policies as _original_search
from rag.tools.get_section import get_section as _original_get_section
from rag.tools.escalate import escalate_to_compliance as _original_escalate
from rag.agent import SYSTEM_PROMPT


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
    raw_results = hybrid_search(query, top_k=top_k)
    _tool_call_log.append({
        "tool": "search_policies",
        "query": query,
        "top_k": top_k,
        "results": [
            {
                "doc_title": r["doc_title"],
                "section": r["section"],
                "clause": r.get("clause", ""),
                "clause_number": r.get("clause_number", ""),
                "rrf_score": round(r["rrf_score"], 4),
            }
            for r in raw_results
        ],
    })
    return _original_search(query, top_k)


def _logged_get_section(doc_id: str, section_name: str) -> str:
    result = _original_get_section(doc_id, section_name)
    _tool_call_log.append({
        "tool": "get_section",
        "doc_id": doc_id,
        "section_name": section_name,
        "result_length": len(result),
        "found": "No section found" not in result,
    })
    return result


def _logged_escalate(reason: str, unanswered_question: str, search_attempted: bool = True) -> str:
    result = _original_escalate(reason, unanswered_question, search_attempted)
    _tool_call_log.append({
        "tool": "escalate_to_compliance",
        "reason": reason,
        "question": unanswered_question,
    })
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
    llm = Ollama(
        model=settings.llm_model,
        base_url=settings.active_ollama_url,
        request_timeout=float(settings.active_request_timeout),
        temperature=settings.llm_temperature,
    )
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


# ============================================================
# Response parser
# ============================================================

def parse_agent_response(raw_response: str) -> dict:
    """Parse agent's JSON response. Handles code fences, embedded JSON, fallback."""
    raw = raw_response.strip()
    json_str = _extract_json(raw)

    if json_str:
        try:
            parsed = json.loads(json_str)
            return {
                "answer": parsed.get("answer", ""),
                "citations": parsed.get("citations", []),
                "escalation": parsed.get("escalation", {"needed": False, "reason": ""}),
                "parse_success": True,
                "raw_response": raw,
            }
        except json.JSONDecodeError:
            pass

    return {
        "answer": raw,
        "citations": [],
        "escalation": {"needed": False, "reason": ""},
        "parse_success": False,
        "raw_response": raw,
    }


def _extract_json(text: str) -> str | None:
    """Extract JSON object from text."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped

    fence_match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
