"""
Non-agentic (vanilla) RAG pipeline.
Single search + single LLM call. No LlamaIndex, no ReAct, no tools.

Usage:
    from rag.pipeline import run_query
    result = run_query("What is the policy on software installation?")
"""

import json
import time
import httpx

from config import settings
from rag.tools.search_policies import search_policies
from rag.agent import SYSTEM_PROMPT
from rag.observability import get_tracer


def run_query(question: str) -> dict:
    """
    Full pipeline: search → rerank → LLM → structured answer.

    Returns dict matching ComplianceAnswer schema:
    {
        "answer": str,
        "citations": [{"source_number", "doc_title", "section", "clause", "clause_number", "quote"}],
        "escalation": {"needed": bool, "reason": str}
    }
    """
    tracer = get_tracer()

    with tracer.start_as_current_span("vanilla_rag_pipeline") as span:
        span.set_attribute("question", question)
        span.set_attribute("pipeline_mode", "vanilla")
        start = time.time()

        # Step 1: Search + rerank
        with tracer.start_as_current_span("search_policies") as search_span:
            sources = search_policies(question)
            search_span.set_attribute("no_results", "NO_RELEVANT_POLICY_FOUND" in sources)

        # Step 2: Programmatic escalation if nothing found
        if "NO_RELEVANT_POLICY_FOUND" in sources:
            span.set_attribute("escalated", True)
            span.set_attribute("latency_ms", int((time.time() - start) * 1000))
            return {
                "answer": "",
                "citations": [],
                "escalation": {"needed": True, "reason": "No relevant policy found."},
            }

        # Step 3: Build prompt and make single LLM call
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{sources}\n\nQuestion: {question}"},
        ]
        with tracer.start_as_current_span("llm_call") as llm_span:
            llm_span.set_attribute("model", settings.llm_model)
            llm_response = _call_ollama(messages)
            llm_span.set_attribute("response_length", len(llm_response))

        # Step 4: Parse JSON response
        result = _parse_response(llm_response)
        span.set_attribute("escalated", result.get("escalation", {}).get("needed", False))
        span.set_attribute("num_citations", len(result.get("citations", [])))
        span.set_attribute("latency_ms", int((time.time() - start) * 1000))
        return result


def _call_ollama(messages: list[dict]) -> str:
    """Call Ollama chat API. Returns raw text response."""
    resp = httpx.post(
        f"{settings.active_ollama_url}/api/chat",
        json={
            "model": settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature,
            },
        },
        timeout=float(settings.active_request_timeout),
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_response(raw: str) -> dict:
    """Parse LLM JSON output. Handles code fences and malformed output."""
    text = raw.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    if text.startswith("json"):
        text = text[4:].strip()

    # Find JSON object
    start = text.find("{")
    if start == -1:
        return _fallback(raw)

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                break
    else:
        return _fallback(raw)

    try:
        parsed = json.loads(json_str)
        return {
            "answer": parsed.get("answer", ""),
            "citations": parsed.get("citations", []),
            "escalation": parsed.get("escalation", {"needed": False, "reason": ""}),
        }
    except (json.JSONDecodeError, Exception):
        return _fallback(raw)


def _fallback(raw: str) -> dict:
    """Fallback when JSON parsing fails."""
    return {
        "answer": raw,
        "citations": [],
        "escalation": {"needed": True, "reason": "Failed to parse structured response."},
    }
