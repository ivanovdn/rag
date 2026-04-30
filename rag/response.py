"""Parse agent's text output into a ComplianceAnswer dict.

Handles code fences, embedded JSON, and falls back gracefully when
the agent returns plain text.
"""

import json
import re


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
