# AGENT & EVALUATION SYSTEM — Claude Code Instructions

## Summary of Changes

This document describes the full agent + evaluation system. Apply all changes described here.

### What changed from previous version:
1. **Agent returns structured JSON** — `ComplianceAnswer` schema with `answer`, `citations[]`, `escalation`
2. **search_policies output** — split `Clause` and `Clause Number` into separate lines
3. **ask_clarification tool removed** — only 3 tools now
4. **Bot role** — policy lookup only, no interpretation, no advice
5. **Evaluation moved to Phoenix** — Datasets + Experiments API, not local JSON + spans
6. **Evaluators unified** — same retrieval evaluators across all tiers
7. **Agent instrumentation** — tool calls logged to capture actual search queries and results
8. **Citation evaluators split** — doc / doc+section / doc+section+clause levels (mirrors retrieval evaluators)

---

## Part 1: Agent Changes

### 1.1 File: `rag/agent.py` — REPLACE ENTIRELY

```python
from pydantic import BaseModel, Field
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama

from config import settings
from rag.tools.search_policies import search_policies_tool
from rag.tools.get_section import get_section_tool
from rag.tools.escalate import escalate_to_compliance_tool


# ============================================================
# Response schema
# ============================================================


class Citation(BaseModel):
    doc_title: str = Field(description="Full document title exactly as shown in search results")
    section: str = Field(description="Section name exactly as shown in search results")
    clause: str = Field(description="Clause name exactly as shown in search results")
    clause_number: str = Field(description="Clause number exactly as shown in search results, e.g. '4.3'")
    quote: str = Field(description="Exact quote from the policy text that answers the question. Copy verbatim, do not paraphrase.")


class Escalation(BaseModel):
    needed: bool = Field(description="True ONLY if search_policies returned NO_RELEVANT_POLICY_FOUND or the question requires legal interpretation beyond policy text")
    reason: str = Field(default="", description="Why escalation is needed. Empty string if not needed.")


class ComplianceAnswer(BaseModel):
    answer: str = Field(description="Direct answer pointing the user to the relevant policy. Do NOT interpret or paraphrase policy — state what the policy says and where to find it.")
    citations: list[Citation] = Field(description="One or more policy sources. Copy doc_title, section, clause, clause_number exactly from search results.")
    escalation: Escalation = Field(description="Set needed=true only if no relevant policy was found.")


# ============================================================
# Prompt builder
# ============================================================

_INSTRUCTION = """\
You are a Compliance Policy Lookup Assistant.

You have access to tools that search internal policy documents. You do NOT have any policy knowledge built in. You MUST use the search_policies tool to find answers.

WORKFLOW — follow these steps for EVERY question:
1. ALWAYS call search_policies first. Never skip this step. Never answer without searching.
2. Read the search results. Each result has: Document, Section, Clause, Clause Number, and Text.
3. Optionally call get_section if you need the full section text for context.
4. Format your final answer as JSON (schema below) using ONLY information from the search results.
5. If search_policies returns NO_RELEVANT_POLICY_FOUND, set escalation.needed=true and call escalate_to_compliance.
6. If the question spans multiple policy areas, call search_policies multiple times with different queries.

RULES:
- ALWAYS respond in English.
- NEVER answer from general knowledge. ONLY use retrieved policy text.
- NEVER interpret, paraphrase, or add opinion to policy text. Quote it exactly.
- Copy doc_title, section, clause, clause_number exactly as they appear in search results.
- Your FINAL response (after all tool calls) MUST be valid JSON matching the schema below. No markdown, no extra text — only JSON.
- During tool-calling steps you may think freely, but the LAST message must be pure JSON."""

_SCHEMA = """\
Your final answer must be valid JSON strictly following this schema:
```
{schema}
```"""

_EXAMPLE = """\
EXAMPLE of a complete interaction:

User: "Can employees install personal software on company laptops?"

Step 1 — You call search_policies with query: "install software company laptop"
Step 2 — You read the results and find relevant policy
Step 3 — Your FINAL response (pure JSON, no other text):

{{
  "answer": "According to the Acceptable Use Policy, Section 4 (Corporate Workstation and Software Use), Clause 4.2 (Software Installation): Team Members are forbidden from installing unlicensed or unauthorized software on corporate devices. Requests for software must be approved by the SOC or IT Infrastructure team.",
  "citations": [
    {{
      "doc_title": "Acceptable Use Policy [Internal]",
      "section": "Corporate Workstation and Software Use",
      "clause": "Software Installation",
      "clause_number": "4.2",
      "quote": "Team Members are forbidden from installing unlicensed, unauthorized software, including browser toolbars, extensions, peer-to-peer (P2P) software, or games on Company corporate devices."
    }}
  ],
  "escalation": {{
    "needed": false,
    "reason": ""
  }}
}}"""


def build_system_prompt() -> str:
    """Build the system prompt with instruction, schema, and example."""
    import json
    schema_str = ComplianceAnswer.model_json_schema()
    schema_formatted = json.dumps(schema_str, indent=2)
    delimiter = "\n\n---\n\n"
    parts = [
        _INSTRUCTION.strip(),
        delimiter,
        _SCHEMA.format(schema=schema_formatted).strip(),
        delimiter,
        _EXAMPLE.strip(),
    ]
    return "".join(parts)


SYSTEM_PROMPT = build_system_prompt()


# ============================================================
# Agent builder
# ============================================================

ALL_TOOLS = [
    search_policies_tool,
    get_section_tool,
    escalate_to_compliance_tool,
]


def get_llm() -> Ollama:
    return Ollama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        request_timeout=float(settings.llm_request_timeout),
        temperature=settings.llm_temperature,
    )


def build_agent() -> AgentWorkflow:
    """Build a ReAct agent with compliance tools."""
    llm = get_llm()
    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=ALL_TOOLS,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=True,
    )
    return agent
```

### 1.2 File: `rag/tools/__init__.py` — REPLACE ENTIRELY

Removed `ask_clarification`. Only 3 tools now:

```python
from rag.tools.search_policies import search_policies_tool
from rag.tools.get_section import get_section_tool
from rag.tools.escalate import escalate_to_compliance_tool

ALL_TOOLS = [
    search_policies_tool,
    get_section_tool,
    escalate_to_compliance_tool,
]
```

### 1.3 File: `rag/tools/search_policies.py` — REPLACE ENTIRELY

Key change: output format now has separate `Clause:` and `Clause Number:` lines so the agent can copy them directly into JSON citations.

```python
from llama_index.core.tools import FunctionTool

from config import settings


def search_policies(query: str, top_k: int = 6) -> str:
    """
    Search approved compliance policy documents for information relevant to the query.
    Uses hybrid search (semantic + keyword matching) for best accuracy.

    Args:
        query: Natural language search query describing what policy info you need.
               Be specific — include relevant terms, clause numbers, or policy names.
        top_k: Number of most relevant policy sections to return (default 6).

    Returns:
        Formatted policy excerpts with document name, section, clause, clause number,
        and full text. Returns "NO_RELEVANT_POLICY_FOUND" if no policies match.
    """
    if settings.bm25_enabled:
        from rag.hybrid_search import hybrid_search

        results = hybrid_search(query=query, top_k=top_k)

        if not results or results[0].get("rrf_score", 0) < settings.min_confidence_score:
            return "NO_RELEVANT_POLICY_FOUND"

        formatted = []
        for i, r in enumerate(results, 1):
            lines = [
                f"--- Result {i} [RRF Score: {r['rrf_score']:.4f}] ---",
                f"Document: {r['doc_title']}",
                f"Section: {r.get('section', '')}",
                f"Clause: {r.get('clause', '')}",
                f"Clause Number: {r.get('clause_number', '')}",
                f"Doc ID: {r['doc_id']}",
                f"Text: {r['text']}",
            ]
            formatted.append("\n".join(lines))

        return "\n\n".join(formatted)

    # Fallback: vector-only search
    from rag.embeddings import embed_query
    from rag.vector_store import search_vectors

    query_vector = embed_query(query)
    results = search_vectors(query_vector, top_k=top_k)

    if not results or results[0].score < settings.min_confidence_score:
        return "NO_RELEVANT_POLICY_FOUND"

    formatted = []
    for i, r in enumerate(results, 1):
        p = r.payload
        lines = [
            f"--- Result {i} [Score: {r.score:.4f}] ---",
            f"Document: {p['doc_title']}",
            f"Section: {p.get('section', '')}",
            f"Clause: {p.get('clause', '')}",
            f"Clause Number: {p.get('clause_number', '')}",
            f"Doc ID: {p['doc_id']}",
            f"Text: {p['text']}",
        ]
        formatted.append("\n".join(lines))

    return "\n\n".join(formatted)


search_policies_tool = FunctionTool.from_defaults(fn=search_policies)
```

### 1.4 Files NOT changed
- `rag/tools/get_section.py` — unchanged
- `rag/tools/escalate.py` — unchanged
- `rag/tools/clarify.py` — kept on disk but no longer imported or used

---

## Part 2: Evaluation System

### 2.1 File: `eval/__init__.py` — CREATE

```python
# Evaluation package for Compliance Q&A Bot
```

### 2.2 File: `eval/evaluators.py` — CREATE

All evaluator functions shared across tiers.

```python
"""
Shared evaluators for all evaluation tiers.

Each evaluator receives (output, expected) and returns
{"score": float, "label": str, "explanation": str}.

Task output format (Tier 2 / Chatbot):
  {
      "answer": str,
      "citations": [{"doc_title", "section", "clause", "clause_number", "quote"}],
      "escalation": {"needed": bool, "reason": str},
      "search_results": [{"doc_title", "section", "clause", ...}],
      "agent_metadata": {"search_queries", "num_searches", ...},
      "parse_success": bool,
      "raw_response": str,
  }

Task output format (Tier 1):
  {
      "search_results": [{"doc_title", "section", "clause", ...}],
  }
"""


# ============================================================
# Helpers
# ============================================================

def _extract_expected(expected: dict) -> list[dict]:
    """Normalize expected format from any tier.
    Tier 1: {"expected_doc", "expected_section", "expected_clause"}
    Tier 2/Chat: {"expected_citations": [{"doc_id", "section", "clause"}]}
    Returns: [{"doc", "section", "clause"}, ...]
    """
    if "expected_citations" in expected and expected["expected_citations"]:
        return [
            {
                "doc": cite.get("doc_id", "").strip(),
                "section": cite.get("section", "").strip(),
                "clause": cite.get("clause", "").strip(),
            }
            for cite in expected["expected_citations"]
        ]
    return [{
        "doc": expected.get("expected_doc", "").strip(),
        "section": expected.get("expected_section", "").strip(),
        "clause": expected.get("expected_clause", "").strip(),
    }]


def _get_search_results(output):
    if output is None:
        return None
    return output.get("search_results", [])


def _match_result(r, exp):
    doc_match = not exp["doc"] or exp["doc"].lower() == r.get("doc_title", "").strip().lower()
    section_match = not exp["section"] or exp["section"].lower() in r.get("section", "").strip().lower()
    clause_match = not exp["clause"] or exp["clause"].lower() in r.get("clause", "").strip().lower()
    return doc_match and section_match and clause_match


# ============================================================
# RETRIEVAL EVALUATORS
# Source: output["search_results"]
# Tier 1: from hybrid_search(user_question)
# Tier 2/Chatbot: from agent's actual search_policies tool calls
# ============================================================

def hit_evaluator(output, expected):
    """Did ANY retrieved chunk match expected doc+section+clause?"""
    results = _get_search_results(output)
    if results is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    expectations = _extract_expected(expected)
    for exp in expectations:
        found = False
        for rank, r in enumerate(results, start=1):
            if _match_result(r, exp):
                found = True
                break
        if not found:
            return {"score": 0.0, "label": "miss",
                    "explanation": f"Expected: {exp['doc']} > {exp['section']} > {exp['clause']}"}
    return {"score": 1.0, "label": f"hit@{rank}",
            "explanation": f"Found at rank {rank}: {r.get('doc_title', '')} > {r.get('section', '')} > {r.get('clause', '')}"}


def mrr_evaluator(output, expected):
    """Mean Reciprocal Rank — 1/rank of first full match."""
    results = _get_search_results(output)
    if results is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    expectations = _extract_expected(expected)
    best_rr, best_rank = 0.0, None
    for exp in expectations:
        for rank, r in enumerate(results, start=1):
            if _match_result(r, exp):
                rr = 1.0 / rank
                if rr > best_rr:
                    best_rr, best_rank = rr, rank
                break
    if best_rank:
        return {"score": round(best_rr, 4), "label": f"rank_{best_rank}",
                "explanation": f"First hit at rank {best_rank}, MRR = {best_rr:.4f}"}
    return {"score": 0.0, "label": "miss", "explanation": "No matching chunk found"}


def retrieval_doc_hit(output, expected):
    """Did search results contain the expected DOCUMENT?"""
    results = _get_search_results(output)
    if results is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    expectations = _extract_expected(expected)
    hits, misses = [], []
    for exp in expectations:
        if not exp["doc"]:
            continue
        found = any(exp["doc"].lower() == r.get("doc_title", "").strip().lower() for r in results)
        (hits if found else misses).append(exp["doc"])
    total = len(hits) + len(misses)
    score = len(hits) / total if total else 1.0
    return {"score": round(score, 4), "label": f"{len(hits)}/{total}",
            "explanation": "All docs retrieved" if not misses else f"Missing: {misses}"}


def retrieval_section_hit(output, expected):
    """Did search results contain the expected doc+section pair?"""
    results = _get_search_results(output)
    if results is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    expectations = _extract_expected(expected)
    hits, misses = [], []
    for exp in expectations:
        if not exp["section"]:
            continue
        found = any(
            (not exp["doc"] or exp["doc"].lower() == r.get("doc_title", "").strip().lower())
            and exp["section"].lower() in r.get("section", "").strip().lower()
            for r in results)
        key = f"{exp['doc']} > {exp['section']}"
        (hits if found else misses).append(key)
    total = len(hits) + len(misses)
    score = len(hits) / total if total else 1.0
    return {"score": round(score, 4), "label": f"{len(hits)}/{total}",
            "explanation": "All sections retrieved" if not misses else f"Missing: {misses}"}


# ============================================================
# GENERATION EVALUATORS
# Source: output["answer"] and output["citations"]
# Used in: Tier 2, Chatbot
# ============================================================

def answer_coverage(output, expected):
    """What fraction of expected answer items appear in the response?
    For each item, checks if >=50% of significant words (>3 chars) appear in answer.
    """
    if output is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    answer = output.get("answer", "").lower()
    expected_items = expected.get("expected_answer", [])
    if isinstance(expected_items, str):
        expected_items = [expected_items] if expected_items else []
    if not expected_items:
        return {"score": 1.0, "label": "skip", "explanation": "No expected items"}
    hits, misses = [], []
    for item in expected_items:
        words = [w for w in item.lower().split() if len(w) > 3]
        if not words:
            hits.append(item)
            continue
        matched = sum(1 for w in words if w in answer)
        (hits if matched / len(words) >= 0.5 else misses).append(item)
    score = len(hits) / len(expected_items)
    return {"score": round(score, 4), "label": f"{len(hits)}/{len(expected_items)}",
            "explanation": f"All {len(hits)} items covered" if not misses else f"Misses: {[m[:60] for m in misses[:3]]}..."}


def citation_doc_accuracy(output, expected):
    """Did the agent's JSON citations reference the correct DOCUMENT?"""
    if output is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    agent_citations = output.get("citations", [])
    expectations = _extract_expected(expected)
    if not expectations or not expectations[0]["doc"]:
        return {"score": 1.0, "label": "skip", "explanation": "No expected doc"}
    if not agent_citations:
        return {"score": 0.0, "label": "no_citations", "explanation": "Agent returned no citations"}
    hits, misses = [], []
    for exp in expectations:
        if not exp["doc"]:
            continue
        found = any(exp["doc"].lower() == c.get("doc_title", "").strip().lower() for c in agent_citations)
        (hits if found else misses).append(exp["doc"])
    total = len(hits) + len(misses)
    score = len(hits) / total if total else 1.0
    return {"score": round(score, 4), "label": f"{len(hits)}/{total}",
            "explanation": "All docs cited" if not misses else f"Missing: {misses}"}


def citation_section_accuracy(output, expected):
    """Did the agent's JSON citations reference the correct DOCUMENT + SECTION?"""
    if output is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    agent_citations = output.get("citations", [])
    expectations = _extract_expected(expected)
    if not expectations or not expectations[0]["doc"]:
        return {"score": 1.0, "label": "skip", "explanation": "No expected citations"}
    if not agent_citations:
        return {"score": 0.0, "label": "no_citations", "explanation": "Agent returned no citations"}
    hits, misses = [], []
    for exp in expectations:
        if not exp["section"]:
            continue
        found = any(
            (not exp["doc"] or exp["doc"].lower() == c.get("doc_title", "").strip().lower())
            and exp["section"].lower() in c.get("section", "").strip().lower()
            for c in agent_citations)
        key = f"{exp['doc']} > {exp['section']}"
        (hits if found else misses).append(key)
    total = len(hits) + len(misses)
    score = len(hits) / total if total else 1.0
    return {"score": round(score, 4), "label": f"{len(hits)}/{total}",
            "explanation": "All sections cited" if not misses else f"Missing: {misses}"}


def citation_clause_accuracy(output, expected):
    """Did the agent's JSON citations reference the correct DOC + SECTION + CLAUSE?
    Skips if no expected clause specified."""
    if output is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    agent_citations = output.get("citations", [])
    expectations = _extract_expected(expected)
    has_clause = any(exp.get("clause") for exp in expectations)
    if not has_clause:
        return {"score": 1.0, "label": "skip", "explanation": "No expected clauses in dataset"}
    if not agent_citations:
        return {"score": 0.0, "label": "no_citations", "explanation": "Agent returned no citations"}
    hits, misses = [], []
    for exp in expectations:
        if not exp["clause"]:
            continue
        found = any(
            (not exp["doc"] or exp["doc"].lower() == c.get("doc_title", "").strip().lower())
            and (not exp["section"] or exp["section"].lower() in c.get("section", "").strip().lower())
            and exp["clause"].lower() in c.get("clause", "").strip().lower()
            for c in agent_citations)
        key = f"{exp['doc']} > {exp['section']} > {exp['clause']}"
        (hits if found else misses).append(key)
    total = len(hits) + len(misses)
    score = len(hits) / total if total else 1.0
    return {"score": round(score, 4), "label": f"{len(hits)}/{total}",
            "explanation": "All clauses cited" if not misses else f"Missing: {misses}"}


def json_parse_success(output, expected):
    """Did the agent return valid JSON matching the ComplianceAnswer schema?"""
    if output is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    success = output.get("parse_success", False)
    if success:
        return {"score": 1.0, "label": "valid_json", "explanation": "Valid JSON"}
    raw = output.get("raw_response", "")[:100]
    return {"score": 0.0, "label": "invalid_json", "explanation": f"Not valid JSON. Start: {raw}..."}


# ============================================================
# AGENT BEHAVIOR EVALUATORS
# Source: output["agent_metadata"]
# Used in: Tier 2, Chatbot
# ============================================================

def agent_search_count(output, expected):
    """How many search calls? 0=hallucinated, 1-3=efficient, 4+=thrashing."""
    if output is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    meta = output.get("agent_metadata", {})
    n = meta.get("num_searches", 0)
    if n == 0:
        return {"score": 0.0, "label": "no_search", "explanation": "Agent never searched — hallucinated"}
    elif 1 <= n <= 3:
        return {"score": 1.0, "label": f"{n}_searches", "explanation": f"{n} search(es) — efficient"}
    else:
        return {"score": 0.5, "label": f"{n}_searches", "explanation": f"{n} searches — thrashing"}


def agent_used_get_section(output, expected):
    """Did agent call get_section? Optional — 0.5 if skipped, not penalized."""
    if output is None:
        return {"score": 0.0, "label": "error", "explanation": "Task returned None"}
    meta = output.get("agent_metadata", {})
    n = meta.get("num_section_fetches", 0)
    fetches = meta.get("section_fetches", [])
    if n == 0:
        return {"score": 0.5, "label": "skipped", "explanation": "Skipped get_section (optional)"}
    all_found = all(f["found"] for f in fetches)
    return {"score": 1.0 if all_found else 0.5, "label": f"{n}_fetches",
            "explanation": f"Fetched {n} section(s), all found: {all_found}"}


# ============================================================
# EVALUATOR GROUPS
# ============================================================

RETRIEVAL_EVALUATORS = [
    hit_evaluator,
    mrr_evaluator,
    retrieval_doc_hit,
    retrieval_section_hit,
]

GENERATION_EVALUATORS = [
    answer_coverage,
    citation_doc_accuracy,
    citation_section_accuracy,
    citation_clause_accuracy,
    json_parse_success,
]

AGENT_EVALUATORS = [
    agent_search_count,
    agent_used_get_section,
]

TIER1_EVALUATORS = RETRIEVAL_EVALUATORS
TIER2_EVALUATORS = RETRIEVAL_EVALUATORS + GENERATION_EVALUATORS + AGENT_EVALUATORS
CHATBOT_EVALUATORS = RETRIEVAL_EVALUATORS + GENERATION_EVALUATORS + AGENT_EVALUATORS
```

### 2.3 File: `eval/agent_wrapper.py` — CREATE

Instrumented agent with tool call logging and JSON response parsing.

```python
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
        base_url=settings.ollama_base_url,
        request_timeout=float(settings.llm_request_timeout),
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
```

### 2.4 File: `eval/run_experiment.py` — CREATE

CLI runner for all tiers:

```python
#!/usr/bin/env python3
"""
Run a Phoenix evaluation experiment.

Usage:
    python eval/run_experiment.py --tier tier1 --name baseline-hybrid-v1
    python eval/run_experiment.py --tier tier2 --name baseline-e2e-v1
    python eval/run_experiment.py --tier chatbot --name baseline-chatbot-v1
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_async():
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("WARNING: pip install nest_asyncio")


def make_tier1_task(top_k: int = 6):
    from rag.hybrid_search import hybrid_search

    def retrieval_task(input):
        results = hybrid_search(input["question"], top_k=top_k)
        return {
            "search_results": [
                {
                    "doc_title": r["doc_title"],
                    "section": r["section"],
                    "clause": r.get("clause", ""),
                    "clause_number": r.get("clause_number", ""),
                    "rrf_score": round(r["rrf_score"], 4),
                }
                for r in results
            ]
        }
    return retrieval_task


def make_agent_task(verbose: bool = False):
    from eval.agent_wrapper import build_instrumented_agent, get_log, clear_log, parse_agent_response

    async def _run_fresh_agent(question, verbose):
        agent = build_instrumented_agent(verbose=verbose)
        return await agent.run(question)

    def e2e_task(input):
        question = input["question"]
        clear_log()

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_run_fresh_agent(question, verbose))

        parsed = parse_agent_response(str(result))
        tool_calls = list(get_log())

        agent_search_results, search_queries = [], []
        for call in tool_calls:
            if call["tool"] == "search_policies":
                search_queries.append(call["query"])
                agent_search_results.extend(call["results"])

        seen = set()
        unique_results = []
        for r in agent_search_results:
            key = (r["doc_title"], r["section"], r["clause"])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        section_calls = [c for c in tool_calls if c["tool"] == "get_section"]
        escalation_calls = [c for c in tool_calls if c["tool"] == "escalate_to_compliance"]

        return {
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "escalation": parsed["escalation"],
            "parse_success": parsed["parse_success"],
            "raw_response": parsed["raw_response"],
            "search_results": unique_results,
            "agent_metadata": {
                "search_queries": search_queries,
                "num_searches": len(search_queries),
                "num_section_fetches": len(section_calls),
                "section_fetches": [
                    {"doc_id": c["doc_id"], "section": c["section_name"], "found": c["found"]}
                    for c in section_calls
                ],
                "escalated": len(escalation_calls) > 0,
                "escalation_reason": escalation_calls[0]["reason"] if escalation_calls else None,
            },
        }
    return e2e_task


TIER_CONFIG = {
    "tier1": {"default_dataset": "retrieval-test-v1", "description": "Retrieval: hybrid search"},
    "tier2": {"default_dataset": "e2e-test-v1", "description": "E2E: full agent + structured JSON"},
    "chatbot": {"default_dataset": "chatbot-test-v1", "description": "Chatbot: realistic user questions"},
}


def main():
    parser = argparse.ArgumentParser(description="Run Phoenix evaluation experiment.")
    parser.add_argument("--tier", choices=["tier1", "tier2", "chatbot"], required=True)
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--phoenix-url", default=None)
    args = parser.parse_args()

    setup_async()
    from phoenix.client import Client
    from eval.evaluators import TIER1_EVALUATORS, TIER2_EVALUATORS, CHATBOT_EVALUATORS
    from config import settings

    tier_cfg = TIER_CONFIG[args.tier]
    dataset_name = args.dataset or tier_cfg["default_dataset"]
    description = args.description or tier_cfg["description"]

    client_kwargs = {}
    if args.phoenix_url:
        client_kwargs["endpoint"] = args.phoenix_url
    client = Client(**client_kwargs)

    try:
        dataset = client.datasets.get_dataset(dataset=dataset_name)
    except Exception:
        print(f"ERROR: Dataset '{dataset_name}' not found.")
        print(f"  Create: python scripts/make_dataset.py eval/datasets/<file>.json")
        sys.exit(1)

    print(f"  Tier:        {args.tier}")
    print(f"  Dataset:     {dataset.name} ({len(dataset)} examples)")
    print(f"  Experiment:  {args.name}")

    if args.tier == "tier1":
        task = make_tier1_task(top_k=args.top_k)
        evaluators = TIER1_EVALUATORS
        metadata = {"search_type": "hybrid_rrf", "embedding_model": "nomic-embed-text",
                     "top_k": args.top_k, "tier": "tier1"}
    else:
        task = make_agent_task(verbose=args.verbose)
        evaluators = TIER2_EVALUATORS if args.tier == "tier2" else CHATBOT_EVALUATORS
        metadata = {"llm": settings.llm_model, "search_type": "hybrid_rrf",
                     "agent_type": "react", "top_k": args.top_k, "tier": args.tier,
                     "structured_output": True}

    print(f"  Evaluators:  {[e.__name__ for e in evaluators]}")

    experiment = client.experiments.run_experiment(
        dataset=dataset, task=task, evaluators=evaluators,
        experiment_name=args.name, experiment_description=description,
        experiment_metadata=metadata,
    )
    print(f"\n  Done: {args.name}")
    print(f"  View: http://localhost:6006/datasets")


if __name__ == "__main__":
    main()
```

---

## Part 3: Dataset Management

### 3.1 File: `scripts/make_dataset.py`

Already created in previous step. No changes needed. Handles all 4 tiers.

---

## Part 4: Evaluator Reference

### Full evaluator matrix — retrieval vs citation (parallel structure)

| Level | Retrieval layer | Citation layer | What it measures |
|---|---|---|---|
| Document only | `retrieval_doc_hit` | `citation_doc_accuracy` | Right policy document? |
| Doc + Section | `retrieval_section_hit` | `citation_section_accuracy` | Right section? |
| Doc + Section + Clause | `hit_evaluator` | `citation_clause_accuracy` | Right clause? (skips if no expected clause) |
| Rank position | `mrr_evaluator` | — | How high is the first match? |
| Answer content | — | `answer_coverage` | Expected info in response? |
| Output format | — | `json_parse_success` | Valid JSON? |
| Tool usage | — | `agent_search_count` | 0=hallucinated, 1-3=good, 4+=thrashing |
| Section fetch | — | `agent_used_get_section` | Called get_section? (optional) |

### Evaluator grouping per tier

| Tier | Evaluators |
|---|---|
| `TIER1_EVALUATORS` | `hit_evaluator`, `mrr_evaluator`, `retrieval_doc_hit`, `retrieval_section_hit` |
| `TIER2_EVALUATORS` | All retrieval + `answer_coverage`, `citation_doc_accuracy`, `citation_section_accuracy`, `citation_clause_accuracy`, `json_parse_success`, `agent_search_count`, `agent_used_get_section` |
| `CHATBOT_EVALUATORS` | Same as `TIER2_EVALUATORS` |

### When expected clause is missing

Both `hit_evaluator` and `citation_clause_accuracy` handle this:
- `_match_result` treats empty `exp["clause"]` as "match any clause" (clause_match = True)
- `citation_clause_accuracy` returns `{"score": 1.0, "label": "skip"}` when no expected clauses exist in the dataset

---

## Part 5: Running

```bash
# 1. Upload datasets (one-time or after updating test cases)
python scripts/make_dataset.py eval/datasets/retrieval_test.json --overwrite
python scripts/make_dataset.py eval/datasets/e2e_test.json --overwrite
python scripts/make_dataset.py eval/datasets/chatbot_test_cases.json

# 2. Run experiments
python eval/run_experiment.py --tier tier1 --name structured-v1-retrieval
python eval/run_experiment.py --tier tier2 --name structured-v1-e2e
python eval/run_experiment.py --tier chatbot --name structured-v1-chatbot

# 3. After changes (new prompt, reranker, etc)
python eval/run_experiment.py --tier tier2 --name prompt-v2-e2e
```

---

## Part 6: Known Issues

### Agent skipping tools
Qwen2.5-14B sometimes skips tools and hallucinates. Detected by `agent_search_count=0` and `json_parse_success=0`. The structured prompt with explicit example should reduce this.

### Tool name mismatch
`FunctionTool.from_defaults(fn=..., name="search_policies")` — the `name=` MUST match system prompt. If names differ, LLM won't call tools.

### Fresh agent per question
`build_instrumented_agent()` creates a new agent each time. Reusing causes state leakage.

### nest_asyncio
Required for Phoenix sync runner + LlamaIndex async agent. `run_experiment.py` calls it automatically.
