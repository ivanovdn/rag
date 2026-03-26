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
