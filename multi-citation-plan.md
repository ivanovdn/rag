# Multi-Citation Support — Implementation Plan

## Goal

Agent should cite ALL relevant policy sources, not just one. Eval should accept any valid citation.

---

## Part 1 — Prompt Changes

**File: `rag/agent.py`** — modify `SYSTEM_PROMPT`

### Change 1: Update item 3 in `== HOW TO RESPOND ==`

Find:
```
3. Identify which source(s) directly answer the question.
4. Quote the relevant policy text VERBATIM in your answer — copy the exact words from the source.
```

Replace with:
```
3. Identify ALL sources that address the question — not just the first match. Multiple policies often cover the same topic. Cite every relevant source.
4. Quote the relevant policy text VERBATIM from EACH cited source — copy the exact words.
```

### Change 2: Replace the answer format example in `== ANSWER FORMAT ==`

Find the current CORRECT example block. Replace with:

```
CORRECT example (single source):
"This is addressed in [Policy Name], Section: [Section Name], Clause [Number] ([Clause Name]): '[verbatim quote from the policy text].'"

CORRECT example (multiple related sources):
"This is addressed in multiple policies:
1. [Policy A], Section: [Section X], Clause [N] ([Clause Name]): '[verbatim quote from policy A].'
2. [Policy B], Section: [Section Y], Clause [M] ([Clause Name]): '[verbatim quote from policy B].'"
```

### Change 3: Add rule about multiple citations

In the `== RULES ==` section, find:
```
- NEVER cite a source you did not use in your answer.
```

Add AFTER it:
```
- If 2 or more sources address the question, cite ALL of them. Each citation must have its own entry in the citations array with its own verbatim quote.
```

### Change 4: Update the JSON example in `== OUTPUT FORMAT ==`

Find the current JSON example. Replace with:

```json
{
  "answer": "This is addressed in multiple policies: 1. [Document A], Section: [X], Clause [N]: '[quote]'. 2. [Document B], Section: [Y], Clause [M]: '[quote]'.",
  "citations": [
    {
      "source_number": 1,
      "doc_title": "exact document title from source header",
      "section": "exact section name from source header",
      "clause": "exact clause name from source header",
      "clause_number": "e.g. 4.7",
      "quote": "verbatim text copied from the source"
    },
    {
      "source_number": 2,
      "doc_title": "second document title from source header",
      "section": "exact section name from source header",
      "clause": "exact clause name from source header",
      "clause_number": "e.g. 8.7",
      "quote": "verbatim text copied from the second source"
    }
  ],
  "escalation": {"needed": false, "reason": ""}
}
```

---

## Part 2 — Eval Changes

### File: `eval/evaluators.py`

Add `match_mode` support to all three citation evaluators. When `match_mode: "any"`, agent passes if it cited ANY one of the expected citations.

#### Change `citation_doc_accuracy`:

Find the loop that checks expected citations. Wrap it with match_mode logic:

```python
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

    match_mode = expected.get("match_mode", "all")

    if match_mode == "any":
        for exp in expectations:
            if not exp["doc"]:
                continue
            found = any(exp["doc"].lower() == c.get("doc_title", "").strip().lower() for c in agent_citations)
            if found:
                return {"score": 1.0, "label": "any_hit",
                        "explanation": f"Matched: {exp['doc']}"}
        return {"score": 0.0, "label": "any_miss",
                "explanation": f"None of {len(expectations)} valid docs were cited"}

    # Default: all mode (existing behavior)
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
```

#### Apply the same pattern to `citation_section_accuracy` and `citation_clause_accuracy`.

The logic is identical — add `match_mode` check, if "any" return 1.0 on first match.

#### Also update `hit_evaluator` and `mrr_evaluator`:

For `hit_evaluator` with `match_mode: "any"`:
- Current: requires ALL expected citations found in search results
- New: pass if ANY expected citation found in search results

For `mrr_evaluator` with `match_mode: "any"`:
- Current: best reciprocal rank across all expected citations
- New: same behavior (already takes best rank, no change needed)

---

## Part 3 — Test Case Updates

### File: `eval/datasets/chatbot_test_cases.json`

Review the ~13 failing cases. For each one where the agent cited a different but valid source:

1. Add the alternative citation to `expected_citations`
2. Add `"match_mode": "any"`

Example — journalist question:

Before:
```json
{
  "question": "A journalist contacted me asking about the incident...",
  "expected_citations": [
    {
      "doc_id": "Business Continuity and Disaster Recovery (BCDR) Plan [Internal]",
      "section": "Communication and Coordination",
      "clause": "Media Communication"
    }
  ]
}
```

After:
```json
{
  "question": "A journalist contacted me asking about the incident...",
  "expected_citations": [
    {
      "doc_id": "Business Continuity and Disaster Recovery (BCDR) Plan [Internal]",
      "section": "Communication and Coordination",
      "clause": "Media Communication"
    },
    {
      "doc_id": "Code Of Ethics & Conduct [Public]",
      "section": "External Communication and Outside Activities",
      "clause": "Public Statements"
    }
  ],
  "match_mode": "any"
}
```

### Cases to update (from eval failures):

| Question | Current expected | Add alternative |
|----------|-----------------|-----------------|
| Client walk around office | Physical Security > Escorting Visitors | (keep, fix is clause name paraphrase) |
| Data not sensitive, report? | Data Breach > Initial Reporting | Data Breach > Notification to Authority |
| System stay logged in | Clear Desk > Computers | Access Management > Inactivity Logoff/Lockout |
| Skip approvals internal tools | Open Source > Approval | Secure Development > Tools and Third-Party Software |
| Journalist asking about incident | BCDR > Media Communication | Code of Ethics > Public Statements |

Review remaining ~8 failing cases manually and add alternatives where the agent's answer was valid.

---

## Verification

```bash
# Test multi-citation behavior
PYTHONPATH=. python scripts/test_query.py -q "A journalist contacted me asking about the incident. Can I give a short statement?"

# Check: agent should now cite BOTH Code of Ethics AND BCDR Plan

# Run full eval
PYTHONPATH=. python eval/run_experiment.py --tier chatbot --name multi-citation-v1

# Expected improvements:
# citation_clause_accuracy: 0.78 → 0.85-0.90
# citation_doc_accuracy: 0.86 → 0.90-0.95
```

---

## File Changes

| File | Change |
|------|--------|
| `rag/agent.py` | Update SYSTEM_PROMPT: multi-citation examples, cite-all rule |
| `eval/evaluators.py` | Add match_mode="any" support to all citation evaluators |
| `eval/datasets/chatbot_test_cases.json` | Add alternative valid citations + match_mode to failing cases |

---
