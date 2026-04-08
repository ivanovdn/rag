# Critical Fixes — Chatbot Eval Pipeline

Three bugs found. Fix all three before re-running the chatbot eval.

---

## Fix 1 — `eval/agent_wrapper.py`: eval logs wrong search results

### Problem

`_logged_search_policies` calls `hybrid_search()` directly with `top_k=6` for logging, bypassing the reranker entirely. The agent sees correct reranked results (via `_original_search`), but the evaluators judge against unranked, fewer-candidate results. This is why chatbot hit=0.68 while tier1 hit=0.90.

### What to change

**File: `eval/agent_wrapper.py`**

1. Remove this import at the top:
```python
# DELETE THIS LINE:
from rag.hybrid_search import hybrid_search
```

2. Replace the entire `_logged_search_policies` function with:
```python
def _logged_search_policies(query: str, top_k: int = 6) -> str:
    """Logged version — calls the real search pipeline (with reranker) and captures results."""
    from rag.tools.search_policies import _last_search_results

    result_text = _original_search(query, top_k)

    _tool_call_log.append({
        "tool": "search_policies",
        "query": query,
        "top_k": top_k,
        "results": list(_last_search_results),
    })

    return result_text
```

---

## Fix 2 — `rag/tools/search_policies.py`: expose structured results for eval logging

### Problem

The eval wrapper needs access to the structured search results (with reranker scores) that the agent actually sees. Currently `search_policies` only returns a formatted string.

### What to change

**File: `rag/tools/search_policies.py`**

1. Add a module-level variable at the top of the file (after imports):
```python
_last_search_results: list[dict] = []
```

2. Inside the `search_policies` function, capture results right before `format_sources`. Add these lines immediately before the `return format_sources(results)` line:
```python
    global _last_search_results
    _last_search_results = [
        {
            "doc_title": r["doc_title"],
            "section": r.get("section", ""),
            "clause": r.get("clause", ""),
            "clause_number": r.get("clause_number", ""),
            "rerank_score": round(r.get("rerank_score", 0), 4),
            "retrieval_score": round(r.get("retrieval_score", 0), 4),
        }
        for r in results
    ]

    # Step 3: Format for the agent
    return format_sources(results)
```

3. Also handle the `NO_RELEVANT_POLICY_FOUND` early returns. In both places where the function returns `"NO_RELEVANT_POLICY_FOUND"`, add before the return:
```python
    _last_search_results = []
    return "NO_RELEVANT_POLICY_FOUND"
```

---

## Fix 3 — `rag/agent.py`: prevent agent from rewriting search queries

### Problem

The agent rewrites user questions before calling `search_policies`. Example: user asks *"If it's just for internal tools, can I skip approvals?"* but agent searches *"internal tools approvals"*. The rewritten query loses critical context and retrieval quality drops.

### What to change

**File: `rag/agent.py`**

Add the following block to the system prompt, immediately after the `== HOW TO RESPOND ==` section, before item 1:

Find this text in SYSTEM_PROMPT:
```
== HOW TO RESPOND ==

1. Call search_policies FIRST for every question. Never answer without searching.
```

Replace with:
```
== HOW TO RESPOND ==

0. When calling search_policies, pass the user's ORIGINAL question as the query.
   Do NOT rewrite, shorten, extract keywords, or rephrase the question.
   The search system is optimized for natural language questions, not keywords.
   WRONG: search_policies("internal tools approvals")
   CORRECT: search_policies("If it's just for internal tools, can I skip approvals?")
1. Call search_policies FIRST for every question. Never answer without searching.
```

Do NOT change anything else in the system prompt.

---

## Verification

After applying all three fixes:

### 1. Verify reranker appears in eval logs

```bash
PYTHONPATH=. python scripts/test_query.py -q "What is the policy on software installation?"
```

Check Phoenix trace — the search tool log should show `rerank_score` (0.0–1.0 range), NOT `rrf_score` (0.01–0.03 range).

### 2. Verify agent passes original query

Same test. Check Phoenix trace — the `search_queries` in agent metadata should be the original question, not a keyword rewrite.

### 3. Re-run chatbot eval

```bash
PYTHONPATH=. python eval/run_experiment.py --tier chatbot --name chatbot-eval-fixed-v1
```

### Expected results after fix

| Metric | Before fix | Expected after |
|--------|-----------|----------------|
| hit_evaluator | 0.68 | ~0.85-0.90 |
| mrr_evaluator | 0.43 | ~0.65-0.75 |
| json_parse_success | 0.98 | ≥0.95 |

---

## File Changes Summary

| File | Change |
|------|--------|
| `eval/agent_wrapper.py` | Remove `hybrid_search` import. Rewrite `_logged_search_policies` to use `_original_search` and capture `_last_search_results`. |
| `rag/tools/search_policies.py` | Add `_last_search_results` module variable. Populate it before returning formatted results. |
| `rag/agent.py` | Add rule 0 to system prompt: pass original question verbatim to search. |

No other files change. No changes to eval datasets, evaluators, or retrieval pipeline.
