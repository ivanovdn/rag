# Augmented Generation — Implementation Plan

## Core Rule

The LLM is a **policy locator**. It finds the relevant policy, quotes it verbatim, and tells the user exactly where it is. It does NOT interpret, advise, summarize, or reason from its own knowledge. The policy text IS the answer.

---

## Current Stack

| Component | Location | Model |
|-----------|----------|-------|
| Vector DB | Local | Qdrant |
| Embeddings | Local | nvidia/llama-nemotron-embed-1b-v2 (HuggingFace) |
| Reranker | Local | Qwen3-Reranker-4B-Q8 via llama-server (llama.cpp, port 8081) |
| LLM | Remote Spark | Qwen 2.5 72B via Ollama |

Pipeline: Query → Vector Search (25 candidates) → Reranker (top 6) → Format Context → LLM → ComplianceAnswer JSON

---

## Task 1 — Rewrite context formatter in `rag/tools/search_policies.py`

### What to change

Replace the current result formatting block that builds `--- Result N ---` strings.

### New format spec

Results are ordered by **reranker score descending** (best match first). No reordering, no grouping.

```
=== RETRIEVED POLICY SOURCES ===

[Source 1] Acceptable Use Policy [Internal]
Section: Corporate Workstation and Software Use
Clause 4.7: Software Installation
---
4.7. Software Installation: Team Members are forbidden to install
any software on corporate workstations without prior approval from
the IT Department. This includes browser extensions, plugins...

[Source 2] Privacy Notice For Team Members [Internal]
Section: How we process your personal data
---
The Company processes personal data for the following purposes...

[Source 3] Information Security Policy [Internal]
Section: Endpoint Security
Clause 12.3: Software Whitelisting
---
12.3. Software Whitelisting: Only software approved by the Security
team and listed in the corporate software catalog may be installed...
```

### Format rules — follow exactly

1. Header line: `=== RETRIEVED POLICY SOURCES ===` — always present, once, at the top.
2. Each source starts with `[Source N]` followed by the document title. N starts at 1, increments sequentially.
3. Next line: `Section: {section name}` — always present.
4. Next line: `Clause {clause_number}: {clause name}` — ONLY if clause is non-empty. Omit entire line if no clause.
5. Next line: `---` separator.
6. Next lines: the chunk text verbatim. Do not modify, truncate, or reformat the chunk text.
7. One blank line between sources.
8. Order: by reranker score descending. Source 1 = highest reranker score.

### Fields to REMOVE from output

- `doc_id` — internal only, LLM does not need it
- `rerank_score` / `retrieval_score` / `vector_score` — creates bias, LLM should read all sources
- `section_display` — redundant with section + clause
- `chunk_index` — internal only

### Fields to KEEP in output

- `doc_title` → shown after `[Source N]`
- `section` → shown on Section line
- `clause` → shown on Clause line (if non-empty)
- `clause_number` → shown before clause name (if non-empty)
- `text` → shown verbatim after `---` separator

### When no results found

If search returns `NO_RELEVANT_POLICY_FOUND`, return exactly:

```
=== RETRIEVED POLICY SOURCES ===

NO_RELEVANT_POLICY_FOUND
```

### Implementation

```python
def format_sources(search_results: list[dict]) -> str:
    if not search_results:
        return "=== RETRIEVED POLICY SOURCES ===\n\nNO_RELEVANT_POLICY_FOUND"

    lines = ["=== RETRIEVED POLICY SOURCES ==="]

    for i, r in enumerate(search_results):
        lines.append("")  # blank line between sources
        lines.append(f"[Source {i + 1}] {r['doc_title']}")
        lines.append(f"Section: {r['section']}")
        if r.get("clause") and r.get("clause_number"):
            lines.append(f"Clause {r['clause_number']}: {r['clause']}")
        elif r.get("clause"):
            lines.append(f"Clause: {r['clause']}")
        lines.append("---")
        lines.append(r["text"])

    return "\n".join(lines)
```

Put this function in `rag/tools/search_policies.py`. Call it instead of the current formatting logic. The `search_results` list must already be sorted by reranker score descending before calling this function.

---

## Task 2 — Rewrite system prompt in `rag/agent.py`

### What to change

Replace the current system prompt string in `rag/agent.py`. The new prompt has 6 sections. Include ALL sections in this exact order. Do not omit or reword sections.

### New system prompt

```
You are an internal Compliance Policy Locator. Your ONLY job is to find the company policy that answers the user's question and show them exactly where it is.

== YOUR ROLE ==

You are a POINTER, not an ADVISOR. You find the policy, quote it, and cite its location. The policy text IS the answer. You never interpret, explain, summarize, or add your own reasoning.

== HOW TO RESPOND ==

1. Call search_policies FIRST for every question. Never answer without searching.
2. Read ALL returned sources before responding.
3. Identify which source(s) directly answer the question.
4. Quote the relevant policy text VERBATIM in your answer — copy the exact words from the source.
5. State the exact document name, section, and clause where the policy is found.
6. Copy the document title, section, clause, and clause number into the citations exactly as shown in the source header.
7. If the answer spans multiple sources, cite each one separately.
8. If no source answers the question, call escalate_to_compliance. Do not guess.

== ANSWER FORMAT ==

Start by naming the document and location, then quote the policy text.

CORRECT example:
"This is addressed in the Acceptable Use Policy [Internal], Section: Corporate Workstation and Software Use, Clause 4.7 (Software Installation): 'Team Members are forbidden to install any software on corporate workstations without prior approval from the IT Department.'"

WRONG example:
"You should not install software because it could pose a security risk. The IT team needs to approve all installations first."

WRONG example:
"Based on industry best practices, software installation should be controlled to prevent security vulnerabilities."

== RULES ==

- ONLY use information from the retrieved policy sources. Never answer from your own knowledge.
- NEVER paraphrase policy text. Always quote verbatim.
- NEVER give advice like "you should...", "it would be best to...", "I recommend...".
- NEVER interpret what a policy means beyond what it explicitly states.
- NEVER invent or assume policy rules that are not written in the sources.
- NEVER cite a source you did not use in your answer.
- If uncertain whether a source applies, escalate. Do not guess.

== ESCALATION ==

If search_policies returns NO_RELEVANT_POLICY_FOUND, or if none of the returned sources answer the question, you MUST call escalate_to_compliance with the full question and context. Do not attempt an answer.

== OUTPUT FORMAT ==

Your final response MUST be valid JSON matching this exact schema. No text before or after the JSON.

{
  "answer": "According to [Document Title], Section: [Section Name], Clause [Number] ([Clause Name]): '[verbatim quote from policy]'",
  "citations": [
    {
      "source_number": 1,
      "doc_title": "exact document title from source header",
      "section": "exact section name from source header",
      "clause": "exact clause name from source header",
      "clause_number": "e.g. 4.7",
      "quote": "verbatim text copied from the source"
    }
  ],
  "escalation": {"needed": false, "reason": ""}
}

The source_number must match the [Source N] number from the search results.
The doc_title, section, clause, and clause_number must be copied exactly from the source header.
The quote must be copied exactly from the source text.
```

### Implementation notes

- Store this as a constant string in `rag/agent.py`, e.g. `SYSTEM_PROMPT`.
- Pass it to `AgentWorkflow.from_tools_or_functions()` via the `system_prompt` parameter.
- Do NOT modify the prompt dynamically. It is static.
- `temperature=0.0` remains mandatory.

---

## Task 3 — Update ComplianceAnswer schema in `rag/agent.py`

### What to change

Add `source_number` field to the `Citation` model.

### New schema

```python
class Citation(BaseModel):
    source_number: int    # matches [Source N] from search results
    doc_title: str        # copied from source header
    section: str          # copied from source header
    clause: str           # copied from source header (empty string if no clause)
    clause_number: str    # e.g. "4.7" (empty string if no clause)
    quote: str            # verbatim from policy text

class Escalation(BaseModel):
    needed: bool
    reason: str

class ComplianceAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    escalation: Escalation
```

### Migration note

The `source_number` field is new. The eval `parse_agent_response()` in `eval/agent_wrapper.py` must handle both old responses (without `source_number`) and new ones. Make `source_number` optional with a default:

```python
source_number: int = 0  # 0 = not provided (backwards compat)
```

---

## Task 4 — Update get_section tool in `rag/tools/get_section.py`

### What to change

Apply same clean formatting to the tool's return string.

### New format

```
=== FULL SECTION ===

Document: Acceptable Use Policy [Internal]
Section: Corporate Workstation and Software Use
---
[full section text verbatim]
```

No source numbering. No scores. Just document, section, separator, text.

---

## Task 5 — Verify and test

### Test 1 — Visual inspection of formatted context

```bash
PYTHONPATH=. python scripts/test_query.py -q "What is the policy on software installation?"
```

Check Phoenix trace at http://localhost:6006. Open the `search_policies` tool call. Verify:
- Sources are numbered `[Source 1]`, `[Source 2]`, etc.
- No scores visible
- No doc_id visible
- Clause line is absent when clause is empty
- Chunk text is verbatim (not truncated or reformatted)

### Test 2 — LLM response is policy-locator style

Same query. Check the agent's final JSON response:
- `answer` starts with document name and location
- `answer` contains a verbatim quote from one of the sources
- `citation.quote` matches text from a source
- `citation.source_number` matches the `[Source N]` where the quote came from
- LLM does NOT interpret, advise, or reason beyond the quote

### Test 3 — Escalation still works

```bash
PYTHONPATH=. python scripts/test_query.py -q "What is the company's policy on cryptocurrency trading?"
```

Expected: `escalation.needed = true`. LLM does not attempt an answer.

### Test 4 — Run evals

```bash
PYTHONPATH=. python eval/run_experiment.py --tier tier2 --name generation-v2
PYTHONPATH=. python eval/run_experiment.py --tier chatbot --name generation-v2
```

Compare against previous baseline. Watch:

| Metric | Direction |
|--------|-----------|
| `answer_coverage` | Should improve (verbatim quotes match expected answers better) |
| `citation_doc_accuracy` | Should improve (cleaner metadata in context) |
| `citation_clause_accuracy` | Should improve (clause number explicit in header) |
| `json_parse_success` | Must stay ≥ 0.95 |

---

## File Changes — Complete List

| File | Action | What exactly |
|------|--------|-------------|
| `rag/tools/search_policies.py` | Modify | Replace result formatting with `format_sources()` function. Keep reranker-score ordering. Remove scores, doc_id, section_display from output. Add `[Source N]` numbering. |
| `rag/agent.py` | Modify | Replace system prompt with new 6-section prompt. Update `Citation` model to add `source_number: int = 0`. |
| `rag/tools/get_section.py` | Modify | Update return format to `=== FULL SECTION ===` style. Remove scores and doc_id. |
| `eval/agent_wrapper.py` | Modify (minor) | Ensure `parse_agent_response()` handles `source_number` field (optional, default 0). |

No new files. No changes to eval datasets. No changes to search or reranker logic.

---

## Do NOT change

- Retrieval pipeline (vector search + reranker) — unchanged
- `temperature=0.0` — mandatory
- `ComplianceAnswer` as the required output schema — only add `source_number`
- Tool names in agent — must match system prompt references
- Eval datasets or evaluator logic — only output format changes
