# TASK: Fix Evaluation Harness — Tier 1 Matching Logic

## Problem

The current `scripts/run_eval.py` has three issues in Tier 1 retrieval evaluation:

1. **`normalize_doc_id` is unnecessary.** Test JSON has document title as-is ("Acceptable Use Policy [Internal]"), Qdrant has `doc_title` with the same string. Simple case-insensitive comparison is enough.

2. **Text comparison (`expected_text_contains`) should be removed from Tier 1.** If document + section + clause all match, we've confirmed the right chunk was found. No need for text matching.

3. **Matching doesn't use the new metadata fields.** After the DOCX parser fix, Qdrant now stores `section`, `section_number`, `clause`, `clause_number` as separate fields. The eval script must read these from search results and match against them.

## Changes Required in `scripts/run_eval.py`

### Change 1: Remove `normalize_doc_id` function

Delete the entire function (lines ~70-78):

```python
# DELETE THIS ENTIRE FUNCTION:
def normalize_doc_id(s: str) -> str:
    """Normalize doc id for flexible matching: lowercase, strip brackets, slugify."""
    import re
    s = s.lower().strip()
    s = re.sub(r"[\[\]()]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s
```

Also delete the `text_contains` helper — it's no longer used in Tier 1 (keep it if Tier 2/4 still use it):

```python
# KEEP this function — still used by other tiers
def text_contains(haystack: str, needle) -> bool:
    ...
```

### Change 2: Update search_results dict to include new metadata fields

In `run_retrieval_eval`, update both the hybrid search and vector-only branches to include the new fields:

**Hybrid search branch (around line 148-159):**

Replace with:
```python
            if settings.bm25_enabled:
                from rag.hybrid_search import hybrid_search

                raw_results = hybrid_search(tc["question"], top_k=top_k)
                search_results = []
                for r in raw_results:
                    search_results.append(
                        {
                            "doc_id": r["doc_id"],
                            "doc_title": r["doc_title"],
                            "section": r.get("section", ""),
                            "section_number": r.get("section_number", ""),
                            "clause": r.get("clause", ""),
                            "clause_number": r.get("clause_number", ""),
                            "section_display": r.get("section_display", ""),
                            "text": r["text"],
                            "score": r["rrf_score"],
                        }
                    )
```

**Vector-only branch (around line 161-175):**

Replace with:
```python
            else:
                vector = embed_query(tc["question"])
                raw_results = search_vectors(vector, top_k=top_k)
                search_results = []
                for r in raw_results:
                    p = r.payload
                    search_results.append(
                        {
                            "doc_id": p.get("doc_id", ""),
                            "doc_title": p.get("doc_title", ""),
                            "section": p.get("section", ""),
                            "section_number": p.get("section_number", ""),
                            "clause": p.get("clause", ""),
                            "clause_number": p.get("clause_number", ""),
                            "section_display": p.get("section_display", ""),
                            "text": p.get("text", ""),
                            "score": r.score,
                        }
                    )
```

### Change 3: Replace the entire matching block

Remove lines ~182-215 (the old matching logic with normalize_doc_id and text_match).

Replace with:

```python
            # Check for hit
            hit = False
            hit_rank = None
            # .strip() guards against trailing whitespace from Excel cells
            expected_doc = tc.get("expected_doc_id", "").strip()
            expected_section = tc.get("expected_section_contains", "").strip()
            expected_clause = tc.get("expected_clause", "").strip()

            for rank, sr in enumerate(search_results, start=1):
                # Document: case-insensitive exact match against doc_title
                doc_match = (
                    not expected_doc
                    or expected_doc.lower() == sr.get("doc_title", "").lower()
                )

                # Section: match against section name
                section_match = (
                    not expected_section
                    or expected_section.lower() in sr.get("section", "").lower()
                )

                # Clause: match against clause name
                clause_match = (
                    not expected_clause
                    or expected_clause.lower() in sr.get("clause", "").lower()
                )

                if doc_match and section_match and clause_match:
                    hit = True
                    hit_rank = rank
                    break
```

That's it. Three clean case-insensitive checks. No normalization, no slugification, no text matching.

### Change 4: Log all search results to Phoenix spans for full inspection

Tier 1 calls `hybrid_search()` directly — it doesn't go through the agent, so LlamaIndex auto-instrumentation doesn't apply. We need to explicitly log what was retrieved into the Phoenix span so you can inspect every query in the Phoenix UI.

After the hit/miss is determined (right after the `break` from the matching loop), update the existing span attributes block:

```python
            span.set_attribute("eval.hit", hit)
            span.set_attribute("eval.hit_rank", hit_rank or 0)
            span.set_attribute("eval.top_score", top_score)
            span.set_attribute("eval.expected_doc", expected_doc)
            span.set_attribute("eval.expected_section", expected_section)
            span.set_attribute("eval.expected_clause", expected_clause)

            # Log ALL returned results so you can inspect in Phoenix
            for ri, sr in enumerate(search_results):
                span.set_attribute(f"eval.result.{ri}.doc_title", sr.get("doc_title", ""))
                span.set_attribute(f"eval.result.{ri}.section", sr.get("section", ""))
                span.set_attribute(f"eval.result.{ri}.clause", sr.get("clause", ""))
                span.set_attribute(f"eval.result.{ri}.score", sr.get("score", 0.0))
```

This means in Phoenix UI, for every Tier 1 test case you can see:
- `eval.expected_doc` / `eval.expected_section` / `eval.expected_clause` — what we were looking for
- `eval.result.0.doc_title`, `eval.result.0.section`, `eval.result.0.clause`, `eval.result.0.score` — rank 1 result
- `eval.result.1.*` — rank 2 result
- ... up to top_k

### Change 5: Update result output with full search results for JSON inspection

Replace the results.append block (lines ~227-237) with:

```python
            results.append(
                {
                    "id": tc["id"],
                    "question": tc["question"],
                    "hit": hit,
                    "hit_rank": hit_rank,
                    "top_score": top_score,
                    # What we expected
                    "expected_doc": expected_doc,
                    "expected_section": expected_section,
                    "expected_clause": expected_clause,
                    # What was actually found (for debugging misses)
                    "matched_doc": search_results[hit_rank - 1]["doc_title"] if hit else "",
                    "matched_section": search_results[hit_rank - 1].get("section", "") if hit else "",
                    "matched_clause": search_results[hit_rank - 1].get("clause", "") if hit else "",
                    # ALL top-k results for inspection
                    "search_results": [
                        {
                            "rank": ri + 1,
                            "doc_title": sr.get("doc_title", ""),
                            "section": sr.get("section", ""),
                            "clause": sr.get("clause", ""),
                            "score": round(sr.get("score", 0.0), 4),
                        }
                        for ri, sr in enumerate(search_results)
                    ],
                }
            )
```

Now when RET-016 misses, you open the results JSON and see exactly what 6 chunks came back:

```json
{
  "id": "RET-016",
  "question": "What are the core responsibilities of the HIPAA Privacy Officer?",
  "hit": false,
  "expected_doc": "Organization Roles and Areas of Responsibility [Internal]",
  "expected_section": "HIPAA Privacy Officer",
  "expected_clause": "Responsibilities",
  "search_results": [
    {"rank": 1, "doc_title": "Data Privacy Policy [Internal]", "section": "Data Processing", "clause": "Third Party Sharing", "score": 0.7123},
    {"rank": 2, "doc_title": "Organization Roles...", "section": "HIPAA Privacy Officer", "clause": "Appointment", "score": 0.6891},
    ...
  ]
}
```

In this example you'd see rank 2 had the right document and section but wrong clause — helpful for tuning.

### Change 6: Update the log line for better debugging

Replace line ~239-240:

```python
            status = f"HIT@{hit_rank}" if hit else "MISS"
            logger.info(f"  [{tc['id']}] {status} (top_score={top_score:.3f})")
```

With:

```python
            if hit:
                logger.info(
                    f"  [{tc['id']}] HIT@{hit_rank} (score={top_score:.3f}) "
                    f"doc={sr.get('doc_title', '')[:30]} | sec={sr.get('section', '')[:20]} | cls={sr.get('clause', '')[:20]}"
                )
            else:
                top = search_results[0] if search_results else {}
                logger.info(
                    f"  [{tc['id']}] MISS (score={top_score:.3f}) "
                    f"expected: doc={expected_doc[:30]} sec={expected_section[:20]} cls={expected_clause[:20]} | "
                    f"got: doc={top.get('doc_title', '')[:30]} sec={top.get('section', '')[:20]} cls={top.get('clause', '')[:20]}"
                )
```

Now when something misses, you immediately see what was expected vs what was found.

## What the Test JSON Should Look Like

Tier 1 (`retrieval_test.json`):

```json
{
  "id": "RET-001",
  "question": "What are the core responsibilities of the HIPAA Privacy Officer?",
  "expected_doc_id": "Organization Roles and Areas of Responsibility [Internal]",
  "expected_section_contains": "HIPAA Privacy Officer",
  "expected_clause": "Responsibilities"
}
```

The `expected_text_contains` field is ignored — you can leave it in the JSON or remove it from your XLSX. The converter still produces it but the eval script no longer uses it.

## How Matching Works After Fix

| Test JSON field | Compared against | Method |
|---|---|---|
| `expected_doc_id` | `doc_title` in Qdrant | Case-insensitive exact match |
| `expected_section_contains` | `section` in Qdrant | Case-insensitive substring (`in`) |
| `expected_clause` | `clause` in Qdrant | Case-insensitive substring (`in`) |

**Why substring (`in`) for section and clause?** Because the Compliance team might write "HIPAA Privacy Officer" and the Qdrant field might be "HIPAA Privacy Officer Responsibilities". Substring match is forgiving enough without being too loose.

**Why exact match for document?** Because document titles are precise and should match exactly. "Acceptable Use Policy [Internal]" should not match "Data Privacy Policy [Internal]" — substring would cause false positives here.

## Also Update `rag/hybrid_search.py`

The `HybridResult` dataclass and the `hybrid_search_formatted` function need to include the new metadata fields so they propagate to the eval script. Make sure `HybridResult` has:

```python
@dataclass
class HybridResult:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str = ""
    section_number: str = ""
    clause: str = ""
    clause_number: str = ""
    section_display: str = ""
    text: str = ""
    doc_link: str = ""
    vector_score: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int = 0
    bm25_rank: int = 0
    rrf_score: float = 0.0
```

And the `reciprocal_rank_fusion` function reads these from Qdrant payloads:

```python
# In the vector results processing:
merged[chunk_id] = HybridResult(
    chunk_id=chunk_id,
    doc_id=payload.get("doc_id", ""),
    doc_title=payload.get("doc_title", ""),
    section=payload.get("section", ""),
    section_number=payload.get("section_number", ""),
    clause=payload.get("clause", ""),
    clause_number=payload.get("clause_number", ""),
    section_display=payload.get("section_display", ""),
    text=payload.get("text", ""),
    doc_link=payload.get("doc_link", ""),
    vector_score=point.score,
    vector_rank=rank,
)
```

## Files Changed

| File | Change |
|------|--------|
| `scripts/run_eval.py` | Remove `normalize_doc_id`, remove text matching from Tier 1, add new metadata fields to search_results, update matching logic, add debug output |
| `rag/hybrid_search.py` | Add `section`, `section_number`, `clause`, `clause_number` to `HybridResult` and populate from Qdrant payload |

## Verification

After implementing, run:

```bash
python scripts/run_eval.py --tier retrieval --tag "test-matching"
```

Check the logs. For HITs you should see:
```
[RET-001] HIT@1 (score=0.82) doc=Acceptable Use Policy [Inte | sec=Private Information | cls=Blogging and Social Me
```

For MISSes you should see both expected and actual:
```
[RET-016] MISS (score=0.71) expected: doc=Organization Roles and Ar sec=HIPAA Privacy Off cls=Responsibilities | got: doc=Data Privacy Policy [Int sec=Data Processing cls=Third Party Sharing
```

This immediately tells you whether it's a retrieval problem (wrong doc found) or a metadata problem (right doc, wrong section).

Also check the results JSON — every MISS now includes the full `search_results` array showing all top-k chunks that were retrieved, so you can see why the expected chunk wasn't found.

In Phoenix UI, filter spans by `eval.retrieval` and click into any one. The `eval.result.0.*`, `eval.result.1.*` attributes show every returned chunk with its metadata and score.

## What's Tracked Where (All Tiers)

| Tier | Phoenix auto-trace (LlamaIndex) | Phoenix eval span attributes | Results JSON |
|------|--------------------------------|------------------------------|--------------|
| 1. Retrieval | NO — calls hybrid_search directly, not the agent | `eval.test_id`, `eval.hit`, `eval.hit_rank`, `eval.top_score`, `eval.expected_*`, `eval.result.N.*` (all returned chunks) | Full search_results array per test case |
| 2. E2E | YES — full agent trace (ReAct steps, tool calls, LLM prompt/response, latency per span) | `eval.test_id`, `eval.citation_correct`, `eval.fact_coverage`, `eval.latency_seconds` | Answer text, citation check, fact coverage |
| 3. Escalation | YES — full agent trace | `eval.test_id`, `eval.was_escalated`, `eval.correctly_escalated`, `eval.false_answer` | Answer text, escalation check |
| 4. Chatbot | YES — full agent trace | `eval.question`, `eval.positive_score`, `eval.negative_score`, `eval.passed`, `eval.policy_cited` | Answer text, pos/neg scores, pass/fail |

**To inspect any query in Phoenix:**
- Open http://localhost:6006
- Filter by span name: `eval.retrieval`, `eval.e2e`, `eval.escalation`, or `eval.chatbot`
- Click into a span to see all attributes
- For Tiers 2-4, the parent trace contains the full agent execution (click up to the root trace)

