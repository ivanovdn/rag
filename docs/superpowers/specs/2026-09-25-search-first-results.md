# Search-first — measured results and follow-ups

**Date:** 2026-09-25
**Branch:** `feat/search-first` (37 commits)
**Spec:** `2026-09-24-search-first-design.md` · **Plan:** `../plans/2026-09-24-search-first.md`

All numbers from `chatbot-test-v1`, 61 distinct labelled questions, run through the
real remote stack (Ollama + Qdrant + vLLM reranker on `172.20.0.22`) from a
throwaway container alongside the live bot.

---

## Results

| metric | main | search-first | + top_n=10 | **+ BM25 hybrid** |
|---|---:|---:|---:|---:|
| `hit_evaluator` | 0.8525 | 0.9180 | 0.9344 | **0.9344** |
| `mrr_evaluator` | 0.6986 | 0.7628 | 0.7647 | **0.7664** |
| `retrieval_doc_hit` | 0.9016 | 0.9672 | 0.9836 | **0.9836** |
| `retrieval_section_hit` | 0.8689 | 0.9262 | **0.9590** | 0.9262 |
| `json_parse_success` | 0.9836 | 1.0000 | 0.9836 | **1.0000** |
| `agent_search_count` | 0.9344 | 1.0000 | 1.0000 | 1.0000 |
| latency (median) | 7,468ms | 4,939ms | 5,893ms | **5,131ms** |

**Best configuration: search-first + BM25 hybrid at `RERANKER_TOP_N=6`.**
Against main: `hit_evaluator` +0.082, latency −31%.

### Why the branch wins, precisely

The entire retrieval gain from main → search-first is **four questions main
answered without retrieving anything at all**:

- "The client didn't mention AI—can I still use it?"
- "I think I may have sent personal data to the wrong person…"
- "Does the company sell my personal data?"
- "If it's just for internal tools, can I skip approvals?"

Each returned an empty answer with zero citations. All four now retrieve at rank 1
and cite. `+0.0656` = exactly 4/61, and it appears identically on `hit_evaluator`,
`retrieval_doc_hit` and `agent_search_count`. Retrieval did not get better — it
started happening.

Main also produced one unparseable answer (unclosed JSON at 681 chars, well under
`num_predict` 1024) — consistent with `num_ctx` exhaustion under main's 1,883
tokens of fixed overhead versus the branch's 571. Inferred, not measured: eval
records no token counts.

### `top_n=10` is not an improvement

It recovers CB-046 but breaks the "Kyiv air alert" case, which truncates at 7
sources. Kyiv needs ≤6, CB-046 needs ≥9; at `num_ctx` 4096 both are impossible.
One escalation traded for another, at +19% median and +68% max latency.

### BM25 gets the same gain for free

CB-046's correct document was ranked **9th by the reranker at 0.1503**, while BM25
ranks it **1st at 14.925**. It was a ranking failure, not a depth failure — so
lexical fusion fixes it at `top_n=6`, leaving the Kyiv case intact.

---

## Findings worth keeping

**Query rewriting was real and unmeasured.** Main's agent rewrote the search query
in 7 of 57 searches (12%) despite the prompt's rule 0 forbidding it — e.g. "My
antivirus slows down my laptop. Can I disable it while working?" → "Can I disable
my antivirus while working because it slows down my laptop?". Occasionally it
helped (it silently fixed a typo in one question, which cost that case a rank when
we went verbatim). It is now structurally impossible.

**No threshold separates good retrieval from bad on this corpus.** Correct
retrievals scored 0.8478–0.9998 (n=59), but one of the two document misses scored
**0.9886**. `RERANKER_MIN_SCORE=0.2` fired on none of the 61 and is deliberately
inert — it exists for obviously-irrelevant questions, not borderline ones.

**A repeated question is not a sample.** An earlier threshold was justified from
"38 production requests, n=3 escalated / n=27 answered". Those 38 were **5 distinct
questions**, one repeated 23 times by load testing. Corrected in `config.py`.

---

## Follow-ups, in priority order

1. **Citation provenance.** Nothing verifies that cited chunks came from the
   retrieved set — `rag/response.py` takes citations verbatim and the backstop
   checks only non-emptiness. On confidently-wrong sources the model's own
   judgement is the sole guard, against sources it is now always handed.
   `sp._last_search_results` is in scope at the parse point, so a doc/section
   membership check is cheap. Deserves its own spec and eval.

2. **Enable BM25 in production.** Measured best. Two prerequisites:
   `.bm25_index.json` must be built on the host (`scripts/build_bm25_from_qdrant.py`)
   **and mounted into the bot container** — `docker-compose-remote.yml` does not
   mount it today, so the bot would load an empty index and silently lose the
   lexical half.

3. **Migrate to Qdrant-native sparse vectors.** `rfq_v1` on the same server already
   uses `bm25 Sparse` alongside dense. That deletes the entire class of bug this
   branch hit: the index lives in the collection, survives re-ingestion, needs no
   file, no mount, and Qdrant fuses server-side. Cost: collection recreate plus
   re-ingest with sparse vectors.

4. **Trace eval runs.** `run_experiment.py` never calls `init_observability()`, so
   eval emits no spans and records no token counts — which is why the truncation
   mechanism above is inferred rather than measured. Opt-in, writing to a separate
   Phoenix project so production traces stay clean.

5. **Retrieval evaluators score `status: "unavailable"` rows as 0**, alongside
   `json_parse_success`, so a backend blip during a gate run looks like a
   regression. The `status` key exists to filter on; the evaluators do not use it.

6. **The remaining 5 failures are section/clause precision** — 3 with the right
   document but the wrong section or clause, 2 true document misses. Levers:
   `RERANKER_INSTRUCTION` (the only reranker knob that applies on the `vllm-score`
   backend), `HYBRID_BM25_CANDIDATES`, chunking.

7. **`num_ctx` above 4096** is now hypothetically safe — tool-free removes
   constrained decoding, one of the five CUDA-crash conditions. It would end the
   Kyiv-class truncation and unlock `top_n>6`. Goes through
   `2026-09-17-ollama-moe-cuda-crash.md`'s matrix, never ad hoc.

8. **`answer_coverage` is unfit for this system** and was removed from the run set.
   It scores word overlap against a human-written summary while the bot is built to
   quote policy verbatim — it measured 0.4754 with 32 zeros on a run whose citation
   accuracy was 0.81–0.88.

9. **`render_escalation` does not HTML-escape** and interpolates a model-supplied
   `reason`, and `bot.py`'s non-transient LLM path still puts an untruncated
   `str(e)` there. Pre-existing; truncation was added on the new path only.
