# Qdrant-native sparse — measured results and follow-ups

**Date:** 2026-10-02
**Branch:** `feat/qdrant-native-sparse` (22 commits, merged at `a689139`)
**Spec:** `2026-09-29-qdrant-native-sparse-design.md` · **Plan:** `../plans/2026-09-29-qdrant-native-sparse.md`
**Supersedes:** follow-ups #2 and #3 of `2026-09-25-search-first-results.md`

All numbers from `chatbot-test-v1`, 61 distinct labelled questions, run through the
real remote stack (Ollama + Qdrant + vLLM reranker on `172.20.0.22`) from a
throwaway container alongside the live bot, with the branch's source mounted over
the image.

---

## Results

| metric | file-based BM25 (2026-09-25) | v2 control, BM25 **off** | **v2 gate, native sparse** |
|---|---:|---:|---:|
| `hit_evaluator` | 0.9344 | 0.9180 | **0.9344** |
| `mrr_evaluator` | 0.7664 | 0.7628 | **0.7697** |
| `retrieval_doc_hit` | 0.9836 | 0.9672 | **0.9836** |
| `retrieval_section_hit` | 0.9262 | 0.9262 | **0.9508** |
| `json_parse_success` | 1.0000 | 1.0000 | 1.0000 |
| latency (median) | 5,131ms | 4,694ms | 5,164ms |

Gate bar was `hit_evaluator` ≥ 0.9180 (spec D9). **Native sparse matched the
file-based index exactly at 0.9344** rather than merely clearing the floor, and beat
it on `retrieval_section_hit` (+0.0246, ~1.5 questions of 61) and `mrr_evaluator`.
Latency is flat — server-side fusion costs nothing measurable versus fusing in
Python.

### The control run is what makes those numbers mean anything

`--verify` compared 25 of 1,602 dense vectors between the old collection and the
new one and found them identical. The control run — same new collection, BM25 off —
returned **0.9180, the dense-only baseline, to four decimal places**. That extends
the guarantee from 25 sampled vectors to all 1,602, end to end through the real
retrieval path.

Without it, 0.9344 would have been a number requiring trust. With it, every
difference above is attributable to the sparse half alone, which is the entire
reason the migration copies points instead of re-ingesting (spec D5) and refuses to
change the indexed text at the same time (spec D6).

The `retrieval_section_hit` gain is most plausibly **stemming**: Qdrant's built-in
tokenizer stems, so `"installation"` matches a chunk containing `"install"`. The
deleted pure-Python tokenizer never did. Unproven, but it is the one encoding
difference that survived.

---

## What the migration deleted

| | lines |
|---|---:|
| `rag/bm25_index.py` | 230 |
| `rag/hybrid_search.py` | 166 |
| `scripts/build_bm25_from_qdrant.py` | 149 |
| the BM25 blocks in `ingest/pipeline.py` | 8 |

Plus the `.bm25_index.json` entries in `.gitignore` and `.dockerignore`, the volume
mount that was never configured, and the failure mode that made the whole thing
unshippable: an index built before a re-ingest had the right chunk **count** and
entirely wrong ids (measured 2026-09-25: 1602 vs 1602, 0 of 25 sampled ids shared),
after which fusion ran against nothing and reported no error.

Tests went 276 → 344.

---

## Verified against the real server before migrating

`scripts/rehearse_sparse.py` ran nine checks on throwaway collections against
`172.20.0.22`, all passing. The two that mattered most:

- **Qdrant encodes the text we send.** We never compute a sparse vector — ingest and
  query both send `Document(text=…, model="qdrant/bm25")` and the server tokenizes,
  weights and applies IDF. Confirmed by reading a stored vector back.
- **A sparse write to a sparse-less collection really is rejected** (`400 Not
  existing vector name`). That is the premise the ingest guard rests on; without it
  the guard would be defending against nothing.

The stemming check also confirmed the IDF formula arithmetically: a query scored
2.9887, and `2 terms × 1.5235 × ln(1 + 2.5/1.5)` = 2.9885 — the same
`ln(1 + (N−df+0.5)/(df+0.5))` the deleted Python implementation used.

---

## Two things we got wrong, and how they were caught

**Qdrant's RRF constant is k=2, not k=60.** The spec asserted throughout that "an
RRF score is ~0.016" — the value the deleted client-side code produced at k=60.
`FusionQuery(fusion=Fusion.RRF)` uses Qdrant's default of 2, giving 0.5 / 0.333 /
0.25 for ranks 1-3: a 30× shift, silently, in the same branch whose gate assumes the
encoder is the only changed variable. Caught by the whole-branch review, fixed by
pinning `RrfQuery(rrf=Rrf(k=RRF_K))`. A gate failure would otherwise have been
blamed on the tokenizer.

**A failed re-ingest would have deleted a policy.** `ingest_document` deletes a
document's chunks and *then* upserts, and the upsert now carries a sparse vector —
so against the pre-migration collection it failed after the delete committed,
removing a policy from the live index while the bot looked healthy. Reachable for
the whole rollout window the spec itself creates. Fixed with `assert_sparse_vector`
running before the delete, ungated by `BM25_ENABLED` because the write is ungated.

---

## Follow-ups, in priority order

1. **Delete `compliance_policies`** once v2 has run in production long enough to
   trust. Until then it is the rollback: restore the two `.env` values and redeploy.
2. **Citation provenance.** Nothing verifies that cited chunks came from the
   retrieved set — carried from `2026-09-25-search-first-results.md` #1, still the
   largest correctness gap, still deserving its own spec and eval.
3. **Four `Settings` fields should be `SecretStr`.** `monkeypatch.setattr`
   interpolates `repr(target)` into its `AttributeError`, and `repr(settings)`
   carries `hf_token` / `teams_client_secret` / `teams_refresh_token` /
   `smtp_password` as plain `str`. No test typos a setting name today, so nothing
   leaks — but the suite is one typo from printing `.env` into CI output, and the
   branch added ~25 more instances of the pattern. One-line fix in `config.py`,
   repo-wide effect.
4. **`scripts/test_query.py` has no sparse preflight.** The only entry point left
   that meets the raw `Not existing vector name` error instead of an actionable
   message — and it is the first tool anyone reaches for when debugging retrieval.
5. **Section and clause precision.** `retrieval_section_hit` is 0.9508; the
   remaining failures are the next real lever. On the `vllm-score` backend
   `RERANKER_INSTRUCTION` is the knob that applies, not `RERANKER_QUERY_TEMPLATE`.
   `RERANKER_CANDIDATES` and chunking are the others (there is no longer a
   per-branch candidate count — see #12).
6. **Extract `cosine_floor_applies()`.** The predicate
   `not reranker_enabled and not bm25_enabled` now exists in four places — the live
   guard in `search_policies.py` and three metadata mirrors. It encodes the branch's
   sharpest lesson (a cosine threshold must never judge an RRF score) and drift
   between the copies would be silent.
7. **`migrate_collection.py` registers Phoenix for nothing.** It opens no spans, so
   the call costs ~3.6s of startup and a banner. `rehearse_sparse.py` skips it
   deliberately for exactly this reason; the two should agree.
8. **`score_type` survives `rerank()` only by inspection.** Verified by reading
   `_rerank_impl` (the success path shallow-copies, the fallbacks return the
   originals), but no test pins it, so a future change to that copy-vs-mutate
   mechanic would drop the unit label from eval JSON silently.
9. **Retrieval evaluators score `status: "unavailable"` rows as 0**, so a backend
   blip during a gate run reads as a regression. Carried from #5.
10. **`num_ctx` above 4096** remains hypothetically safe now that the agent is
    tool-free. Goes through `2026-09-17-ollama-moe-cuda-crash.md`'s matrix, never
    ad hoc. Carried from #7.
11. **`render_escalation` does not HTML-escape** and interpolates a model-supplied
    reason. Pre-existing. Carried from #9.
12. ~~**Prefetch limits ignore the caller's `top_k`**~~ — **DONE 2026-10-02**,
    and worse than recorded here. The threshold is not their *sum* but their
    *union*: production ran 20/20 against a fused limit of 25, so any query where
    the dense and sparse branches agreed on more than 15 documents returned fewer
    than 25 candidates, silently — and high overlap is what a working hybrid
    produces, so the shortfall tracked retrieval quality. Fixed by collapsing
    `RETRIEVAL_TOP_K`, `HYBRID_VECTOR_CANDIDATES` and `HYBRID_BM25_CANDIDATES`
    into `RERANKER_CANDIDATES`, which now sizes both prefetches and the limit
    they feed, making the union >= the limit by construction.
