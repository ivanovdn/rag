# One candidate knob — measured results and follow-ups

**Date:** 2026-10-06
**Branches:** `feat/one-candidate-knob` (7 commits, merged at `10be4a2`) and
`feat/prefetch-limit-span` (merged at `419bfb0`)
**Supersedes:** follow-up #12 of `2026-10-02-qdrant-native-sparse-results.md`

No spec. This started as a question — *"we always use the reranker; can the four
candidate settings collapse into one with no damage to logic?"* — and the answer
turned out to need measuring rather than reasoning.

---

## The question and the short answer

Retrieval had four settings governing how many candidates it fetched:

```
RETRIEVAL_TOP_K           20
RERANKER_CANDIDATES       25
HYBRID_VECTOR_CANDIDATES  20
HYBRID_BM25_CANDIDATES    20
```

Three are now gone. `RERANKER_CANDIDATES` sizes the dense prefetch, the sparse
prefetch, and the fused limit they feed.

**`RETRIEVAL_TOP_K` was unreachable.** `search_chunks` read it only through
`limit = top_k or settings.retrieval_top_k`, and every caller passed a truthy
`top_k` — `reranker_candidates` with the reranker on, its own `top_k` with it
off. No `.env` edit could have woken it. `top_k` is now a required argument, so
the dead fallback cannot return.

**The two `HYBRID_*` settings were worse than inert.** The fused result is drawn
from the **union** of the two prefetch branches. Production ran 20/20 against a
fused limit of 25, so the union ranged from 20 (identical branches) to 40
(disjoint): any query where dense and sparse agreed on more than 15 documents
returned fewer than 25 candidates, with nothing reporting it. High overlap is
what a working hybrid produces, so the shortfall tracked retrieval quality.

Sized together, the union is `>=` the limit by construction.

---

## Results

`chatbot-test-v1`, 61 questions, real remote stack, from a container alongside
the live bot with the branch mounted over the image. Experiment 8 is the
2026-10-02 native-sparse gate; experiment 9 is this change. Both carry identical
metadata — `cand=25`, `bm25=True`, `collection=compliance_policies_v2` — so the
prefetch sizing is the only variable.

| metric | exp 8 (prefetch 20/20) | exp 9 (prefetch 25/25) |
|---|---:|---:|
| `hit_evaluator` | 0.9344 | **0.93442623** |
| `mrr_evaluator` | 0.7697 | **0.76967049** |
| `retrieval_doc_hit` | 0.9836 | **0.98360656** |
| `retrieval_section_hit` | 0.9508 | **0.95081967** |
| `json_parse_success` | 1.0000 | 1.0000 |
| latency (median) | 5,164ms | 5,142ms |

Bit-identical on every metric. 61 task runs, 0 errors.

**Identical is evidence the new code ran, not that it didn't.**
`hybrid_vector_candidates` no longer exists on `Settings`, so a surviving old
prefetch line would have raised `AttributeError` on all 61 questions. Combined
with a version assertion run through the same mounts before the GPU spend, the
new path definitely executed.

### How often the bug actually fired

Counting `qdrant.returned_count` across every `search_vectors` span in both
experiments' Phoenix projects:

| run | spans | `returned_count` |
|---|---:|---|
| exp 8 — gate, prefetch 20/20 | 61 | `{25: 60, 24: 1}` |
| exp 9 — this change, 25/25 | 61 | `{25: 61}` |

**Once in 61 questions, costing one candidate.** The query was *"Can I use an
open source library in a client project if it speeds up development?"*, and
`returned_count: 24` is readable: the union of two 20-item lists was 24, so the
branches agreed on 16 of 20 — exactly the high-overlap condition predicted to
starve the pool.

The 24th-ranked candidate was never going to survive the reranker's top 6, which
is why no metric moved.

### So the honest verdict

The bug was **real but inert** on this corpus. Returning 20 instead of 25 was not
changing answers. This is a simplification with a proven no-regression, not a
rescue, and it would have been wrong to present it as one.

What the change actually buys:

1. **The knob is tunable.** Raising `RERANKER_CANDIDATES` 25 → 40 used to be
   half-inert: the pool stayed bounded by a 20+20 union regardless. It now moves
   all three limits together, which makes follow-up #5 of the sparse results
   (section/clause precision) a one-line experiment instead of three settings to
   keep in sync.
2. **Four settings became one**, and its number is true.
3. `top_k` being required kills the dead-fallback class that made
   `RETRIEVAL_TOP_K` inert in the first place.

---

## Confirmed in production

The same Teams question was answered three times across the change:

| | 2026-10-02 12:07 | 2026-10-06 10:07 | 2026-10-06 10:24 |
|---|---|---|---|
| build | pre-change | indeterminate | provably current |
| `qdrant.limit` | 25 | 25 | 25 |
| `qdrant.prefetch_limit` | — | — | **25** |
| `qdrant.returned_count` | 25 | 25 | 25 |
| `qdrant.top_score` | 0.032795697 | 0.032795697 | 0.032795697 |
| rerank in → out | 25 → 6 | 25 → 6 | 25 → 6 |
| `reranker.top_score` | 0.9980826377868652 | identical | identical |

`0.032795697` is `1/60 + 1/62` — rank 0 in one branch and rank 2 in the other,
the `k=60` signature, unchanged.

The middle column is why `qdrant.prefetch_limit` now exists. That run was
bit-identical to the pre-change one and **nothing in the trace could distinguish
"the fix is live" from "the deploy did not happen"** — answering it took a
`docker exec` against the running container. The attribute equals `qdrant.limit`
in correct code, which is the entire point of recording it: redundant until it
is not.

---

## What else came out of it

### `--top-k` was itself a setting that never fired

It was threaded as a parameter into `make_tier1_task()`. But tier2 and chatbot
retrieve through `rag.tools.search_policies`, which reads
`settings.reranker_candidates` directly — so the flag reached tier1 only while
advertising itself generically. A sweep on the chatbot tier, the tier every gate
is measured from, would have silently used the `.env` value and reported 25
against 25 believing it was 25 against 40.

Renamed `--candidates` (`--top-k` kept as an alias) and written **into** settings,
where every reader looks.

### Nine settings had no reader anywhere

Audited all 72 `Settings` fields for a reader in any module or computed property.
Nine had none: `smtp_host` / `smtp_port` / `smtp_user` / `smtp_password` /
`compliance_team_email` for an escalation email never built, `api_secret_key` /
`admin_api_key` for the removed HTTP API, `database_url` for the `db/` stub, and
`eval_confidence_threshold` for nothing at all. Eight were advertised in
`.env.example`, so setup filled in values that went nowhere. 72 → 63 fields.

`smtp_password` was also one of the four plain-`str` secrets that
`repr(settings)` carries into a pytest `AttributeError` — follow-up #3 of the
sparse results. Deleting an unused secret is the cheapest quarter of that fix.

**Kept deliberately: `min_confidence_score`.** Unlike the nine, it *is* read and
*is* reachable — just not with the reranker and BM25 both on. Deleting it would
change behaviour on the pure-dense path `scripts/test_query.py` uses, and its
guard is what stops a cosine threshold from judging an RRF score. The line drawn:
**unreferenced gets deleted; unreachable-on-this-config does not.** Its `.env`
line was removed, because it was set to a hand-tuned 0.05 that could not fire.

### `.gitignore` covered `.env` but not `.env.*`

`sed -i.bak` on the deploy host — which this cleanup told an operator to run —
left `.env.bak` untracked but **not ignored**. A `git add -A` in `~/rag` would
have staged a file holding the HF token, the Teams client secret and the live
rotating refresh token, and pushed it to two remotes. `.dockerignore` had covered
`.env.*` since the image was built; `.gitignore` never did. Nothing was
committed: the working tree was clean and the files were hours old.

### The orphan check, and what it found

`extra="ignore"` is load-bearing — without it a stale `.env` key is a hard
`ValidationError` at import and the bot does not start. The cost is that deleting
a field silently demotes its `.env` key to decoration. That is how
`MIN_CONFIDENCE_SCORE`, `RERANKER_QUERY_TEMPLATE`, `BM25_AVG_LEN` and
`RETRIEVAL_TOP_K` each survived long enough to be tuned by hand.

`config.unknown_env_keys()` is the counterweight: ignore the key, name it once
per process. On its first run against the deployed `.env` it found
**`PIPELINE_MODE`, dead for 109 days** — its field was deleted in `73bb90a` on
2026-06-19, and the removal spec had not missed it:

> **`.env`** — remove line `PIPELINE_MODE=agentic`.
> `model_config` does not set `extra="forbid"`, so a leftover env line is
> harmless — but we remove it anyway to keep config honest.

Knowing was never the problem. The `.env` is untracked and lives on a host no
commit reaches, so the single step that mattered was the only one nothing could
verify. It was also the first orphan *not* present on the dev machine, which
answered whether the two files had drifted.

**The check is host-side only**, and deliberately so: `.env` is dockerignored and
injected via `env_file:`, so inside the container the file does not exist and
there is no set of "keys the operator meant as settings" to compare against. On
`srv-agent-01` neither place has both the deps and the file, so it has to be
handed in on a mount — which is why `unknown_env_keys()` takes text, not a path.

---

## Two things this write-up got wrong first

**The threshold is the union, not the sum.** Follow-up #12 of the sparse results
recorded it as "a caller asking for more than their sum would silently
under-return" — implying >40. The real bound is the union, which is routinely
well under 40 and is *smallest* when the hybrid is working best.

**The reranker backend.** The 12-of-61 five-source rows were attributed to
`RERANKER_BACKEND=vllm`, with a recommendation to switch to `vllm-score` to take
the decision back from the server. The deployed `.env` already said `vllm-score`,
so the recommendation was void — and `_call_score` posts all 25 documents and
trims to 6 locally, so it should return exactly 6 every time. The anomaly is less
explained than before, not more.

---

## Scope

19 test functions added, 2 removed; 471 collected. 140 insertions / 77 deletions
across 5 source files.

---

## Follow-ups, in priority order

1. **12 of 61 questions return 5 sources, not `RERANKER_TOP_N=6`.** Not the
   relevance floor — 6-source rows keep rerank scores as low as 0.0071, so
   nothing is being cut client-side. `_last_search_results` neither dedupes nor
   trims. On `vllm-score`, `_call_score` scores all 25 and takes `[:6]`, which
   should be deterministic. That leaves `/v1/score` returning fewer entries than
   documents sent, and the span cannot currently show it: `reranker.candidates_in`
   records what was sent, nothing records what came back before the trim. One
   attribute would make it measurable.
2. **`RERANKER_MIN_SCORE` is undeclared in the deployed `.env`**, so the live
   relevance floor runs on the `config.py` default of 0.2. It gates every
   question. Declare it at the value you mean.
3. **Citation provenance.** Nothing verifies cited chunks came from the retrieved
   set. Carried from `2026-09-25-search-first-results.md` #1 and
   `2026-10-02` #2 — still the largest correctness gap.
4. **Three `Settings` fields should be `SecretStr`** — `hf_token`,
   `teams_client_secret`, `teams_refresh_token`. Was four; `smtp_password` is
   gone. One-line fix, repo-wide effect.
5. **Delete `compliance_policies`** once v2 is trusted. Carried from `2026-10-02`
   #1; it remains the rollback.
6. **`scripts/test_query.py` has no sparse preflight.** Carried from `2026-10-02`
   #4. It is also the entry point that would actually exercise
   `min_confidence_score`, which is now the only reason that setting is kept.
7. **Extract `cosine_floor_applies()`.** The predicate
   `not reranker_enabled and not bm25_enabled` is in three places since the eval
   metadata mirrors shrank by one. Carried from `2026-10-02` #6.
8. **`RERANKER_QUERY_TEMPLATE` is set in the deployed `.env` but inert** on the
   `vllm-score` backend, where `RERANKER_INSTRUCTION` applies instead. Not an
   orphan — the field exists — so `unknown_env_keys()` cannot catch it. The
   family's remaining shape: a setting that is read, but only on a configuration
   you do not run.
9. **`PHOENIX_ENDPOINT` in the deployed `.env` points at `localhost`** and is
   overridden by compose. Harmless, and a line that states something untrue.
