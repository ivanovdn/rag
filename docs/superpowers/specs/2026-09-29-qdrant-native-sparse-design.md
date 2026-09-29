# Qdrant-Native Sparse Vectors — Design

**Date:** 2026-09-29
**Status:** approved, not yet implemented
**Supersedes:** follow-ups #2 and #3 of `2026-09-25-search-first-results.md`

## Problem

BM25 works, and it is the best-measured retrieval configuration we have
(`hit_evaluator` 0.9344 with BM25 versus 0.9180 dense-only on `chatbot-test-v1`).
It is also unshippable in its current form.

The lexical index is a JSON file, `.bm25_index.json`. It is built only during
ingest and only when BM25 is *already* enabled, it is gitignored, and it is not
in the Docker image. Enabling BM25 in production therefore needs the file built
on the host and mounted into the bot container, which
`docker-compose-remote.yml` does not do — so `BM25_ENABLED=true` alone gives the
bot an empty index and silently drops the lexical half of every search.

Worse, the file desynchronises invisibly. `chunk_id` is a fresh `uuid4()` per
ingest, so an index built before a re-ingest has the right chunk *count* and
entirely wrong ids. Observed on 2026-09-25: 1602 chunks in the index, 1602
points in Qdrant, and 0 of 25 sampled ids present in both. `hybrid_search` then
fused the vector list against nothing and reported no error.

Every failure in that paragraph is quiet. Qdrant stores sparse vectors natively,
in the same collection, in the same point, upserted in the same call — which
deletes the file, the mount, the rebuild script, and the entire class of bug.

## What was verified

Everything below was measured on 2026-09-29 against throwaway Qdrant 1.17.1
containers (removed afterwards) and read-only probes of the production server.
None of it is inferred.

| Claim | Result |
|---|---|
| Sparse vectors can be added to an existing collection | **No.** `PATCH /collections/…` with `sparse_vectors` returns `Wrong input: Not existing vector name error: bm25`. Migration requires a new collection. |
| Unnamed dense can coexist with a named sparse vector | **Yes.** `{"vectors":{"size":768,"distance":"Cosine"},"sparse_vectors":{"bm25":{"modifier":"idf"}}}` creates and reads back correctly. The dense search path needs no renaming. |
| Qdrant's `idf` modifier formula | `ln(1 + (N−df+0.5)/(df+0.5))`. Measured 1.145132 at N=10, df=3; our `rag/bm25_index.py` computes the identical value. |
| Self-hosted server-side `qdrant/bm25` inference | **Works, with no InferenceService.** Upsert and query by text both succeed; `"software installation"` retrieved a document containing `"install"`, so it stems. A *bogus* model name errors with `InferenceService URL not configured`, so only the built-in models work self-hosted. |
| Collection-level `Bm25Config` (`k`/`b`/`avg_len`) | **Accepted and silently discarded** by 1.17.1. The `PUT` returns `ok`; reading the collection back shows only `{"modifier": "idf"}`. |
| Per-document `options` | **Works.** `avg_len=50` changed stored values from 1.6609 to 1.5428 versus the 256 default. |
| Query API `prefetch` + RRF fusion over unnamed dense + named sparse | **Works**, over REST and through `qdrant_client` 1.17.0. |
| `cloud_inference=True` against a self-hosted server | **Works**, producing a score identical to the default (0.9106 both ways). |
| Corpus statistics | 1602 chunks, mean 49.7 tokens, median 38, p90 109, max 350, zero empty. |

Server: Qdrant 1.17.1 at `172.20.0.22:6333`. Client: `qdrant-client` 1.17.0.
`fastembed` is not installed and is not added by this design.

## Decisions

**D1 — Replace the file-based BM25 entirely.** `rag/bm25_index.py`,
`rag/hybrid_search.py` and `scripts/build_bm25_from_qdrant.py` are deleted, not
kept behind a flag. Rollback is switching the collection back, not switching
code back. Keeping both would preserve the desync bug class in the tree, which
is the thing this work exists to remove.

**D2 — Server-side `qdrant/bm25` inference, not a client-side encoder.** Ingest
and query send text; Qdrant tokenizes, weights, and applies IDF. This adds no
dependency, deletes roughly 400 lines with nothing replacing them, and gets
stemming that our tokenizer does not have.

The cost is that ranking will not exactly reproduce the file-based index, whose
+1-case gain was measured with our own tokenizer. The eval is the gate (D9), and
D10 is the fallback if it fails.

`fastembed` was considered and rejected: now that server-side inference is
verified, fastembed delivers approximately the same encoding client-side while
adding onnxruntime, huggingface-hub and a runtime artifact download to a Docker
image on a host we do not control.

**D3 — `cloud_inference=True` on the Qdrant client.** The flag is badly named: it
means "do not encode locally, send the text to the server," which is exactly
what we want from a self-hosted server that performs inference. It was verified
to work self-hosted and to produce an identical score.

This is load-bearing, not cosmetic. The client's default is
`cloud_inference=False`, which means *encode locally via fastembed*. Today that
falls through to the server only because fastembed is absent. Anyone installing
fastembed for an unrelated reason would silently relocate BM25 encoding from the
server to the client, changing retrieval with no error and no log line. The flag
pins the behaviour.

**D4 — New collection plus a `QDRANT_COLLECTION` switch, not an alias and not an
in-place recreate.** Sparse cannot be added in place, and recreating
`compliance_policies` would leave the bot answering nothing for the duration of
the migration with no rollback.

An alias was considered. It buys an atomic, restart-free data switch — but this
migration ships code changes too, so the bot is redeployed regardless and the
atomicity buys nothing here. Ordinary re-ingestion is already incremental
(`delete_document(doc_id)` then upsert, per document), so the collection is
never empty in normal operation; only this one schema change needs a second
collection. An alias is the right answer for a future migration that is
data-only, and adopting one then costs no more than adopting one now.

**D5 — Migrate by copying points, not by re-parsing the `.docx` corpus.**
`scripts/migrate_collection.py` scrolls all 1602 points from the source
collection and re-upserts them into the target with the same ids, the same
payloads and the same dense vectors, adding only the sparse vector.

This makes the dense half of retrieval *provably* unchanged, so any eval delta
is attributable to the sparse half alone — which is the whole point of running
the gate. It also skips re-embedding entirely: no Ollama round-trip, no risk of
embedding drift from a model or prefix change since the original ingest, and no
dependency on the `.docx` corpus being present wherever the migration runs.

Re-ingesting from source remains available and needs no new code — the
`QDRANT_COLLECTION` env var already overrides the target.

**D6 — Index `chunk.text` for sparse, not the metadata-prefixed text.** The
dense embedding prepends `Document: … | Section: … | Clause: …`; the sparse
vector will not.

The encoder is already the variable under test. Indexing prefixed text at the
same time would change two variables at once and make a bad result
undiagnosable. Adding the prefix to the sparse side is a named follow-up
experiment, and it is cheap — a re-run of the migration script against a fresh
collection.

**D7 — `BM25_ENABLED` survives, as a query-side-only toggle.** Sparse vectors
are written unconditionally at ingest; the flag decides only whether the sparse
prefetch is included in the query.

This is strictly better than today, where the same flag also gated index
*building* — the root of the desync bug. It also preserves the A/B knob wanted
for parameter experimentation, and flipping it now requires no re-ingest.

**D8 — `bm25_avg_len` is a setting, defaulting to 50.0.** Qdrant's default is
256. Our corpus averages 49.7 tokens, so at 256 the length-normalisation term
`(1 − b + b·dl/avg_len)` stays near 0.25 for every chunk, `b` becomes nearly
inert, and long chunks go unpenalised across a corpus spanning 38 to 350 tokens.
Since the collection-level config is silently discarded (see the table above),
this value must be passed in `options` on *every* document and *every* query.

**D9 — The gate is `hit_evaluator` ≥ 0.9180 on `chatbot-test-v1`.** That is the
dense-only score, so native sparse must prove that lexical retrieval still
helps, but need not reproduce the file-based index's 0.9344 exactly. The
difference between those two numbers is a single question out of 61, which is
inside noise, while the migration's real payoff is structural.

**D10 — Below the gate, build a client-side encoder rather than abandoning the
migration.** Reuse `_tokenize` from the deleted `bm25_index.py` with
`k1=1.2, b=0.75, avg_len=49.7`, a stable `hashlib`-based token-to-index map, and
values computed as `tf·(k1+1)/(tf + k1·(1 − b + b·dl/avg_dl))`, sent as an
explicit `SparseVector`. Because Qdrant's IDF formula is already verified
identical to ours, this reproduces the exact ranking that produced 0.9344. The
collection schema, the fusion query, the migration script and the rollout are
unchanged — only the encoding call site differs.

## Collection schema

`compliance_policies_v2`, created by `init_collection()`:

- **dense: unnamed**, size `settings.qdrant_vector_dim` (768), `Distance.COSINE`
  — byte-identical to today
- **sparse: `bm25`**, `modifier: Modifier.IDF`
- the same six payload indexes as today: `doc_id`, `section`, `section_number`,
  `clause`, `clause_number` as `KEYWORD`, `section_display` as `TEXT`

`init_collection()` gains an optional parameter —
`init_collection(collection_name: str | None = None)`, defaulting to
`settings.qdrant_collection` — so the migration script can create a target other
than the configured one without duplicating the schema definition.

Points address the dense vector by the empty-string key:

```python
vector={"": embedding, "bm25": models.Document(...)}
```

## Ingest path

`upsert_chunks` writes both vectors per point:

```python
PointStruct(
    id=chunk.chunk_id,
    vector={
        "": embedding,
        "bm25": models.Document(
            text=chunk.text,
            model="qdrant/bm25",
            options={"avg_len": settings.bm25_avg_len},
        ),
    },
    payload=chunk.model_dump(),
)
```

Both `if settings.bm25_enabled:` blocks in `ingest/pipeline.py` — the
`remove_document_from_bm25` call and the `add_chunks_to_bm25` call — are
deleted. Sparse is written unconditionally (D7).

## Retrieval path

`rag/vector_store.py`: `search_vectors(query_vector, top_k)` becomes
`search_chunks(query_text, query_vector, top_k)`.

```python
if settings.bm25_enabled:
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(query=query_vector,
                            limit=settings.hybrid_vector_candidates),
            models.Prefetch(
                query=models.Document(text=query_text, model="qdrant/bm25",
                                      options={"avg_len": settings.bm25_avg_len}),
                using="bm25",
                limit=settings.hybrid_bm25_candidates),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
else:
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
```

Both branches return `list[ScoredPoint]` with identical payload shape. That is
the structural win: fusion moves server-side, so the roughly 100 lines of
dict-merging RRF in `hybrid_search.py` evaporate, **and** the two divergent
branches in `rag/tools/search_policies.py`, `scripts/run_eval.py` and
`eval/run_experiment.py` each collapse into one path. Only `score_type`
(`"rrf"` versus `"cosine"`) still varies, and it is carried for observability.

The OpenTelemetry span keeps the name `search_vectors` even though the function
is renamed to `search_chunks`, so Phoenix comparisons against historical runs
stay valid. It is extended with `qdrant.bm25_enabled`,
`qdrant.fusion`, `qdrant.bm25_avg_len`, plus the existing
`qdrant.collection` / `qdrant.limit` / `qdrant.returned_count` /
`qdrant.top_score` and the `retrieval.documents.*` attributes.

### The `min_confidence_score` trap

`min_confidence_score` (0.45) is compared against a **cosine** score. Today it is
guarded by `if not settings.reranker_enabled`, and the hybrid branch never
reaches it because that branch returns early with RRF-scored dicts.

Once both modes share one code path, that guard is no longer sufficient. The
condition must become:

```python
if not settings.reranker_enabled and not settings.bm25_enabled and \
        raw[0].score < settings.min_confidence_score:
```

An RRF score is around 0.016, so applying a 0.45 cosine threshold to it would
return `NO_RELEVANT_POLICY_FOUND` for every question — escalating the entire
corpus with no error. The guard is on the score's *meaning*, not on the reranker
alone. This is the same class of bug as the `rerank_score`-presence guard in
`search_policies.py`, and it gets its own regression test.

## Migration script

`scripts/migrate_collection.py`, following the conventions of
`scripts/build_bm25_from_qdrant.py` (which it replaces):

- `--source` (default `settings.qdrant_collection`), `--target` (required)
- `--dry-run`: report source count, target existence, and the planned action
  without writing
- `--verify`: after migration, assert target count equals source count, assert
  the target's sparse config contains `bm25` with `modifier: idf`, sample 25
  point ids and confirm each exists in the target with a non-empty sparse vector
- creates the target via `init_collection()` against the target name, so schema
  and payload indexes come from one definition and cannot drift
- scrolls the source with `with_payload=True, with_vectors=True` in pages of 500
- upserts in batches of 100, reusing the source `id`, `payload`, and dense
  vector verbatim, adding the sparse `Document` built from `payload["text"]`
- refuses to run if the target already exists and is non-empty, unless `--force`

## Deletions

| Path | Lines |
|---|---|
| `rag/bm25_index.py` | 230 |
| `rag/hybrid_search.py` (includes `hybrid_search_formatted`, already dead — its only caller is inside the same file) | 166 |
| `scripts/build_bm25_from_qdrant.py` | 149 |
| the two BM25 blocks in `ingest/pipeline.py` | 8 |

That is 545 lines of deleted module code, replaced by roughly 30 lines of
prefetch construction and a migration script.

Also removed: the `.bm25_index.json` entry in `.gitignore`, the
`BM25_ENABLED=true silently does nothing` gotcha row in `CLAUDE.md`, and the
stale `hybrid_search` example in `rag/observability.py`'s `get_tracer` docstring.

## Config

| Setting | Change |
|---|---|
| `qdrant_collection` | unchanged in code; switched in `.env` at cutover |
| `bm25_enabled` | kept, now query-side only (D7) |
| `hybrid_vector_candidates` (20) | kept, now the dense prefetch limit |
| `hybrid_bm25_candidates` (20) | kept, now the sparse prefetch limit |
| `bm25_avg_len` | **new**, `float = 50.0` (D8) |

`.env.example` gains `BM25_AVG_LEN=50.0` and a comment that `BM25_ENABLED` no
longer requires a re-ingest to flip.

## Error handling

Sparse retrieval rides inside the existing Qdrant call, so `retry_transient` and
`record_infra_unavailable("qdrant", …)` cover it with no new failure mode. This
is an improvement on today: a broken lexical half now fails loudly instead of
fusing against nothing.

One genuinely new failure exists. Pointing `BM25_ENABLED=true` at a collection
with no `bm25` sparse vector — mid-rollback, or a fresh environment — makes
Qdrant return `Not existing vector name error`. That is non-transient, so it
would surface as a content escalation on *every* question: the bot would look
like it could not find any policy.

The fix is a **startup preflight** — a function in `rag/vector_store.py` called
from `scripts/start_teams_bot.py` after `init_observability()` and before the
bot is constructed: when `bm25_enabled` is true, read the collection's
sparse config and refuse to start with an explicit message naming the collection
and the missing vector. Deliberately not an auto-disable — silent degradation is
the exact failure this migration exists to remove, and a bot that quietly drops
to dense-only looks healthy while answering worse.

## Testing

All unit tests stay offline. Nothing touches the network or `172.20.0.22`.

- **Request shape**, with `client.query_points` mocked: with `bm25_enabled=True`,
  two `Prefetch` entries are sent, the sparse one carries `using="bm25"` and a
  `Document` with `model="qdrant/bm25"` and `options={"avg_len": …}`, the query
  is `FusionQuery(RRF)`, and both limits come from settings. With
  `bm25_enabled=False`, no `prefetch` and no `FusionQuery` are sent.
- **`avg_len` threading**: changing `settings.bm25_avg_len` changes the value in
  both the upsert Document and the query Document.
- **`cloud_inference=True`** is set on the constructed client (D3).
- **`min_confidence_score` regression**: with `bm25_enabled=True` and
  `reranker_enabled=False`, a top score of 0.016 does **not** produce
  `NO_RELEVANT_POLICY_FOUND`. This test is the guard on the trap above.
- **Preflight**: a collection whose sparse config lacks `bm25` fails startup with
  a message naming the collection; with `bm25_enabled=False` it does not.
- **Migration script**: `--dry-run` writes nothing; a point is copied with id,
  payload and dense vector preserved; `--verify` fails on a count mismatch.
- The three existing tests that pin `bm25_enabled=False`
  (`test_retrieval_spans.py`, `test_request_span.py`, `test_search_floor.py`)
  keep passing and gain `True` mirrors.

Per the standing constraint, tests must bind `settings` values to locals before
asserting — a pytest `AttributeError` or assertion touching `settings` directly
embeds the full `Settings` repr, which contains real `.env` secrets.

## Rollout

Run by the operator on the VM. Nothing in this sequence touches server-side
Qdrant configuration, only our own collections.

1. **Deploy the code.** `QDRANT_COLLECTION=compliance_policies`,
   `BM25_ENABLED=false`. The dense path is unchanged, the preflight passes
   trivially, and behaviour is identical to today. This step is independently
   revertible.
2. **Build v2 alongside.**
   `PYTHONPATH=. python scripts/migrate_collection.py --target compliance_policies_v2`
   The live bot is still reading `compliance_policies` and is not affected.
3. **Verify.** `--verify` must report 1602 of 1602 points and 25 of 25 sampled
   ids present with non-empty sparse vectors, and the collection must read back
   with `sparse: {"bm25": {"modifier": "idf"}}`.
4. **Run the gate.** `chatbot-test-v1` against v2 with `BM25_ENABLED=true`.
   Compare `hit_evaluator` to 0.9180. Per the standing gotcha, the eval container
   must mount `config.py`, `rag/`, `eval/` and `scripts/` together, or it runs
   main's pipeline against the new harness.
5. **Cut over, or don't.** At or above 0.9180: set
   `QDRANT_COLLECTION=compliance_policies_v2` and `BM25_ENABLED=true`, redeploy.
   Below: nothing has been switched; implement D10 and return to step 2 with a
   fresh target collection.
6. **Keep `compliance_policies`** as the rollback until the new collection has
   run in production long enough to trust. Deleting it is a separate, manual,
   explicitly-requested action.

Rollback at any point after step 5 is: restore both `.env` values, redeploy. The
old collection is untouched throughout.

## Out of scope

- **Aliases** (D4). Worth adopting at the next data-only migration.
- **Prefixed sparse text** (D6). First follow-up experiment after the gate.
- **Tuning `k` and `b`.** `avg_len` is set from measurement; `k` and `b` keep
  Qdrant's 1.2 and 0.75, which match our current index. Tuning them is an
  experiment on the live collection, needing no code change beyond D8's pattern.
- **`retrieval_top_k` / `reranker_candidates` retuning.** Fusion changes what
  the candidate window means, but changing it during this migration would
  confound the gate.
- Follow-ups #1 and #4 through #9 of `2026-09-25-search-first-results.md` are
  untouched. #2 and #3 are superseded by this document.

## Gotchas discovered, for `CLAUDE.md`

- **Collection-level `Bm25Config` is accepted and silently discarded** by Qdrant
  1.17.1. `PUT` returns `ok` and the collection reads back with only
  `{"modifier": "idf"}`. `k`, `b` and `avg_len` work *only* in per-document
  `options`, and must therefore be passed on every upsert and every query.
- **`cloud_inference=False` is the client default and means "encode locally via
  fastembed".** Server-side inference works today only because fastembed is
  absent. Installing it for any reason silently moves BM25 encoding from the
  server to the client. Set `cloud_inference=True` explicitly.
- **Sparse vectors cannot be added to an existing collection.** `Not existing
  vector name error: bm25`. Any future vector-schema change means a new
  collection and a migration.
- **Self-hosted inference works only for built-in models.** `qdrant/bm25`
  succeeds; any other model name fails with `InferenceService URL not
  configured`, which requires server-side config on the shared host.
