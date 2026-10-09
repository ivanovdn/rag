# Query rephrasing for retrieval — design

**Date:** 2026-10-09 · **Status:** agreed design, not built · **Baseline:**
`baseline-identity-v1` (chatbot-test-v1, `01a235a`, prompt `42d2fec870f6`, input
`e4720ee79a85`, LLM `07d35212591f`)

## Problem

The user's message reaches every retrieval stage verbatim — dense embedding,
BM25, the reranker — and the answering model. That was deliberate (search-first
D2, 2026-09-24): the old agent rewrote queries on its own in 12% of searches,
against its prompt, unmeasured. It is now impossible, which also means nothing
bridges the gap when **a user's words are not the policy's words**.

The test set understates that gap. Its 61 questions are well-formed English,
median 12 words, shortest 6. Production has no real-user signal yet: 53
requests, 14 distinct questions, nearly all ours (Phoenix, 2026-09-22 → 10-07).

## Scope — what the user contract already rules out

The welcome message (`channels/teams/renderer.py` `WELCOME_HTML`) tells users:

- **"Ask your question in plain English."** No translation step. A non-English
  message is a router / reply-text concern, not a retrieval one.
- **"I don't remember earlier messages yet — put your whole question in one
  message."** No conversational (history-aware) rewriting until the bot keeps
  history. If that changes, this is where follow-up resolution would plug in.

What remains is **vocabulary mismatch**: vague phrasing, everyday words for
policy terms ("thank-you fee" vs kickbacks/Rules and Measures), and incidental
typos.

## Headroom on the current dataset

The clean set is not saturated, so it can be used before any messy variant
exists:

- 4/61 never retrieve the expected clause (hit 0.934), all vocabulary mismatches:
  "thank-you fee" → *Rules and Measures*; "vendor without a contract" →
  *Disclosure of Personal Data to Third Parties*; "don't report a breach" →
  *Non-Compliance Implications*; "personal laptop to access company files" →
  *Device Usage and Encryption*.
- MRR 0.770 — the right clause is often not first.
- **Dense-only** ranks for the worst of them (2026-10-09 probe, below): 72, 59,
  37, 19 of 1602. BM25 and the reranker rescue some, but these are the questions
  a different wording would have to reach.

## Ruled out first: EmbeddingGemma task prompts

The model card recommends `task: search result | query: {q}` for queries and
`title: {title | "none"} | text: {content}` for documents. Neither is used: both
prefixes default to `""`, and the stored v2 vectors were embedded **without** a
document prompt — re-embedding three stored chunks as
`Document: … | Section: … | Clause: …\n{text}` reproduces them at cosine
0.99998, while the `title: … | text:` forms give ~0.95.

Measured dense-only on all 61 questions against all 1602 chunks (stored vectors
read from Qdrant, document-prompt vectors re-embedded on the production
embedder `embeddinggemma:latest` @ `85462619ee72`, ranking done locally — no
write to the shared host):

| documents | query prompt | @1 | @6 | @25 | MRR |
|---|---|---|---|---|---|
| stored (no doc prompt) | none — **production** | 28 | 52 | 58 | 0.622 |
| stored (no doc prompt) | `task: search result \| query:` | 27 | 52 | 57 | 0.611 |
| stored (no doc prompt) | `task: question answering \| query:` | 25 | 51 | 58 | 0.584 |
| `title: {doc} \| text:` | none | 29 | 52 | 58 | 0.629 |
| `title: {doc} \| text:` | `task: search result \| query:` | 30 | 53 | 56 | 0.636 |
| `title: {doc} \| text:` | `task: question answering \| query:` | 28 | 48 | 55 | 0.605 |

Not a free win. A query prompt alone moves 11 questions up and 17 down; the full
model-card pair is a reshuffle (16 up, 15 down), with large individual losses
(a "should I report a colleague" question 4 → 57) against a better MRR by 0.014,
and it would cost a full re-embed into a new collection. **Not pursued.** The
deployed `.env`'s `EMBEDDING_QUERY_PREFIX` was not read; the table covers every
value it plausibly holds. Re-measure if the embedding model changes.

## Design

**Technique: multi-query** — the original question plus **2** LLM rephrasings,
retrieved together and fused. HyDE (embed a hypothetical policy passage) is
measured as a second variant through the same plumbing. Single rewrite is not
built: on English questions it mostly fixes grammar, which clean questions do
not need, and it replaces the original instead of adding to it.

```
question ─► rewrite: 1 LLM call → 2 rephrasings
         ─► embed all 3 in ONE /api/embed call
         ─► ONE Qdrant Query API call: 3 dense + 3 BM25 prefetches, RRF server-side
         ─► rerank against the ORIGINAL question ─► top_n ─► agent sees the ORIGINAL
```

**Where:** inside retrieval, before embedding — reached by `search_policies`
(production and the agent tiers) **and** by tier1. Tier1 today duplicates the
embed → `search_chunks` → rerank path in `eval/run_experiment.py`; the candidate
step must become one shared function both call, or tier1 would measure a
pipeline that does not ship.

**Rules:**

1. **The original is always one of the queries.** Rephrasings can only add
   candidates, never remove what the verbatim question found.
2. **The reranker and the answering model see only the original.** Relevance and
   citation are judged against what the user asked, not what the rewriter wrote.
3. **A rewrite failure never blocks.** Any error or unparseable output → search
   with the original alone, recorded on the span — the reranker's fallback
   pattern. It is not an `infra_unavailable` event: retrieval still works.
4. **Plain-text output, temperature 0, no tools, no `format`/structured output.**
   Constrained decoding is one of the five conditions of the documented Ollama
   MoE crash (gotchas table); the router's plain-text-JSON pattern is the safe
   precedent. `think: false`, `keep_alive` passed, fresh client per call (never
   cached).
5. **Sizing invariant kept.** `RERANKER_CANDIDATES` sizes every prefetch and the
   fused limit. Six prefetches make the union larger; the limit is unchanged, so
   the reranker's workload is unchanged.
6. **Recorded.** The rephrasings go on their own `query_rewrite` span
   (`query_rewrite.mode`, `.queries`, `.fallback`, `.latency_ms`, `.error`),
   which also parents the rewrite's LLM span; `search_vectors` gains
   `qdrant.query_count`;
   `rewrite_prompt_sha12` joins `rag/run_identity.py`, so it lands in experiment
   metadata and on every `compliance_request` span; `QUERY_REWRITE` is in the
   experiment metadata.

**Setting:** `QUERY_REWRITE=off|multi|multi_titles|hyde`, default `off` — the
deployed behaviour does not change until a measurement says it should.

## Decisions (agreed 2026-10-09)

- **A — policy vocabulary in the rewrite prompt:** measured as its own variant
  (`multi_titles`): the 52 policy titles plus a few corpus terms (Team Member,
  SOC, corporate workstation). Strongest lever and the likeliest to bias toward
  whichever policy *sounds* closest — hence measured, not assumed.
- **B — separate LLM call**, not folded into the router. Keeps the variable
  isolated; folding (≈ −2s) is a later, separate change with its own risk to the
  router's safe-default invariant.
- **C — 2 rephrasings.** A third mostly repeats the first two and adds output
  tokens.
- **D — measurement**, below.

## Measurement

1. Retrieval-only over the 61 chatbot questions (`--tier tier1 --dataset
   chatbot-test-v1`; `_extract_expected` already reads `expected_citations`),
   one run each for `off`, `multi`, `multi_titles`, `hyde`, all on the
   production profile (v2, BM25 on, reranker on, VM — never the Mac).
2. Judge **per question**, not by the mean: which of the 4 misses are recovered,
   which of the 57 working questions get *worse* (rank of first match), and the
   added latency (p50/p95 of the rewrite call). With n=61, one recovered miss and
   one new miss average to "no change".
3. The winner, if any, gets one full chatbot run with the `citation_*`
   evaluators re-enabled (follow-up #10 step 1), compared with
   `baseline-identity-v1` by identity metadata.
4. **Stop rule:** if no variant recovers at least 2 of the 4 misses without
   losing a working question, rephrasing is not worth its latency on this
   corpus — record it and do not build the messy-variant dataset for it.
5. Only if a variant passes: the messy-variant dataset (short/keyword, typos,
   vague/story-form, two-questions-in-one), same expected citations, owner
   review — to confirm the gain grows where real users will be.

## Out of scope

Translation; history-aware rewriting; decomposition of multi-part questions
(revisit if the messy set shows them failing); folding into the router;
re-embedding with EmbeddingGemma prompts (measured above).
