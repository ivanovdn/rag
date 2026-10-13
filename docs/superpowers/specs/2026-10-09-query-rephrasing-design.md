# Query rephrasing for retrieval — design

**Date:** 2026-10-09 · **Status:** built and measured (`feat/query-rewrite`, `1a3af31`) — all three modes fail the stop rule; ships `off`. Results at the end. · **Baseline:**
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

## Results (2026-10-09)

Four tier1 runs on `chatbot-test-v1` (61 questions), commit `1a3af31`, the
production profile: `compliance_policies_v2`, BM25 on, reranker
`qwen3-rerank-4b` 25 → 6, floor 0.2, LLM `qwen3.6:latest` @ `07d35212591f`.
Retrieval only — no answering model. Compared with `eval/compare_runs.py`
against `rewrite-off`.

| Phoenix experiment | id | prompt sha12 |
|---|---|---|
| `rewrite-off` | `RXhwZXJpbWVudDoxMg==` | — |
| `rewrite-multi` | `RXhwZXJpbWVudDoxMw==` | `e270ed7a00e0` |
| `rewrite-multi_titles` | `RXhwZXJpbWVudDoxNA==` | `a911c01c0809` |
| `rewrite-hyde` | `RXhwZXJpbWVudDoxNQ==` | `5ffca5a1bbaa` |

**Control first.** `rewrite-off` reproduces `baseline-identity-v1` (chatbot
tier, `01a235a`): identical aggregates, and the first expected clause at the
same rank for all 61 questions — the new tier1 path is the production path.
Its top-6 lists differ from the baseline's in 14 places, all explained: 12 are
the chatbot tier collapsing split chunks of one clause before recording
(`search_results` there is deduplicated, tier1's is not — compare tier1 only
with tier1), and 2 are a swap at positions 5-6 between borderline candidates.
That is the run-to-run noise floor: about 2/61 lists change at the tail, and
expected-clause ranks do not move.

| mode | recovered (of 4) | lost | better / worse | hit | MRR | section hit | added latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| `off` | — | — | — | 0.934 | 0.770 | 0.951 | — |
| `multi` | 1 — thank-you fee | **0** | 2 / 3 | 0.951 | 0.769 | 0.951 | 0.65s / 0.86s |
| `multi_titles` | 1 — vendor without contract | **0** | 1 / 3 | 0.951 | 0.768 | 0.967 | 0.84s / 1.19s |
| `hyde` | 1 — thank-you fee | **1** | 0 / 2 | 0.934 | 0.752 | 0.934 | 0.86s / 1.03s |

No fallbacks in any run (0/61 each); one cold-load outlier of 5.8s in `multi`.
Every better/worse move is 1-3 ranks.

**Stop rule (≥ 2 of 4 misses recovered, no working question lost): all three
fail.** Each recovers exactly one miss, and not the same one — which miss is
recovered depends on the wording the model happened to write, so a single
recovery is not evidence of a dependable effect.

What the rephrasings showed, read question by question:

- **`multi` rephrasings are faithful and formal**, 122 of 122: no answers, no
  added facts. The recovery came from clause vocabulary — *"accept a personal
  gratuity … in exchange for approving"* reached *Rules and Measures*. The only
  mode with a clean record (0 lost) and the cheapest.
- **`multi_titles` names a policy in 59 of 122 rephrasings** (`multi`: 0),
  despite the prompt telling it not to. That turns a rephrasing into a title
  match: *"… permitted under the Anti-Bribery And Anti-Corruption Policy?"*
  finds the right document and not the clause, losing the thank-you-fee
  recovery `multi` had; *"… under the Mobile Device And Teleworking Policy?"*
  names the expected document and still does not reach its clause. Its one
  recovery came from clause-like wording (*"disclosure of Team Member personal
  data … third-party supplier"*), not from a title. Rejected.
- **`hyde` invents organisational detail** absent from the corpus — *"the
  Compliance Department or via the Ethics Hotline"*, *"the Chief Information
  Security Officer"*, *"a signed Data Processing Agreement … Record of
  Processing Activities"* — and that vocabulary steers retrieval: it lost AUP
  *Reporting* on "colleague sharing confidential files, should I report it?"
  (rank 2 → gone) and moved "is my data shared with external vendors?" from
  rank 1 to 4. The one mode that lost a working question. Rejected.
- **Misses no mode reached:** "what happens if I don't report a breach?"
  (*Non-Compliance Implications*) and "personal laptop to access company files"
  (*Device Usage and Encryption*; follow-up #10 records that this label
  contradicts its neighbour's, so it may not be a retrieval miss at all).

### Is reranking against the original what limits recovery? No.

Rule 2 scores every candidate against the original question only, which raised
the question whether that was why so little was recovered: a rephrasing might
pull the right clause into the pool and the reranker then drop it against the
user's informal wording. Checked read-only for the 4 misses: each mode's
25-candidate pool rebuilt from the rephrasings recorded in its experiment (no
LLM calls), with the repo's own retrieval code against the production
endpoints, then reranked. Reconstruction validated against the recorded `off`
run first — same final top 6 and rerank scores within 0.001 on 8 questions;
the only differences are ±1 swaps of fused rank between RRF-tied candidates.

| missed question | expected clause in the 25-pool? | rerank vs original | vs each rephrasing |
|---|---|---|---|
| thank-you fee | `off` no; `multi` #10, `multi_titles` #7, `hyde` #3 | 0.911 — top 6 in `multi`/`hyde`, misses by 0.009 in `multi_titles` | 0.843 / 0.582 |
| don't report a breach | **yes, already in `off`** (#12) | 0.564 → #10 (6th 0.700) | 0.048 / 0.066 |
| vendor without contract | only `multi_titles` (#4) | 0.708 → top 6 | 0.621 |
| personal laptop | only `multi_titles` (#22) | 0.343 → #14 | 0.203 |

- **Scoring against the rephrasings would never have helped**: in every case
  the expected clause scored *lower* against them than against the original,
  so even a max(original, rephrasing) score reproduces today's outcome. Rule 2
  stands.
- **When a rephrasing gets the clause into the pool, the reranker almost always
  takes it** — both recoveries happened that way. The rewrite's whole job is
  recall into the 25, and it achieves that on some wordings and not others.
- **"Don't report a breach" is not a recall problem**: the clause is in the pool
  with no rewrite and the reranker places it 10th. No rewrite can fix it; it is
  a reranker judgement or a labelling question (it prefers *Initial Reporting*,
  which arguably answers "what happens if I don't report" too).
- **Personal laptop** scores 0.343 even when found — consistent with
  follow-up #10's finding that its label contradicts its neighbour's.

So of the 4 misses, 2 are recall (rewriting can reach them, by luck of
wording), 1 is reranker-or-label, 1 is label.

### Does the reranker do better when it sees the chunk's header? Slightly.

The three retrieval stages read different text. The dense embedding was built
from `Document: … | Section: … | Clause: …\n<text>` (`ingest/pipeline.py`); BM25
encodes the bare text (`rag/vector_store.py`); the reranker scores the bare text
(`rag/reranker.py`, `_build_documents`), so it never sees that a clause is
called *Disciplinary Action* in a section called *Non-Compliance
Implications*. Checked read-only on all 61 questions: the production `off` pool
of 25 per question, reranked twice — bare text, and the same text behind the
embedding's header. No LLM calls. Sanity: the bare rerank reproduced the
recorded production top 6 on 60/61 questions.

| | bare text (production) | with header |
|---|---|---|
| expected clause in top 6 | 57/61 (0.934) | **58/61 (0.951)** |
| MRR@6 | 0.770 | **0.783** |
| recovered / lost | — | **1 / 0** |
| better / worse inside the top 6 | — | 9 / 7 |
| median top rerank score | 0.994 | 0.996 |

- **Recovered: "what happens if I don't report a breach?"**, 10 → 3, the miss the
  pool check above attributed to the reranker. Its clause is reached through
  the section name *Non-Compliance Implications*, which the bare text lacks.
- **Follow-up #10's harmful case moves the right way**: on "if the data wasn't
  very sensitive, do we still need to report it?", *Initial Reporting* — the
  clause the answer should have cited — goes 2 → 1. Retrieval only; the model
  still chooses what to cite.
- **The 7 worse are all reorders inside the top 6, none leave it.** The header
  lets a matching document title lift *sibling* clauses: on the Kyiv air alert,
  *Natural Phenomena* (same Business Continuity policy) scores 0.998 above
  *Air Alert Procedure* (1 → 4); suspicious activity puts *InfoSec General
  Responsibilities* above *Incident Response Reporting*; a deletion request puts
  the *Exhibit A/B* request forms above *Initiating* (arguably also useful).
  Scores bunch above 0.99, so small differences reorder.
- **Same size of gain as `multi`, without its cost**: no LLM call, no latency,
  and it fixes a different class of miss (in the pool, ranked low) than a
  rewrite can.

Re-run 2026-10-13, after the reranker endpoint had been briefly unreachable
and came back: same headline (recovered 1, lost 0, top 6 0.934 → 0.951, header
MRR@6 0.783), better 8 / worse 7. The bare arm now differs from the recorded
production top 6 on 11/61 questions (1/61 before), because the server's scores
moved a little — median |Δ| 0.001, p90 0.009, max 0.034, mean signed 0.000 on
306 matched chunks — which reorders near-ties. No systematic shift, and both
arms run against the same server in one pass, so the comparison holds; but the
VM runs should take their own `off` baseline the same day rather than reuse
the 2026-10-09 one.

Not shipped from this check: one dataset, retrieval only. Next is a setting
(`RERANKER_DOC_HEADER`, default off) on its own branch, measured with tier1 and
one full chatbot run with answers read.

## Decision and what stays open

- **Ships `QUERY_REWRITE=off`.** On well-formed questions there is almost
  nothing for a rewrite to fix: the clean set's misses are few, partly label
  noise, and recovered by luck of wording.
- **`multi` is the only candidate** if rewriting is revisited; `multi_titles`
  and `hyde` are rejected for the reasons above.
- **The question this was for is unanswered.** The motivation was users less
  precise than the dataset, and the dataset contains none. Per the stop rule
  the invented messy dataset is not built. The better evidence will be real:
  every production `compliance_request` span records the question verbatim,
  so once real traffic exists, the questions that escalated or retrieved
  poorly are the test set for `multi` — via `compare_runs` on a dataset built
  from them.
- Kept regardless of the decision: tier1 now measures the shipped retrieval
  path; `compare_runs.py`; the rewrite identity and its prompt-registry mirror.
