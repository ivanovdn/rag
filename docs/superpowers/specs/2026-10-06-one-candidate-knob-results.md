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

### A NameError shipped in the eval harness

Renaming `top_k` -> `candidates` in `main()` fixed two of three uses. The third
sat in the agent tiers' metadata dict and raised `NameError: name 'top_k' is not
defined` — after the banner printed, on a 61-question GPU run, past the version
assertion that is supposed to be the cheap failure point.

It shipped because the harness changed *after* the gate run, so nothing executed
that path again, and because the test guarding the rename inspects `main()`'s AST
rather than running it. `tests/unit/test_no_undefined_names.py` is the
dependency-free net: for every top-level function in `eval/` and `scripts/`, the
names it reads must be bound in its own subtree, at module level, or in builtins.
Mutation-verified against the exact bug.

Writing it surfaced the same class one level down — `out |= _bindings(s)` inside
a nested helper rebinds `out` as a local of that helper, so the checker crashed
with `UnboundLocalError` before it could check anything. `out.update(...)` does
not rebind.

### The `top_k` metadata key was recording the wrong number

Worth knowing before comparing old experiments. Through experiment 8 it recorded
`retrieval_top_k` (20) while retrieval used `reranker_candidates` (25);
experiment 9 recorded 25, because `e3d5c1d` changed where it read from. The same
field name therefore means two different quantities either side of that run. It
is now `candidates`, which draws the line visibly rather than leaving a key that
silently changed meaning.

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

1. ~~**12 of 61 questions return 5 sources, not `RERANKER_TOP_N=6`**~~ —
   **ANSWERED 2026-10-06: nothing was wrong.** `eval/run_experiment.py` dedupes
   `search_results` by `(doc_title, section, clause)` before writing the output,
   and twelve questions had two of the reranked top-6 sharing that triple.

   The reranker was never involved: `reranker.results_out` reads 6 on all 61
   spans of both experiments, and the probe run (experiment 10, 2026-10-06)
   measured `candidates_in → scores_returned → results_out` at **25 → 25 → 6 for
   every one of 61 questions**, zero fallbacks. Falsification holds too — no
   exported row contains a duplicate key, which must be true if the dedup ran.

   **Production is unaffected.** `format_sources` shows the agent all six; the
   dedup is eval-only.

   The key is also right rather than merely harmless: `evaluators._match_result`
   reads `doc_title`, `section` and `clause` and never `clause_number`, so two
   chunks of one clause are indistinguishable to every retrieval metric.
   `search_results_before_dedup` now records the pre-dedup count so the output
   says what it did to itself, guarded by a test that pins the key to
   `_match_result`'s fields.

   **Worth recording about the method, not the bug:** `reranker.results_out`
   already answered this and had existed all along. The detour — inferring the
   wrong reranker backend, then adding `reranker.scores_returned` — came from
   reasoning about the code instead of reading the span that was already there.
   The new attribute is still worth having, since it separates "the backend
   returned fewer" from "we trimmed", but it did not find this.

2. ~~**`RERANKER_MIN_SCORE` is undeclared in the deployed `.env`**~~ —
   **DONE 2026-10-06.** The value 0.2 was never the problem; the contradiction
   was. `.env.example` shipped `RERANKER_MIN_SCORE=0.0`, annotated "0.0 = off",
   while `config.py` defaulted to 0.2 — so a deployment that copied the template
   ran with no floor and one that omitted the line ran with one. Same repo, two
   safety behaviours, decided by whether a file was copied.

   Measured before changing anything: the floor has **never fired** — 0
   `retrieval_floor_rejected` spans across 45 reranked production questions,
   2026-09-22 to 2026-10-06, p10 0.9772, median 0.9981. The only sub-threshold
   scores on record are three askings of *"Can I bring penguin into office"* on
   2026-09-24, before the floor existed, all of which escalated correctly on
   model judgement. That is the case it exists for.

   `.env.example` now carries 0.2, pinned by a test: every relevance floor must
   match its code default and none may ship disabled. The deployed `.env`
   declares it explicitly, so the live value is readable in the file instead of
   inferred from Python. No behaviour change — 0.2 is what was already running.
3. ~~**Citation provenance**~~ — **CLOSED 2026-10-07: measured, not built.**
   Carried from `2026-09-25-search-first-results.md` #1 and `2026-10-02` #2, and
   called the largest correctness gap in both. It is not one.

   Measured against the 2026-10-06 11:03–11:09 UTC probe — 61 questions, 354
   retrieved chunks, 96 citations, BM25 and reranker both on — by checking every
   citation against its own `search_results`, and every quote against a fresh
   re-chunk of all 52 DOCX files (1,602 chunks, `ingest/docx_parser.py`):

       location matched a retrieved chunk, exact on
         doc_title + section + clause + clause_number   96/96
       quote verbatim in the chunk it cites             92/96
       quote verbatim but elided with "..."              4/96
       quote fabricated or altered                       0/96

   So the guard would have fired zero times. The four elided quotes split on the
   ellipsis into 8 fragments, every one verbatim in the correct chunk — meaning a
   naive verbatim check would have escalated 4% of *correct* answers and been a
   net loss. **Any future quote check must split on the ellipsis before
   comparing.** The strict location check is, separately, now measured safe: 0
   false positives in 96.

   0 of 96 bounds the fabrication rate below roughly 3% (rule of three), not at
   zero — one run, one model, one day, and with BM25 on, which production does
   not run. Re-measure after a model or retrieval change rather than inheriting
   this result.

   What the probe did expose is #10, which this guard would have passed in full.
4. ~~**Three `Settings` fields should be `SecretStr`**~~ — **DONE 2026-10-06**
   (`fix/secrets-as-secretstr`). `hf_token`, `teams_client_secret` and
   `teams_refresh_token` now mask themselves in both `repr()` and `str()`, so
   the convention that kept `settings` off assert lines — three comments in
   `tests/unit/test_llm_config.py`, enforced by nothing — became a property of
   the type. Measured after the change: all three load from the real `.env`,
   none appears in `repr(settings) + str(settings)`.

   Two things checked rather than assumed. `bool(SecretStr(""))` is `False` on
   pydantic 2.12.5 (it defines `__len__`), so the truthiness guards at
   `rag/embeddings.py:15` and `channels/teams/auth.py:29` keep working — had it
   been truthy, an unset refresh token would have silently seeded `""` instead
   of raising `No refresh token found`. And a missed `.get_secret_value()`
   raises rather than sending the mask: `TypeError` at `os.environ`, at
   `requests`' urlencode, and at `json.dump`.

   The call site with no behavioural cover is `rag/embeddings.py` (it only runs
   on `EMBEDDING_SOURCE=huggingface`, which production does not use), so the
   guard is an AST check in the shape of `test_no_undefined_names.py`: every
   `settings.<secret>` read is unwrapped or is the bare test of an `if`.
   Mutation-tested — removing the unwrap fails it by file and line.
5. **Delete `compliance_policies` on or after 2026-11-01**, provided no rollback
   has been needed by then. Carried from `2026-10-02` #1, where the condition was
   "once v2 is trusted" — which has no resolution, which is why it is still open.
   A date has one. Until then it is the rollback: restore the two `.env` values
   (`QDRANT_COLLECTION`, `BM25_ENABLED`) and redeploy. Deleting it costs disk on a
   host this project does not own and buys nothing else, so the only reason to
   hurry is if that host is short of space.
6. ~~**`scripts/test_query.py` has no sparse preflight**~~ — **DONE 2026-10-07.**
   Carried from `2026-10-02` #4. It was the last query entry point without one,
   and the one a person reaches for when working out what is wrong: the error is
   non-transient, so prefetch handed it a bare Qdrant traceback. Gated on a query
   actually being requested, so `--help` stays offline.
7. ~~**Extract `cosine_floor_applies()`**~~ — **DONE 2026-10-07.** Four places,
   not three: the guard in `search_policies` and three eval metadata mirrors
   (`run_eval` once, `run_experiment` twice). Now `Settings.cosine_floor_applies`,
   with a source check pinning that nothing re-derives it inline. Carried from
   `2026-10-02` #6.
8. ~~**`RERANKER_QUERY_TEMPLATE` is set in the deployed `.env` but inert**~~ —
   **DONE 2026-10-07**, as a detector rather than one deleted line, because this
   is the fourth member of the family and there was nothing that would find a
   fifth. `config.inert_env_keys()` holds a table of (key, when-it-is-inert, which
   knob applies instead) and reports at startup beside the orphan warning.

   It fires where `unknown_env_keys()` structurally cannot — inside the container.
   That docstring is right that there is no set of "keys the operator meant as
   settings" to enumerate there; the difference is that this never enumerates. It
   asks after two keys by name, and checks both `.env` and `os.environ` because
   the dev host has the file and the container has the variables.

   Write-time settings are deliberately excluded: `BM25_AVG_LEN` is inert at query
   time and live during ingest, so a warning would fire wrongly in `ingest_all.py`.

   Extracting the backend predicate onto `Settings` turned up an unrelated hole —
   nothing tested the reranker's wire format at all, and reverting `vllm-score` to
   the llama-server path left the suite green while silently sending an unwrapped
   query to a model that requires the chat template. `tests/unit/test_reranker_wire_format.py`
   now pins it.

   **Verified in production 2026-10-08**, a day late. The first deploy ran
   `git pull` without `--build`, so the container kept the previous image and the
   warning simply never appeared — which is how the deploy audit found it. After a
   rebuild it fired for `RERANKER_QUERY_TEMPLATE` alone (the deployed `.env` never
   set `MIN_CONFIDENCE_SCORE`). That line is now commented out on the VM, and the
   warning is gone.
9. ~~**`PHOENIX_ENDPOINT` in the deployed `.env` points at `localhost`**~~ —
   **DONE 2026-10-08.** Replaced on the VM by a comment saying compose sets it.
   No code change was available or wanted: `config.py`'s default is that value and
   compose overrides it regardless. The first attempt — a `sed` given without
   being tried — glued the old URL onto the end of the commented line; the second
   was tested on a copy of those lines first, which is the rule for anything run
   against a `.env` holding live secrets.
10. **The model cites the wrong retrieved chunk — mostly a labelling problem;
    one real defect, a role mismatch.** Parked 2026-10-08 as not critical.
    Originally recorded from the 2026-10-06 probe as three wrong-document answers,
    with the auto-lock question as the sharpest case. Re-measured on
    `baseline-identity-v1` (2026-10-08, `01a235a`, prompt `42d2fec870f6`, input
    `e4720ee79a85`, digest `07d35212591f`) by comparing every answer's citations
    with its `expected_citations`, then reading the clause text of each mismatch:

        cites the expected clause                              52/61
        expected clause retrieved, model cited another          5
        expected clause never retrieved (answer still sound)    3
        escalated (expected section never retrieved)            1

    Of the five selection cases, after reading the clauses:
    - **Auto-lock — correct, better than the label.** Access Management 4.14
      says *"Users are prohibited from modifying, disabling, or overriding…
      automatic lock settings"*; the label's Clear Screen 5.2 only states the
      5-minute rule. The headline example of this item was a right answer.
    - **Teams chats private? — correct.** AUP 7.3 (*"chat messages solely between
      Team Members are considered personal"*) is the direct clause; label 7.4.
    - **Free productivity app — correct.** AUP 4.3 (unapproved software) plus the
      Default Workstation approval route; the label's 4.7 is about *unlicensed*
      software, which "free" is not.
    - **Open source in a client project — incomplete.** Cites 3.2 (approval of
      client + team lead) but omits 1.1, rank 3: client use is *"usually strictly
      prohibited"*.
    - **"If the data wasn't very sensitive, do we still need to report it?" —
      wrong, and the harmful kind.** Cites Breach Procedure 6.3 (*"no
      notification is required"*), which is the DPO notifying a supervisory
      authority. The asker's duty is 5.1, retrieved at rank 2: *"Any Team Member
      aware of… a suspected personal data breach must report"* to the SOC. The
      answer reads as "no need to report".

    **The defect is role mismatch:** the model picks the clause whose wording
    matches the question but whose addressee (DPO, System Owner, controller) is
    not the asker. Ranking and `num_ctx` are ruled out — the right clause was
    at rank 2.

    **Bigger gap found on the way: nothing measures citations.** The three
    `citation_*` evaluators in `eval/evaluators.py` were dropped from the run
    set 2026-09-25; `hit_evaluator` (0.934) is retrieval. A prompt change can
    degrade citations today with every reported number unchanged.

    When picked up, in this order:
    1. Re-enable `citation_doc/section/clause_accuracy` (they ignore extra
       citations, so the 19 answers citing more than expected are not punished).
       No bot change.
    2. Relabel with alternatives + `match_mode: any`: auto-lock, Teams chats,
       productivity app, and "personal laptop to access company files" (whose
       label contradicts the AUP 4.2 label on the neighbouring personal-laptop
       question). Owner's sign-off — the dataset is ground truth.
    3. Add 6-8 role-mismatch questions (a nearby clause addressed to the DPO /
       System Owner / controller) — one failing case cannot show a fix worked.
    4. Only then try one prompt rule ("the asker is a Team Member; prefer the
       clause that says what they must do; if you cite one addressed to another
       role, say so") and compare against `baseline-identity-v1` by its identity
       metadata.
    Analysis script: classify each run's citations against `expected_citations`
    from `GET /v1/experiments/<id>/json` on Phoenix.
11. **The startup clamp measures idle time, not downtime.** (Deploy audit,
    2026-10-08.) `_save_state` rewrites `bot_state.json` every poll but records
    only `last_check` and the processed ids — never *when* it saved. So
    `_load_state` judges downtime by the watermark's age, and the watermark only
    advances when a newer message arrives: through a quiet spell it sits at the
    last message. A 4-second restart at 11:03:58 after a quiet morning clamped a
    09:21 watermark and warned of a backlog that did not exist. Two costs. The
    false alarm fires on nearly every deploy at current volume, which teaches the
    operator to ignore the one warning that matters on a real outage. And a narrow
    loss: idle for an hour, then down for longer than
    `TEAMS_INITIAL_LOOKBACK_MINUTES` (5) — a crash or host reboot, not a deploy,
    since compose builds before it swaps — and messages from the early part of the
    downtime land below the clamped watermark and are marked seen without being
    answered, where a busy bot with the same outage loses none. Fix shape: persist
    `saved_at` on every save and clamp on `now − saved_at`, which is true downtime
    even after a crash because the save runs every poll; keep the watermark-age
    test only for state files that predate the field. Watermark code — mutation
    tests and a `tests/load/` run, not a quick patch.
12. **Eval runs do not record the model digest.** (Deploy audit.) Production's
    model is `qwen3.6:latest`, the only tag on a host this project does not own,
    and no measurement here recorded what it resolved to — so whether the
    2026-10-06 probe, the 571-token prompt budget or the crash matrix ran on
    `07d35212591f` can no longer be known. `rag/model_digest.py` now exists for the
    startup banner; one field in `eval/run_experiment.py`'s metadata would make
    every future measurement say which weights it measured.
    **Done 2026-10-08** (`feat/run-identity`), widened: `rag/run_identity.py`
    gives eval metadata and every production `compliance_request` span the same
    identity — `git_commit`, the LLM digest, `system_prompt_sha12`, and the new
    `agent_input_sha12`, a hash of the question+`[Source N]` layout the model
    reads, which the prompt hash could not see and which #10's fix will likely
    change. Eval warns when the digest is not `LLM_MODEL_DIGEST`. Past runs stay
    unrecoverable.
13. **qdrant-client 1.19.1 against server 1.17.1.** (Deploy audit.) Outside
    Qdrant's supported window of one minor version, and in the VM logs since at
    least 2026-09-21. Nobody chose it: `requirements-bot.txt` says
    `qdrant-client>=1.17`, so the image got whatever was newest when Docker last
    rebuilt its dependency layer, while the local suite runs 1.17.0. Pinning the
    image's dependencies freezes 1.19.1 — the version proven in production — and
    turns staying outside the window, or moving to `~=1.17.0` to match the server
    and the tests, into a deliberate choice. The server is on the shared host, so
    upgrading it is not ours to do.
14. **Phoenix exports every span synchronously** — `SimpleSpanProcessor`, as its
    own startup warning says. (Deploy audit.) For answers that happens on the one
    worker thread, so a Phoenix container that hangs rather than refuses would
    stall replies. Never observed. `register(..., batch=True)` moves export to a
    background thread; arize-phoenix-otel 0.15.0 supports it.
15. **`[worker] Reply sent` does not say what was sent.** (Deploy audit.) It
    prints identically for an answer and an escalation, so a container log cannot
    say how a question went — the audit could confirm two replies, not two
    answers. Phoenix's `compliance_request.outcome` has it; the log line could
    carry it.
