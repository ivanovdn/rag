# Search-first retrieval, prompt/tool audit, and the grounding backstop

**Date:** 2026-09-24
**Status:** Draft (design), pending review
**Author:** Dmytro Ivanov (with Claude Code)
**Branch:** `feat/search-first`

Retrieval stops being something the agent *decides* to do and becomes something that
has *already happened* when the agent is invoked. The agent becomes tool-free, the
system prompt loses 46% of its tokens, and no answer can reach a user without a
citation.

---

## Problem

Three defects, one root: the agent is trusted with decisions that code should make.

**1. The agent can skip retrieval.** `search_policies` is a tool the model chooses to
call. Measured in production: **21 invocations across 25 agent runs**. A compliance
answer produced without retrieval violates the project's first constraint — *"Agent must
never answer without citing a retrieved chunk"* — and nothing downstream catches it.

**2. 46% of the context window is spent before any policy text.** Measured against a
live `qwen3.6` 36B at `num_ctx` 4096:

| | tokens |
|---|---:|
| System prompt | 1,066 (4,467 chars) |
| 3 tool schemas | 817 |
| **Fixed overhead** | **1,883** |
| Left for sources + question + answer | 2,213 |

At the largest prompt observed in Phoenix (3,140 tokens), sources account for ~1,257 and
only **956 tokens remain for the answer while `num_predict` is 1024** — the cap is
nominal, not payable. `num_ctx` cannot be raised: 4096 is the only value the MoE+CUDA
crash matrix proved safe (`2026-09-17-ollama-moe-cuda-crash.md`, §9).

**3. An answer with no citation renders as a normal answer.** `channels/teams/bot.py:743`
branches `escalation.needed` → `elif result.get("answer")` → else error. There is no
check for citations, and `parse_success` is read only by `eval/` and tests — never by
`bot.py`. `channels/teams/renderer.py:79-80` then renders a citation-less answer as bare
prose. This directly violates the grounding constraint.

**4. The relevance floor is dead in production.** `rag/tools/search_policies.py:81`
applies `min_confidence_score` **only when the reranker is off**, and production runs
`RERANKER_ENABLED=true`:

```python
if not settings.reranker_enabled and raw[0].score < settings.min_confidence_score:
    return "NO_RELEVANT_POLICY_FOUND"
```

So the sentinel requires Qdrant to return literally zero rows — which will not happen
against a populated collection. Retrieval always returns `reranker_top_n` sources however
irrelevant they are (the uncovered-question probe got six "No Retaliation" clauses and no
sentinel), and the only thing standing between that and a user is the model's judgment.

Meanwhile `rerank_score` — documented at `rag/reranker.py:122` as a **0.0-1.0 relevance
probability**, captured into `_last_search_results`, and already emitted to Phoenix as
`reranker.top_score` (`rag/reranker.py:145`) — is computed for every question and used for
nothing.

## Goal

- Retrieval is unconditional and happens in code, not by model choice.
- Fixed per-request overhead drops from 1,883 to 572 tokens.
- No answer reaches a user without a citation and a successful parse.
- "Retrieval found nothing relevant" becomes a number decided in code, not a judgement
  the model is trusted to make.

---

## Evidence (measured, not assumed)

All probes ran against a local `qwen3.6:latest` (36B) — same family and size class as
production `qwen3.6:35b`, so token counts and behaviour transfer; **latencies do not**
(local Qdrant, reranker down). Scripts are throwaway, in the session scratchpad.

### The two unused tools are unused for different reasons

`get_section` and `escalate_to_compliance` have **never been invoked** in production.

| probe | setup | tool calls | result |
|---|---|---:|---|
| 1 | prod prompt, no matching policy | 1 | `search_policies` → escalated **via the JSON field** |
| 2 | prod prompt, covered question | 1 | `search_policies` → 3 citations |
| 3 | prod prompt **+ explicit order** to call `get_section` | 1 | order ignored |
| 3b | same order moved to the **end** of the prompt | 1 | ignored |
| 3c | same order **+ gating condition stripped** from its docstring | 1 | ignored |
| 4 | escalation case, **JSON escape hatch removed** | **2** | `search_policies` → **`escalate_to_compliance`** |

- **The model is capable.** Probe 4 chains two tools. The framework is not the cause
  either: `from_tools_or_functions` builds a **`FunctionAgent`** (native Ollama tool
  calls), not a `ReActAgent` — llama-index picks on `is_function_calling_model`, which
  `Ollama` reports `True`. `rag/agent.py:208`'s docstring claiming "ReAct agent" is wrong.
- **`escalate_to_compliance` loses to the JSON field.** The model must emit
  `escalation.needed` regardless, so the tool call is a wasted round-trip. It skips it —
  correctly.
- **`get_section` loses to satisfaction.** Once `search_policies` returns usable sources
  the model stops. Three controlled variations failed to force a second retrieval step.

### The escalate tool, when it does fire, corrupts the output

Probe 4's full chain: tool returns prose (`"ESCALATED: ... (Ticket #ESC-2026-0001). They
will respond within 2 business days."`) → model emits prose instead of JSON →
`parse_success: False` → `rag/response.py:33` fallback sets `escalation.needed = False`
→ `bot.py:743` falls to `elif result.get("answer")` → span records **`outcome="answered"`**
→ `renderer.py:79` renders the prose bare.

A question the model **decided to escalate** is delivered as an answer, carrying an
invented ticket ID and an SLA nothing backs, and logged as answered. `_escalations` is an
in-memory dict with zero consumers; no human ever sees it.

### Search-first works, tool-free

| case | tools available | tool calls | outcome |
|---|---|---:|---|
| covered question | all 3 | 0 | answered, 2 citations |
| uncovered question | all 3 | 0 | escalated, correct specific reason |
| covered question | **none** | 0 | answered, 2 citations, `parse_success: True` |

In the uncovered case the model had off-topic sources, a prompt explicitly offering
re-search, and the tool available — and escalated instead of re-searching. This is the
scenario re-search exists for.

### Token budget after the audit

| | today | after |
|---|---:|---:|
| System prompt | 1,066 | **572** (4,467 → 2,327 chars) |
| Tool schemas | 817 (3 tools) | **0** |
| **Fixed overhead** | **1,883** | **572** |
| Left for sources + question + answer | 2,213 | **3,524** |

**1,311 tokens/request reclaimed — 32% of the whole window.**

---

## Decisions (locked)

**D1 — Retrieval runs before the agent, in code.** The user's question is passed to
`search_policies` verbatim. No rewriting, no keyword extraction.

**D2 — The agent is tool-free.** All three tools are removed from the agent.
*This revises the approved design, which kept `search_policies` for re-search.* The
evidence above shows the model will not re-search even when told it may. Reversible in
one line if eval disagrees; the eval gate (below) is where that is decided.

**D3 — The system prompt is rewritten**, 1,066 → 572 tokens. Every behavioural rule is
preserved. Only duplication is cut: "cite all relevant sources" appeared 3×, the
citation-copying rule 2×, the pointer-not-advisor stance 4×; two CORRECT examples collapse
to one (the JSON block already demonstrates multi-citation) and two WRONG examples to one
(both showed the same failure). Rules 0 and 1 become a SOURCES section.

**D4 — Escalation is decided in three layers** (see below). The JSON `escalation.needed`
field is the only escalation signal, as it already is in production.

**D5 — `rag/tools/escalate.py` and `rag/tools/get_section.py` are deleted**, with their
exports. Durable escalation recording (parked option C) hooks into `bot.py`'s escalation
branch, not a tool.

**D6 — `num_ctx` stays 4096 and is not revisited by this work.** Going tool-free removes
tool-calling, one of the five MoE+CUDA crash conditions, which makes a higher `num_ctx`
*hypothetically* safe. That hypothesis is recorded and **not acted on**: a previous
attempt to raise it passed local probing and crashed in production
(`2026-09-17-ollama-moe-cuda-crash.md` §9). Any future attempt goes through that spec's
matrix, not this one.

**D7 — `num_predict` stays 1024.** The reclaimed budget makes it *payable* rather than
nominal: at the largest observed source payload the answer budget rises from 956 to
2,267. Observed max completion is 617 tokens, so nothing is truncating and raising the cap
would be speculative. 2048 would also fit (3,896 < 4,096) if truncation is ever observed.

**D8 — The relevance floor is wired to `rerank_score`.** After reranking,
`results[0]["rerank_score"] < settings.reranker_min_score` returns
`NO_RELEVANT_POLICY_FOUND`. This is what makes escalation layer 1 real; without it that
layer almost never fires.

A **new** setting, not a reuse of `min_confidence_score`: cosine similarity and a reranker
relevance probability are different scales, and one knob for both would be a latent bug.
`min_confidence_score` keeps its existing reranker-off meaning.

**The default ships as `0.0` (floor off), and choosing the real value is a task in the
plan, not a guess in this spec.** `reranker.top_score` is already in Phoenix, so the
threshold is read off the VM's production traces — split by whether the model escalated —
before the branch merges. Guessing is specifically unsafe here: CLAUDE.md already records
a "reranker scores compressed" failure mode, so the score distribution in this deployment
must be looked at, not assumed.

---

## Components

### `rag/search_first.py` (new)

One purpose: get sources before the agent runs, and say what happened.

```
prefetch(question) -> PrefetchResult
    status: "ok" | "unavailable" | "no_match"
    sources: str   # formatted [Source N] blocks, empty unless ok

compose_agent_input(question, sources) -> str
```

`prefetch` calls `search_policies(question)` and maps its return: the
`POLICY_SEARCH_UNAVAILABLE` sentinel (or `sp._retrieval_unavailable`) → `unavailable`;
`NO_RELEVANT_POLICY_FOUND` → `no_match`; anything else → `ok`.

Shared by `bot.py` and `eval/agent_wrapper.py` so eval cannot drift from production —
today it builds its own three-tool agent while importing production's `SYSTEM_PROMPT`.

### `rag/tools/search_policies.py` — relevance floor

After Step 2 (rerank), before Step 3:

```
if reranker_enabled and results and reranker_min_score > 0:
    if results[0]["rerank_score"] < settings.reranker_min_score:
        -> return NO_RELEVANT_POLICY_FOUND
```

Deliberately different from the other sentinel paths: `_last_search_results` is **kept
populated** on the floor path rather than cleared. Retrieval did return candidates, and
both the tier-1 retrieval evaluators and Phoenix diagnostics need to see what was found
and how it scored in order to tune the threshold. Record a span attribute
(`retrieval.floor_rejected`, with the top score) so rejections are countable in production.
Do not "fix" this into matching the other paths.

### `rag/agent.py`

- `SYSTEM_PROMPT` replaced (D3).
- `ALL_TOOLS = []`; tool imports removed.
- `build_agent()` docstring corrected — it builds a `FunctionAgent`, not a ReAct agent.
- `Citation` / `Escalation` / `ComplianceAnswer` are never instantiated and never sent to
  the model; the real contract is the JSON block inside the prompt, parsed by
  `dict.get`. Add a comment naming the prompt as authoritative, so the two copies cannot
  silently drift.

### `channels/teams/bot.py` — `_run_rag`

```
prefetch(question)
  unavailable -> {"status": "unavailable"}
  no_match    -> {"answer": "", "citations": [],
                  "escalation": {"needed": True, "reason": "no relevant policy found"}}
  ok          -> build agent, run with compose_agent_input(...), parse
```

Both non-`ok` branches return **without building an agent or calling the LLM**.

The post-run `sp._retrieval_unavailable` check is **removed**, not kept. It exists because
`AgentWorkflow` swallows tool exceptions, so a failure inside `search_policies` was only
visible via that flag after the run. With retrieval hoisted out of the agent and no tools
left, the only caller is `prefetch`, which sees the failure directly — the post-run check
is unreachable. `prefetch` takes over its logging duty: it must keep printing the
`[worker] Unavailable (retrieval): ...` line, which is the only container-log record that
retrieval, not the LLM, was the failing component.

### `channels/teams/bot.py` — grounding backstop (new)

After parsing, before rendering:

- `parse_success` is False → **not an answer**. Escalate with the parse failure as the
  reason. `bot.py` currently ignores this field entirely.
- `escalation.needed` is False **and** `citations` is empty → **not an answer**. Escalate.

Both record their own `compliance_request.outcome` value so the mislabelling in probe 4
is visible in Phoenix rather than hidden as `answered`.

### `channels/teams/renderer.py`

Delete the citation-less fallback at lines 79-80 (`if not citations: return f"<p>{answer}</p>"`).
It violates the grounding constraint, and after the backstop nothing can reach it.

### `eval/`

- `agent_wrapper.py`: use `prefetch` + `compose_agent_input`; tool-free agent; log the
  pre-search so retrieval is still measured.
- `evaluators.py`: delete `agent_used_get_section` (it has scored 0.5/"skipped" on every
  run ever). `agent_search_count` must count the pre-search, not tool calls, or it reads 0
  for every case.
- `run_experiment.py:146-147`: drop `section_calls`; keep escalation counting from the
  JSON field.

### `rag/tools/__init__.py`

Holds a **second, dead `ALL_TOOLS`** — `rag/agent.py` defines its own and never imports
this one. Nothing imports it. Remove the two deleted tools; the duplicate list goes with
them.

### `config.py` + `.env.example`

**New:** `reranker_min_score: float = 0.0` (D8), with a comment stating that 0.0 means
off and that the live value is measured from `reranker.top_score`, not guessed. Added to
`.env.example`.

`agent_max_iterations` and `agent_timeout` are consumed by nothing. `FunctionAgent` has no
iteration knob, so `agent_max_iterations` cannot be wired as written — **delete it**.
`agent_timeout` **is** wirable (`from_tools_or_functions(timeout=)`) and today nothing
bounds an agent run except the per-call `Ollama.request_timeout` (300s remote), under a
12s shutdown drain — **wire it**.

`escalation_ticket_prefix` has exactly one consumer, the deleted `escalate.py` — delete it
too. The `smtp_*` / `compliance_team_email` settings belong to unimplemented email
escalation and stay.

`.env.example` drops the removed keys. The deployed `.env` and the untracked
`.env.remote-backup` are **not edited** — unknown keys are inert to pydantic-settings, and
a restore artefact is not ours to rewrite.

### `CLAUDE.md`

The architecture block names three agent tools and describes the search flow as
agent-driven; the Message-flow and Search-flow paragraphs both need updating, plus a
gotchas row for the FunctionAgent/ReActAgent distinction and the satisfaction behaviour
(the model will not chain a second retrieval step).

---

## Escalation decision — three layers

1. **Deterministic (code, no LLM).** `prefetch` returns `no_match` → escalate. Today
   CLAUDE.md requires this (*"If `search_policies` returns `NO_RELEVANT_POLICY_FOUND` →
   escalate"*) but it is a prompt instruction the model may ignore. It becomes a guarantee.
   **This layer is only meaningful once D8's floor is on** — with the floor at 0.0 the
   sentinel fires about as rarely as it does today.
2. **Model judgment.** Sources returned but none answer the question → the model sets
   `escalation.needed = true` with a reason. Unchanged from production, and demonstrated
   working tool-free.
3. **Backstop (code).** Failed parse, or no citations with `needed: false` → escalate
   regardless of what the model claimed.

## Error handling

- `unavailable` short-circuits before any LLM call — faster, and it cannot be
  misread as a content escalation.
- Non-transient errors continue to propagate to escalation as today.
- The unavailable reply still gets no rating prompt and creates no feedback row.

## Testing

**Unit (offline, no LLM):**
- `prefetch` maps each sentinel to the right status.
- `unavailable` and `no_match` short-circuit **without building an agent**.
- The question reaches `search_policies` verbatim — no rewriting.
- `compose_agent_input` puts the sources in the agent input.
- Backstop: `parse_success: False` → escalated, not answered.
- Backstop: `needed: false` + `citations: []` → escalated, not answered.
- `ALL_TOOLS` is empty (guards D2 against a silent re-add).
- Token budget, offline-checkable: assert
  `num_predict + FIXED_OVERHEAD_TOKENS + MAX_SOURCE_TOKENS <= num_ctx`, where
  `FIXED_OVERHEAD_TOKENS` is a constant carrying the measured value (572) and is pinned to
  the prompt by a second assertion on `len(SYSTEM_PROMPT)`, so editing the prompt without
  re-measuring fails the suite rather than rotting the constant. (An equivalent guard was
  written during the `num_ctx` episode and lost in the revert.)
- `renderer.render_answer` is never reached with empty citations.

- Floor: a top `rerank_score` below the threshold returns the sentinel; at or above it
  returns sources; `reranker_min_score = 0.0` disables the check entirely; the floor path
  leaves `_last_search_results` populated (mutation-test this — clearing it is the obvious
  wrong "cleanup").

**Live eval — the merge gate. Runs on the VM, not locally**, against production infra
(remote Ollama, remote Qdrant, vLLM reranker **on**). Local runs cannot substitute: every
probe behind this spec ran with the reranker down, so no local `rerank_score` exists and
the floor cannot be tuned here. Dataset upload and the eval runs happen on the VM's
Phoenix.

`e2e-test-v1` and `chatbot-test-v1`, old prompt vs new,
pass/fail on **citation accuracy** (`citation_doc_accuracy`, `citation_section_accuracy`,
`citation_clause_accuracy`). Token savings alone do not justify a merge.

Known signal to resolve there: on one covered question the new prompt produced **2
citations where the production prompt produced 3** — reproduced 3× (draft, draft+tools,
draft with a cite-all repetition restored), so it is not noise and not caused by the cut
repetition. Whether 2 or 3 is *correct* needs the labelled dataset.

## What is explicitly unchanged

`num_ctx` (4096), `num_predict` (1024), `temperature` (0.0), the router, the resilience
and retry layer, the one-worker queue, the watermark logic, `rag/response.py`'s parser,
the rating flow, and Phoenix span structure.

## Out of scope (YAGNI)

- **Replacing `AgentWorkflow` with a direct `llm.achat`.** With zero tools the workflow is
  machinery around one LLM call, but the instrumentation, retry path and response handling
  all key off `agent.run()`. Revisit once eval confirms tool-free.
- **Rephrase-and-retry on a rejected search.** With D8 the pipeline can finally *tell*
  that retrieval failed, which makes "rewrite the query and search again" implementable —
  in code, off the floor signal, with no reliance on the model choosing to re-search.
  Deliberately not built here, pending two measurements:
  (a) on `retrieval-test-v1`, do the cases that miss actually *hit* after rephrasing? If
  the hit rate is already high this buys little;
  (b) what share of production questions does the floor reject at the chosen threshold?
  It is also the dangerous direction — a second search that surfaces something
  tangentially related converts a *correct* escalation into a weakly-grounded answer, and
  for a compliance bot escalating is the safe failure. It doubles latency on the worst
  path. Build it only against evidence, with the retry bounded to exactly one.
- **Durable escalation records** (option C) — parked; this spec only fixes where they
  would hook in.
- **Raising `num_ctx`** — see D6.
- **`render_escalation` HTML escaping** — pre-existing, tracked separately.

## Risks

| Risk | Mitigation |
|---|---|
| Shorter prompt costs citation accuracy | The eval gate; 2-vs-3 signal already flagged |
| Tool-free removes any recovery from bad retrieval | Escalation layers 1-3; eval decides |
| Deleting tools breaks comparability with past eval runs | Expected; new baseline recorded at merge |
| Backstop turns marginal answers into escalations | Escalation is the safe direction for a compliance bot; rate is measurable in Phoenix |
| Floor set too high → answerable questions escalate | Ships at 0.0 (off); threshold chosen from VM Phoenix data, and `retrieval.floor_rejected` makes the rate visible before and after |
| Floor set too low → no effect, layer 1 stays dead | Same measurement; an ineffective floor is visible as a zero rejection rate |
