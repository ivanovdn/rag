# Compliance Bot Scaling Audit — Can the bot carry 30 users?

**Date:** 16 September 2026 · **Type:** infrastructure audit · **Implemented by:** `docs/superpowers/plans/2026-09-16-scaling-to-30-users.md`
**Canonical:** <https://claude.ai/code/artifact/97477412-f0f1-4fe8-986a-21b2564eac20> (private artifact — this file is the shareable copy, plus an addendum from the plan audit)

A measured audit of the Teams compliance bot — the remote inference stack, the real latency budget, and what actually breaks between 5 and 30 people.

Provenance: **Measured** = taken against the live stack on 15–16 Sep 2026 with the project's own pipeline and test cases. **Estimated** = derived from those measurements. **Unverified** = still needs checking. Latency measured across a 285 ms link has that overhead subtracted where a co-located figure is quoted.

## Verdict — yes, and the GPU was never the problem

One bot process completes roughly **210 questions per hour**. Thirty people asking three questions a day puts the system at about **5% utilisation**. Capacity is not close to a limit, and per-question latency barely moves between 5 users and 30.

What breaks is narrower: a **burst** — everyone reacting to the same policy announcement — drains strictly one at a time, and the people waiting see no acknowledgement at all until their turn comes. That is a user-experience cliff, not a throughput cliff, and the fix is small.

Three constraints:

1. **Head-of-line blocking.** The poll loop stops polling while it answers, so a queued user gets silence, not a "searching…" notice.
2. **Graph polling grows with users.** 31 API calls every cycle at 30 chats, for a signal that is almost always empty.
3. **Shared-box contention.** The inference host belongs to several projects, and nothing reserves capacity for this one.

## 1 — The remote stack (Measured)

| Host | Service | Model | Notes |
|---|---|---|---|
| .22 | Ollama 0.21.2 | qwen3.6:latest | 36B **MoE**, Q4_K_M, 23.9 GB — the main agent LLM and the router |
| .22 | Ollama 0.21.2 | embeddinggemma | 308M, BF16, 0.6 GB — 768-dim query embeddings |
| .22:8267 | vLLM 0.17.1 | Qwen3-Reranker-4B | `max_model_len 4096`, `gpu_mem_util 0.15`, prefix caching on |
| .22:6333 | Qdrant | compliance_policies | 1,602 points · 768-dim cosine · on-disk payload · status green |
| .23:8269 | vLLM 0.24.0 | qwen3-14b | Dense 14B, `max_model_len 40960` — evaluated as a router, **rejected** |
| .22:8081 | speaches-ai | Kokoro-82M TTS | **Not this project.** Evidence the host is shared |

Also resident but unused here: `mistral-large` (73 GB), `gpt-oss:120b` (65 GB), `qwen3-next:80b` (85 GB), `qwen2.5:72b` (77 GB) — on 128 GB of unified memory.

**Contention is not theoretical.** A cold model load was measured twice: **3.77 s** on the first probe and **51.7 s** a day later — same model, same host, a 13× spread. The cause wasn't isolated, but the variance is the point. With the default 5-minute `keep_alive` and sporadic use, roughly half of all questions pay *some* reload, and the size of that penalty is not under this project's control.

## 2 — The latency budget: where the 16 seconds go (Measured)

Four genuine questions from `chatbot_test_cases.json` through the real pipeline (router, agent, retrieval, reranker). End-to-end **14.6–21.0 s** (mean 18.0 s) from a laptop across a 285 ms link; subtracting that link's six round trips puts a co-located bot at roughly **16 s**.

| Component | Time | Share | Detail |
|---|---|---|---|
| Decode | 12.5 s | 78% | ~750 tokens at a flat **60 tok/s** |
| Prefill | 2.1 s | 13% | ~4,400 tokens at 1,411–1,809 tok/s |
| Retrieval | 0.85 s | 5% | embed ~85 ms, Qdrant ~160 ms, rerank 25 candidates ~245 ms |
| Router | 0.65 s | 4% | one temperature-0 classification |

Answers measured 254–614 tokens. **Retrieval is not the bottleneck and never was.** The only component with real headroom is answer length, which the system prompt sets deliberately (a verbatim quote from *every* relevant source); shortening trades directly against citation completeness, which is the thing the bot exists to do.

## 3 — Concurrency: Ollama serialises, vLLM batches (Measured)

Identical requests fired simultaneously, with distinct prompts so prefix caching could not mask the result; 300-token generations on `qwen3.6:latest`.

| Concurrent | Ollama total wall | Per-request completion | Aggregate throughput | vLLM reranker total wall |
|---|---|---|---|---|
| 1 | 4.80 s (1.00×) | 4.8 | 30.6 tok/s | 0.56 s (1.00×) |
| 3 | 12.53 s (2.61×) | 4.6 · 8.2 · **12.5** | 33.0 tok/s | 0.88 s (1.57×) |
| 6 | 25.48 s (5.31×) | 4.7 · 8.8 · 13.0 · 17.1 · 20.7 · **25.5** | 33.9 tok/s | 0.96 s (1.71×) |

Perfect serialisation tracks 1× / 3× / 6×; perfect batching stays flat. Ollama sits almost exactly on the serial line — that is `OLLAMA_NUM_PARALLEL=1` — and decode held at exactly 60 tok/s in every row, so extra load buys nothing. The reranker on the same physical host absorbs six concurrent requests for a 1.7× cost. `OLLAMA_NUM_PARALLEL=4` would unlock real batching, but it lives on a host this project doesn't own, and it isn't needed yet.

## 4 — Capacity and bursts: the mean is fine, the tail is not (Derived)

Wait for the *last* person in the burst, from a 16 s service time:

| Scenario | Today (serial) | With a worker queue |
|---|---|---|
| Steady state — 90 questions/day | < 1 s queue wait | < 1 s |
| Burst of 5 | 80 s | 80 s — but acknowledged at 2 s |
| Burst of 10 | **2.7 min** | 2.7 min — acknowledged at 2 s |
| Burst of 20 | **5.3 min** | 5.3 min — acknowledged at 2 s |

The queue does not make the work faster — the GPU still serialises. It turns silence into an immediate acknowledgement, which is the difference between "busy" and "broken". At 5 users a burst is at most 5 deep; at 30 it can be 15–20 deep, and today the loading indicator is sent *after* the queue wait because the thread that would send it is busy answering someone else. That inversion, not the raw latency, is what will generate complaints.

## 5 — A landmine in the obvious fix: concurrency isn't safe here yet

Two module-level globals act as a side channel, because `AgentWorkflow` swallows tool exceptions and the bot cannot see a retrieval failure any other way:

```text
search_policies.py:6   _retrieval_unavailable = False    # process-wide
bot.py:55              sp._retrieval_unavailable = False # (C) reset before the run
search_policies.py:62  _retrieval_unavailable = True     # (B) set on transient failure
bot.py:76              if sp._retrieval_unavailable:     # (D) read after the run
```

A reset-then-read pair with **16 seconds of network I/O in between** — safe today only because the loop is strictly serial. Add a second request:

| t | Alice — Qdrant blip | Bob — arrives 5 s later | Flag |
|---|---|---|---|
| 0 s | (C) reset | — | False |
| 3 s | **(B) retries exhausted → set** | — | True ✓ |
| 4 s | agent writes a 600-token escalation | — | True |
| 5 s | *still decoding…* | **(C) reset** | **False ✗** |
| 14 s | **(D) reads False → falls through** | — | False |

Alice receives a **content escalation** for a transient infrastructure blip — precisely the invariant the resilience layer exists to uphold. The mirror case is worse: Bob's correct, fully-cited answer is discarded and replaced with "service unavailable".

**`asyncio` alone is enough to trigger this — threads are not required.** `agent.run()` awaits on every network hop, so two coroutines on one event loop interleave at each `await`. "Just make it async so polling doesn't stall" *is* the trap. Not everything is affected: `_pending_ratings` is keyed by chat and safe; `_embedding_model` is an idempotent singleton. The hazard is the two reset-then-read globals, plus `_last_search_results`, which would silently mislabel which chunks produced which answer in evaluation runs.

**The way out — and why it can wait.** A queue with **exactly one worker thread** preserves the "one pipeline in flight" invariant, so the globals never race. That fixes the acknowledgement problem *without* arming the landmine. Raising the worker count above one is the explicit trigger for fixing the globals properly — cleanly gated, and not needed until `NUM_PARALLEL` is raised on the shared host. When that day comes, the clean fix is available: `ToolCallResult` exposes `tool_output`, and `search_policies` already returns its sentinel in-band, so reading tool results from the per-request handler removes the side channel entirely.

## 6 — Microsoft Graph: polling cost that scales with people

Every cycle issues one call for the chat list plus one per chat, with no `$top` and no `$filter` — each returning a full 20-message page of HTML bodies. The useful signal is almost always empty.

| Users | Calls per cycle | Real cycle time | Mean detection delay |
|---|---|---|---|
| 5 | 6 | ~6.5 s | ~3.3 s |
| 30 | 31 | **~12.8 s** | ~6.4 s |

*(Estimated — assumes ~250 ms per Graph call, **not measured**.)* The configured 5-second poll interval effectively becomes ~13 s at 30 chats, and daily volume approaches 210,000 GETs against a delegated user token.

What can be done — all of it in this project's control:

- **Collapse the N+1.** `GET /me/chats?$expand=lastMessagePreview` should reveal which chats changed in a *single* call, with per-chat fetches only for those that did. *(Unverified at audit time — see the addendum.)*
- **Back off when idle.** Polling every 5 s around the clock for a business-hours tool; a schedule-aware interval removes roughly 70% of the volume at no cost to users.
- **Add `$top`.** Stop pulling 20 full messages per chat per cycle to find at most one new one.

**A risk that outranks throughput.** The bot authenticates as a **delegated user account**, not an application. That account's token, MFA posture and conditional-access policy are now production dependencies — if its sessions are revoked or policy tightens, the bot stops for everyone. At 5 users that is an annoyance; at 30 it is an outage with an audience. A proper Teams app registration removes polling and this dependency together, but needs an endpoint Microsoft can reach — which is likely why the current design exists.

## 7 — Recommended work, ordered by value over effort

| Lever | Owner | Gain | Notes |
|---|---|---|---|
| Extend `keep_alive` on the Ollama client | Yours | −3.8 s on ~half of questions | A client-side field, default `'5m'` — no host configuration and no owner permission needed. Prefer ~30 m over infinite: parking 24 GB on a shared box is a social cost, not a technical one |
| Queue + one worker thread | Yours | silence → 2 s | Poll thread detects and acknowledges immediately; worker runs the pipeline one at a time. Roughly 25 lines, no new infrastructure; the globals stay safe. **Decide deliberately how `last_check` advances** — today a restart with queued work would lose it silently |
| Close the `OpenAILike` thinking gap | Yours | prevents a 50× regression | The Ollama path sets `thinking=False`; the `openai-compatible` branch has no equivalent. Flipping `LLM_BACKEND` with any Qwen3 model silently restores reasoning on the router *and* every agent turn — measured 294 reasoning tokens and 34.5 s for one classification. It still parses, so it degrades quietly |
| Graph: `$expand`, `$top`, adaptive interval | Yours | ~31 calls → 1 | Takes request volume from ~210k/day to low thousands and stops it scaling with headcount |
| Profile the client-construction gap | Yours | est. −1.2 s | Raw HTTP to the router is 650 ms, but in-pipeline it measured 1.8 s. `classify_message` and `build_agent()` construct a fresh LlamaIndex client on *every* call; ~1.2 s unexplained, inside this project's process. *(Corrected by the addendum: measure, but do not cache.)* |
| `OLLAMA_NUM_PARALLEL=4` | Host owner | deferred | The only change that would raise real throughput — and the trigger that would require fixing the globals first. Not needed at this scale |

## 8 — Tested and rejected (recorded so they aren't re-litigated)

- **Qwen3-14B on `.23` as the router — 3.4× slower.** 2,184 ms versus **650 ms** for the model already in use. The 36B wins because it is **MoE** — few active parameters per token — while the 14B is dense on weaker silicon: 8.5 tok/s against 75. Accuracy was identical (12/12, plus 10/10 on adversarial cases). Also unusable for the main agent: 250–614-token answers would take 30–70 s.
- **A BERT classifier container — optimises 4%.** The router is 4% of end-to-end latency, so the ceiling is 16.0 s → 15.4 s. It would also cost a container on a host this project doesn't own, labelled training data, and a retraining loop. A nearest-centroid classifier over the *existing* embedding model reaches ~115 ms with no new infrastructure and scored the same 12/12 and 10/10 — but cosine similarity is not a calibrated probability, so `ROUTER_CONFIDENCE_FLOOR` would need redesigning around top-1/top-2 margin. A real change to safety-critical logic, for 0.5 s.
- **Celery — wrong tool.** Celery solves multi-host workers, durable retries and scheduled jobs — none of which apply to a single VM handling ~90 questions a day. Decisively: **it would not add one question per hour**, because every worker still serialises at a host configured for `NUM_PARALLEL=1`. It would add a broker, a container and two failure modes to feed a single-threaded bottleneck. A `queue.Queue` and one thread does the job.
- **Turning off thinking in the router — already done.** `rag/agent.py:176` sets `thinking=False`, and the raw probe confirms it: 23 clean tokens, no reasoning trace. The real exposure is the `OpenAILike` branch, above.

## 9 — Open questions

- **Does `lastMessagePreview` carry a complete message body?** Verifying means using production Teams credentials — and Azure AD refresh tokens rotate on use, so a careless test could invalidate the live bot's token and take it offline. Needs a deliberate, supervised check. *(Partly settled in the addendum: body completeness turns out not to matter.)*
- **Is 16 s acceptable to these 30 people?** It barely changes with user count, so if it is too slow today, scaling is the wrong thing to work on. That is a product judgement.
- **What is the real burst shape?** Whether questions actually cluster after policy announcements decides whether any of the concurrency work is ever needed.
- **Who else depends on the inference host?** If another project loads an 80 GB model beside this one, evictions follow — invisible at 5 users, a reliability problem at 30.
- **Graph call latency is assumed, not measured** at ~250 ms. Every polling figure in section 6 moves with it.

One free way to de-risk a router change: `record_classification` already logs the category, the confidence and **the full message** to Phoenix for audit. Real traffic can therefore be replayed through any candidate router offline, measuring the only metric that matters — *did it ever reject a genuine question that the current router accepted?*

---

## Addendum — plan audit, 16 September 2026

Found while auditing the implementation plan against the code, the installed packages and the Microsoft Graph documentation. Both corrections are folded into the plan (Tasks 4–6).

1. **Graph pages `GET /me/chats`, and the bot reads only page one.** Documented: 20 chats per page by default, `$top` max 50, and "if the result set for all chats spans multiple pages, the response object includes an `@odata.nextLink` property … continue making additional requests with the `@odata.nextLink` URL". `process_new_messages` never follows it, so past ~20 chats some users are simply never polled — invisible at 5 users, a dropped-user bug at 30. Also documented as supported: `$expand=lastMessagePreview` and `$orderby=lastMessagePreview/createdDateTime desc`; the `chatMessageInfo` preview carries `id`, `createdDateTime`, `from`, `body`, `isDeleted`, `messageType` (`from` is null for `systemEventMessage`). Body completeness is irrelevant to the change-detection skip, which is keyed on the preview's `id`.

2. **Caching the LLM client is unsafe with the per-request event loop.** `_run_rag` runs each request under `asyncio.run()`; llama-index's `Ollama` (llama-index-llms-ollama 0.9.1) creates its `httpx.AsyncClient` once and reuses it for the object's lifetime. Reproduced with httpx 0.28.1 / httpcore 1.0.9 — one client, three successive `asyncio.run()` loops: run 0 OK, run 1 `RuntimeError: Event loop is closed`, run 2 OK. `RuntimeError` is not in `_TRANSIENT_TYPES`, so in production it would surface as a false content escalation carrying a raw error, on roughly every other question. The "profile the client-construction gap" lever therefore *measures* but does not cache. Construction of a pydantic client is expected to be milliseconds; the likelier cause of the router's extra ~1.2 s is server-side prompt-cache eviction as router and agent prompts alternate through Ollama's single slot (untested — costs GPU time).

Also verified sound at audit time: `Ollama.keep_alive` exists in llama-index-llms-ollama 0.9.1 (default `'5m'`); `OpenAILike` forwards `additional_kwargs` into `chat.completions.create(**kwargs)` and openai 2.31.0 accepts `extra_body`; the unit suite on `main` was green (101 passed).
