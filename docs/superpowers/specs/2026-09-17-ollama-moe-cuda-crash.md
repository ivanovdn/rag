# Ollama MoE CUDA Crash — Diagnosed, Not Fixed

**Date:** 17 September 2026 (incident) · 18 September 2026 (written up; revised same day with a direct `num_ctx` measurement and a production A/B trace) · **Type:** incident diagnosis — documented, not fixed
**Upstream:** ollama/ollama#17434 — <https://github.com/ollama/ollama/issues/17434> (open; filed by us)

A production 500 that took down every tool-calling Ollama request on the shared Spark host, traced to a five-condition MoE + CUDA interaction inside Ollama itself — not a bug in this bot. Recorded here because the one confirmed fix (§6) is server-side, on a host this project doesn't own; the one option that was ours — lowering `num_ctx` client-side — has been taken, and its cost is recorded in §6.

Provenance: **Reported** = from the upstream issue thread, reproduced by the reporter on their own box (`qwen3.6:35b`, Ollama 0.32.5) — not independently re-run by us. **Confirmed** = holds against our own deployment. **Ours** = specific to this deployment, not covered by the upstream report.

## Verdict — understood, not fixed

On 17 September 2026 the bot answered a user with "⚠️ Policy service temporarily unavailable". The resilience layer worked exactly as designed — `is_transient` classified the failure as retryable, retries were exhausted cleanly, the unavailable notice went out, and no feedback row or false content escalation was created (see CLAUDE.md, "Infra resilience"). What was new was the cause: a CUDA fault on the shared Ollama host, specific to constrained-decoding (tool-calling) requests rather than a runner-wide poisoning — plain requests on the same model instance kept succeeding throughout, and the apparent ten-minute recovery was most likely our own diagnostic calls forcing a reload, not Ollama self-healing (see §1). We filed it upstream (ollama/ollama#17434, still open) and, working from that thread plus our own Phoenix traces, pinned down all five conditions that must hold at once, dated the onset to a single day, and ruled out our own `keep_alive` deploy as the cause (§8) — the leading explanation is a host-side Ollama upgrade, now an open question for the team that owns the host, not us. **The decision made today is to write the mechanism down, change nothing server-side, and take the one client-side lever the matrix already proved safe** — see §6: the confirmed fix isn't ours to make; lowering `num_ctx` to 4096 is, and it trades the crash for a documented correctness risk instead of an unmeasured gamble.

## Before you touch the host — this is time-sensitive

> Measuring a safe client-side `num_ctx` (the sweep in §4) requires the crash to still be reproducible. On 18 September it was — every tool-calling request was failing at the time, which is what made a clean sweep from 8192 down to 4096 possible in one sitting; that determinism will not always hold, so don't expect to repeat it on demand. `OLLAMA_FLASH_ATTENTION=0` clears the crash completely, and the moment that flag is set on the host — by us or by anyone else, for this reason or another — the fault disappears and the measurement window closes until something re-enables flash attention. **If this boundary ever needs re-measuring — say, after an Ollama or driver upgrade — do it while the crash is reproducible and before anyone disables flash attention; the window will not reopen on request.**

## 1 — The failure (Confirmed)

```
ollama._types.ResponseError: an error was encountered while running the model: CUDA error: an illegal memory access was encountered (status code: 500)
```

This is not a runner-wide poisoning — it's specific to constrained-decoding requests. Every tool-calling request against the loaded model gets the same 500, while plain, tool-free requests on that same model instance keep succeeding. Phoenix traces from 18 September, 10:04, show it inside a single user request, one second apart, same model instance, same `num_ctx: 8192`, same `think: false`:

```
10:04:35  Ollama.chat                       591ms  OK      <- router: no tools
10:04:36  classification -> in_scope, confidence 1.0, fallback False
10:04:36  Ollama._prepare_chat_with_tools     5ms  OK      <- agent: tools attached
10:04:36  Ollama.astream_chat                 0ms  ERROR   <- CUDA error
```

The only variable is the tools schema. This reproduces two rows of the upstream matrix (`none/false/8192 -> ok`, `constrained/false/8192 -> crash`) simultaneously in production, and is the strongest isolation of constrained decoding as the trigger we have. It also means the ten-minute "self-heal" previously assumed here was probably never Ollama replacing a poisoned runner: a plain `llm.complete` succeeded at 09:58:52 the same morning, so the model instance itself stayed healthy throughout. The more likely explanation is our own diagnostic `curl` calls — they pass no `keep_alive`, so each one resets the model's expiry to Ollama's default and forces an unload and clean reload, which looks like a self-heal from the outside but is really us restarting it.

The bot's own handling is not in question here: `is_transient` (`rag/resilience.py`) classifies the 500 as retryable (`ollama.ResponseError` plus a 5xx status, via `_is_5xx`), `retry_transient` makes the full `len(RETRY_BACKOFFS) + 1 = 4` attempts over roughly 35 seconds, and when all four still fail `_run_rag` logs `[worker] Unavailable (llm): ResponseError: ...` (`channels/teams/bot.py:94`) and returns `{"status": "unavailable"}`. The worker sends the clean unavailable notice — no rating prompt, no feedback row, no content escalation. This incident is evidence the resilience layer works, not that it failed.

## 2 — The mechanism: five conditions, all required (Reported)

Per the upstream issue, remove any one of these five and the crash does not happen:

1. **An MoE model.** Ours is `qwen3.6:latest` — arch `qwen35moe`, 36.0B, Q4_K_M, 256 experts / 8 used per token.
2. **Constrained decoding of any kind** — a JSON schema, plain `"json"` format, or tool-calling all count.
3. **`think: false`.**
4. **`num_ctx` ≥ 8192 in the upstream report** — the *allocated* context, not tokens the request actually uses. Our own sweep (§4) found this deployment's real boundary is lower: ≥4352 already crashes.
5. **Flash attention enabled** on the host.

## 3 — The reported cross matrix (Reported)

Same box, `qwen3.6:35b`, Ollama 0.32.5:

```
format         think   num_ctx   result
JSON schema    false   8192      crash
JSON schema    false   4096      ok
"json"         false   8192      crash
none           false   8192      ok
JSON schema    true    8192      ok
```

Plus: `OLLAMA_FLASH_ATTENTION=0` clears it completely (confirmed by the reporter). This only brackets the boundary between two points — crash at 8192, ok at 4096 — it doesn't locate it; §4 has our own finer sweep of that gap on this deployment.

## 4 — What our deployment adds (Ours)

The upstream report tests explicit `format=` values. Our own findings widen the trigger:

- **We reach condition 2 through tool-calling, not an explicit `format`.** `ComplianceAnswer` (`rag/agent.py`) is a parsing convention only — its JSON shape is described as plain text inside `SYSTEM_PROMPT`, and is never sent to Ollama as a `format`/JSON-schema argument (nothing in `rag/` or `channels/` sets `response_format`, `format=`, or a schema). The actual grammar constraint comes from `AgentWorkflow`'s tool-calling (`is_function_calling_model=True`, three tools registered in `build_agent()`) — condition 2 does not require an explicit `format` at all.
- **It reproduces on Ollama 0.34.0**, not just the 0.32.5 the upstream report was filed against.
- **Hardware:** NVIDIA DGX Spark GB10 — same class of box as the report.
- **Condition 4 is the allocated `num_ctx`, not usage.** `rag/agent.py` sets `additional_kwargs={"num_predict": 4096, "num_ctx": 8192}`. Measured over 83 LLM spans in Phoenix: max prompt 3,140 tokens, max completion 617, max combined **3,544** — under 43% of the 8,192 we allocate. We have never come close to the limit that supposedly matters, and we still crash.

**We also measured our own threshold directly, on 18 September, and it's tighter than the upstream range.** This was only possible because the fault had turned deterministic — every tool-calling request was failing at the time; that will not always be true, so treat this as a one-off measurement, not a repeatable procedure (see the callout above). Using the real agent code path — tools attached, `think: false`, `keep_alive` set, `num_predict: 1024` — one full `AgentWorkflow.run` per value, same question ("What is the remote work policy?"):

```
num_ctx  result
8192     CUDA crash
7168     CUDA crash
6144     CUDA crash
5120     CUDA crash
4864     CUDA crash
4608     CUDA crash
4352     CUDA crash
4096     OK  (19.1s, returned a real cited answer)
```

The threshold on this deployment is **≥4352, not ≥8192** — 4096 is the ceiling itself, not a margin below one. 4096 passing while 4352 fails points at a KV-cache allocation boundary rather than the round number in the upstream matrix.

**`keep_alive` and model residency are also exonerated.** A separate, earlier deploy — not the `num_ctx` fix in §6, which came a day later in response to the crash — landed ~45 minutes before the first CUDA error (§8), and `keep_alive` was the only thing it changed on the Ollama call path. That made it the obvious next suspect, especially with diagnostic `curl` calls that skip `keep_alive` already implicated (§1) in the misread "self-heal". It is not the cause. Tested directly, with the model verifiably unloaded first — `/api/ps` confirmed `qwen3.6:latest` specifically absent, not merely "some model" gone; an earlier version of this test reported a misleading result because it waited for `/api/ps` to be *empty*, which never happens on a shared host where another team's model stays resident:

```
A) COLD load, NO keep_alive, num_ctx=8192 : FAIL  6.1s  CUDA
B) WARM (resident), NO keep_alive, 8192   : FAIL  7.1s  CUDA
C) COLD load, NO keep_alive, num_ctx=4096 : OK   18.7s
```

Three things follow. A freshly loaded model fails identically to a resident one, so residency is irrelevant — this also kills a plausible-sounding theory that our 30-minute `keep_alive` versus Ollama's 5-minute default explained the timing. Every row above ran with no `keep_alive` at all — the exact configuration from before that deploy — and 8192 still failed: reverting `keep_alive` would not have helped. `num_ctx` is the variable that matters, and 4096 working from cold (row C) confirms the shipped fix independently of the warm-instance measurements in the sweep above.

## 5 — Recognising it in production (Ours)

- Log line: `[worker] Unavailable (llm): ResponseError: ...CUDA error...` (`channels/teams/bot.py:94`, added this branch).
- The verbose agent trace shows repeated `AgentWorkflowStartEvent` cycles — each retry rebuilds and reruns the agent from scratch.
- The user gets the unavailable notice (`[worker] Unavailable notice sent`, `channels/teams/bot.py:473`) — no rating prompt, no escalation.
- Confirm live: `curl http://172.20.0.22:11434/api/ps`, then retry the same question. The crash fires on a short prompt and a long one alike — it's the tool-calling schema that triggers it, not prompt size — while a healthy host serves both; that's what distinguishes this from a genuine capacity or latency problem. Note `/api/ps` itself is a harmless status GET; a diagnostic call that actually generates, sent without `keep_alive`, will force an unload/reload of its own and can look like a self-heal from the outside (§1).

## 6 — Options considered, one taken

| # | Option | Tradeoff |
|---|---|---|
| 1 | `OLLAMA_FLASH_ATTENTION=0` | **Confirmed fix.** But it's a server-side flag on a host this project doesn't own, and it costs flash attention — plus, in llama.cpp-derived stacks, quantized KV cache — for *every* team on that box. Contention there is already real: `gpt-oss:120b` at 64.5 GB evicted our model on the day of the incident. |
| 2 | Lower `num_ctx` client-side | **Taken, 18 September** (`OLLAMA_NUM_CTX=4096`, `config.py`). Free, entirely ours. The upstream matrix (§3) suggested 4096 was safe; our own sweep (§4) has since measured this deployment's boundary directly and found it sits right above 4096 (4352 already crashes) — so 4096 is a hard ceiling, not headroom. Cost: see below. |
| 3 | `think: true` | Proven safe at `num_ctx=8192` (matrix row 5). Costs reasoning tokens on every answer — more affordable since the worker-queue change landed (users are acked in under a second, so a longer answer is a wait, not silence) — but still unmeasured for our agent/prompt shape. |
| 4 | Dense model instead of MoE | **Rejected on latency.** `batiai/qwen3.6-27b:q6` (26.9B) and `qwen2.5:32b-instruct-q8_0` (32.8B) are both dense — every token reads all weights, against our MoE's 8-of-256 experts. Roughly 4–7× slower decode, turning a ~16 s answer into 60–100 s. `qwen2.5:32b` also caps at a 32k context (vs. our 262k) and has no thinking mode. |
| 5 | Downgrade Ollama | The regression reportedly landed in 0.32.5; we're on 0.34.0 and the upstream issue is still open, so there is no known-good version to move to — and downgrading a shared box several releases for one project's benefit is a large ask. |

**What option 2 costs.** Lowering `num_ctx` to 4096 — now measured as a hard ceiling, not a comfortable margin (§4) — trades the crash for less room for the prompt and answer to share:

- **Budget at 4096.** Observed max prompt 3,140 tokens leaves 956 for the answer — comfortably over the observed max completion of 617. Theoretical worst case does not fit: ~3,866 tokens (1,116-token system prompt + 6 chunks × `chunk_max_tokens` 400 + tool schemas + question) would leave only ~230.
- **Failure mode if exceeded.** Ollama silently truncates the prompt — no error. On a compliance bot that means retrieved policy chunks or the system prompt itself can be dropped without warning, producing an answer that is ungrounded or missing citations. That's a correctness risk, not a performance one, which is why it's worth watching rather than filing away.
- **Levers if truncation shows up.** `OLLAMA_NUM_CTX` is not one of them while this bug is open — despite being a plain `.env` value, raising it re-enters the crash rather than buying headroom (§4: the boundary is 4352, one step above 4096). The two levers that don't touch it are reducing `reranker_top_n` (currently 6) or `chunk_max_tokens` (currently 400) to shrink the prompt — both trade retrieval quality on a compliance system, so neither is free either. `OLLAMA_NUM_CTX` becomes a real lever again only once the upstream issue closes or the host disables flash attention.

## 7 — Residual risk

`OLLAMA_NUM_CTX=4096` (client-side, §6) keeps condition 4 false no matter what the host does with flash attention, so the mechanism in §2 should no longer trigger through our traffic. Two risks remain. The old one, softened rather than closed: if `num_ctx` is ever raised above 4096 by someone who hasn't read this document, the crash comes back — and per §4's measured boundary it doesn't take much: 4352 is already enough, well short of the 8192 someone skimming only the upstream issue might assume is the danger zone. `config.py` carries the warning, but warnings get skipped, and without this document the crash would look like a brand-new mystery instead of a five-condition failure mode we already understand. The new one, introduced by the fix itself: capping the prompt+answer budget at 4096 tokens risks the silent truncation described in §6 — a correctness problem, not an availability one, and one we haven't hit yet.

## 8 — Why now: onset timeline, and an open question for the host team (Ours)

A full Phoenix audit, 2026-05-04 through 2026-09-18, dates the onset precisely and points away from anything we did:

- ~74 `AgentWorkflow.run` spans from May through 2026-09-15, zero ERROR spans — the same code path, tools attached, `num_ctx: 8192`, `thinking=False`, throughout.
- Agent runs by day: teens per day from 4–28 May, then sparse single runs through June and July, then 2026-09-10 (1 run) and 2026-09-15 (2 runs) — all clean.
- **First CUDA error: 2026-09-17T12:02:21.** Near-total failure of tool-calling requests since.
- The model itself hasn't changed since 2026-04-24 (`/api/tags` `modified_at`).

`num_ctx: 8192` and `thinking=False` predate all of this by months, so neither is what changed — and neither is our own `keep_alive` change, which is independently ruled out above (§4).

**Open question for the host team — a hypothesis, not yet a finding.** The leading explanation is that the Ollama host itself was upgraded. The upstream issue reports the regression landing in 0.32.5; `172.20.0.22` now runs 0.34.0. If the host crossed 0.32.5 between 15 and 17 September, every fact above fits: four clean months, an abrupt onset, a known regression, identical hardware throughout. We can't confirm this without the team that owns the host — that's the question to put to them. A confirmed date would also be useful upstream: it would date the regression against a real production workload instead of only the reporter's own synthetic probe.

This resolves what earlier drafts of this document left open: the cause is a host-side change (pending the host team's confirmation above), not our code and not our config. The trigger itself, independent of when or why the host changed, is `num_ctx` ≥ 4352 together with constrained decoding (§2, §4). Our client-side fix (§6, option 2) stands on its own regardless of how the host question resolves.

---

**See also:** `CLAUDE.md` → Gotchas / Lessons Learned (short version, log signature); `docs/superpowers/specs/2026-09-16-scaling-audit.md` (host contention, MoE model profile, the `keep_alive`/eviction backdrop this incident sits on).

---

## 9. 2026-09-24 — flash attention disabled, `num_ctx` raised, reverted the same hour

The Spark admin set `OLLAMA_FLASH_ATTENTION=0`, removing one of the five conditions.
We raised `num_ctx` back to 8192 and `num_predict` to 4096 (commit a0097a4), and
**reverted within the hour after the fault reappeared in production**.

### What we measured before shipping

Through a hand-rolled `/api/chat` call with a tool schema, `think:false`, temperature 0:

| `num_ctx` | §4 sweep (FA on) | 2026-09-24 (FA off) |
|---|---|---|
| 4352 / 6144 / 8192 | **FAIL — CUDA 500, every time** | OK |
| 8192 + `num_predict` 4096 | **FAIL** | OK, 5 consecutive |

Eight passes at values that had never once passed. We shipped.

### What happened

Trace `940a8169…`, 09:17:11, the first real question after the deploy:
`Ollama.chat` (router, no tools) OK in 627ms, then **four** `AgentWorkflow.run`
attempts, every one `ERROR` on `Ollama.astream_chat` with
`CUDA error: an illegal memory access was encountered`, then `infra_unavailable`.
A user saw "⚠️ Policy service temporarily unavailable".

Re-probed minutes later: 8192 passed again, including with a production-shaped
request (~3,000-token system prompt, three tool schemas). So the fault did not
return permanently — it is now **intermittent**.

### Why the verification was insufficient — the part worth remembering

1. **A pass-based probe cannot distinguish "fixed" from "less frequent".** The old
   fault was deterministic, so a handful of passes was meaningful evidence. Removing
   a condition changed the fault's *character*, not just its rate — and against an
   intermittent fault, N passes prove only that N passes are possible. The right
   test is duration and volume under real traffic, not a burst of probes.
2. **The probe was weaker than the measurement it claimed to supersede.** §4's sweep
   drove the real `AgentWorkflow` path. This one hand-rolled the HTTP request, so it
   exercised neither the real prompt, nor the three real tool schemas, nor the agent's
   multi-turn loop — and the production failure occurred on the second agent turn.
3. **The probes passed no `keep_alive`**, so each reset the model's TTL to Ollama's
   5-minute default and may have forced reloads between tests — the same artefact
   §1 already identified as faking a self-heal on 17 September. A probe that perturbs
   residency is measuring something other than steady state.

### Current position

`num_ctx` 4096, `num_predict` 1024 — the configuration that served production for a
week with zero CUDA errors. `OLLAMA_FLASH_ATTENTION=0` stays on the host and is
presumably still worth having; it is simply not sufficient on its own.

Raising `num_ctx` again needs evidence of a different kind: a sustained period of
real traffic at 8192 on a non-production path, or an upstream fix in
ollama/ollama#17434 (still open). Eight probe passes is not that evidence, and this
section exists so nobody repeats the inference.
