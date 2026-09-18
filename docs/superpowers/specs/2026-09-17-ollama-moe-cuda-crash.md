# Ollama MoE CUDA Crash — Diagnosed, Not Fixed

**Date:** 17 September 2026 (incident) · 18 September 2026 (written up) · **Type:** incident diagnosis — documented, not fixed
**Upstream:** ollama/ollama#17434 — <https://github.com/ollama/ollama/issues/17434> (open; filed by us)

A production 500 that took down every Ollama request on the shared Spark host, traced to a five-condition MoE + CUDA interaction inside Ollama itself — not a bug in this bot. Recorded here because the one confirmed fix (§6) is server-side, on a host this project doesn't own; the one option that was ours — lowering `num_ctx` client-side — has been taken, and its cost is recorded in §6.

Provenance: **Reported** = from the upstream issue thread, reproduced by the reporter on their own box (`qwen3.6:35b`, Ollama 0.32.5) — not independently re-run by us. **Confirmed** = holds against our own deployment. **Ours** = specific to this deployment, not covered by the upstream report.

## Verdict — understood, not fixed

On 17 September 2026 the bot answered a user with "⚠️ Policy service temporarily unavailable". The resilience layer worked exactly as designed — `is_transient` classified the failure as retryable, retries were exhausted cleanly, the unavailable notice went out, and no feedback row or false content escalation was created (see CLAUDE.md, "Infra resilience"). What was new was the cause: a CUDA fault on the shared Ollama host that poisons the model runner for *every* request until Ollama replaces it, roughly ten minutes later. We filed it upstream (ollama/ollama#17434, still open) and, working from that thread plus our own Phoenix traces, pinned down all five conditions that must hold at once. **The decision made today is to write the mechanism down, change nothing server-side, and take the one client-side lever the matrix already proved safe** — see §6: the confirmed fix isn't ours to make; lowering `num_ctx` to 4096 is, and it trades the crash for a documented correctness risk instead of an unmeasured gamble.

## Before you touch the host — this is time-sensitive

> Measuring a safe client-side `num_ctx` (the table in §6) requires the crash to still be reproducible. `OLLAMA_FLASH_ATTENTION=0` clears it completely. The moment that flag is set on the host — by us or by anyone else, for this reason or another — the fault disappears and the measurement window closes until something re-enables flash attention. **If a client-side `num_ctx` value is ever wanted, measure it before asking the host owner to disable flash attention, not after.**

## 1 — The failure (Confirmed)

```
ollama._types.ResponseError: an error was encountered while running the model: CUDA error: an illegal memory access was encountered (status code: 500)
```

This poisons the Ollama model-runner process serving our model: every subsequent request — ours or anyone else's on the box — gets the same 500 until Ollama replaces the runner. It self-heals in roughly ten minutes with no intervention from us.

The bot's own handling is not in question here: `is_transient` (`rag/resilience.py`) classifies the 500 as retryable (`ollama.ResponseError` plus a 5xx status, via `_is_5xx`), `retry_transient` makes the full `len(RETRY_BACKOFFS) + 1 = 4` attempts over roughly 35 seconds, and when all four still fail `_run_rag` logs `[worker] Unavailable (llm): ResponseError: ...` (`channels/teams/bot.py:94`) and returns `{"status": "unavailable"}`. The worker sends the clean unavailable notice — no rating prompt, no feedback row, no content escalation. This incident is evidence the resilience layer works, not that it failed.

## 2 — The mechanism: five conditions, all required (Reported)

Per the upstream issue, remove any one of these five and the crash does not happen:

1. **An MoE model.** Ours is `qwen3.6:latest` — arch `qwen35moe`, 36.0B, Q4_K_M, 256 experts / 8 used per token.
2. **Constrained decoding of any kind** — a JSON schema, plain `"json"` format, or tool-calling all count.
3. **`think: false`.**
4. **`num_ctx` ≥ 8192** — the *allocated* context, not tokens the request actually uses.
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

Plus: `OLLAMA_FLASH_ATTENTION=0` clears it completely (confirmed by the reporter).

## 4 — What our deployment adds (Ours)

The upstream report tests explicit `format=` values. Our own findings widen the trigger:

- **We reach condition 2 through tool-calling, not an explicit `format`.** `ComplianceAnswer` (`rag/agent.py`) is a parsing convention only — its JSON shape is described as plain text inside `SYSTEM_PROMPT`, and is never sent to Ollama as a `format`/JSON-schema argument (nothing in `rag/` or `channels/` sets `response_format`, `format=`, or a schema). The actual grammar constraint comes from `AgentWorkflow`'s tool-calling (`is_function_calling_model=True`, three tools registered in `build_agent()`) — condition 2 does not require an explicit `format` at all.
- **It reproduces on Ollama 0.34.0**, not just the 0.32.5 the upstream report was filed against.
- **Hardware:** NVIDIA DGX Spark GB10 — same class of box as the report.
- **Condition 4 is the allocated `num_ctx`, not usage.** `rag/agent.py` sets `additional_kwargs={"num_predict": 4096, "num_ctx": 8192}`. Measured over 83 LLM spans in Phoenix: max prompt 3,140 tokens, max completion 617, max combined **3,544** — under 43% of the 8,192 we allocate. We have never come close to the limit that supposedly matters, and we still crash.

## 5 — Recognising it in production (Ours)

- Log line: `[worker] Unavailable (llm): ResponseError: ...CUDA error...` (`channels/teams/bot.py:94`, added this branch).
- The verbose agent trace shows repeated `AgentWorkflowStartEvent` cycles — each retry rebuilds and reruns the agent from scratch.
- The user gets the unavailable notice (`[worker] Unavailable notice sent`, `channels/teams/bot.py:473`) — no rating prompt, no escalation.
- Confirm live: `curl http://172.20.0.22:11434/api/ps`, then retry the same question. A poisoned runner fails a short prompt and a long one alike; a healthy host serves both. That's what distinguishes this from a genuine capacity or latency problem.

## 6 — Options considered, one taken

| # | Option | Tradeoff |
|---|---|---|
| 1 | `OLLAMA_FLASH_ATTENTION=0` | **Confirmed fix.** But it's a server-side flag on a host this project doesn't own, and it costs flash attention — plus, in llama.cpp-derived stacks, quantized KV cache — for *every* team on that box. Contention there is already real: `gpt-oss:120b` at 64.5 GB evicted our model on the day of the incident. |
| 2 | Lower `num_ctx` client-side | **Taken, 18 September** (`OLLAMA_NUM_CTX=4096`, `config.py`). Free, entirely ours, and the matrix in §3 already proved 4096 safe — no new measurement needed. Cost: see below. |
| 3 | `think: true` | Proven safe at `num_ctx=8192` (matrix row 5). Costs reasoning tokens on every answer — more affordable since the worker-queue change landed (users are acked in under a second, so a longer answer is a wait, not silence) — but still unmeasured for our agent/prompt shape. |
| 4 | Dense model instead of MoE | **Rejected on latency.** `batiai/qwen3.6-27b:q6` (26.9B) and `qwen2.5:32b-instruct-q8_0` (32.8B) are both dense — every token reads all weights, against our MoE's 8-of-256 experts. Roughly 4–7× slower decode, turning a ~16 s answer into 60–100 s. `qwen2.5:32b` also caps at a 32k context (vs. our 262k) and has no thinking mode. |
| 5 | Downgrade Ollama | The regression reportedly landed in 0.32.5; we're on 0.34.0 and the upstream issue is still open, so there is no known-good version to move to — and downgrading a shared box several releases for one project's benefit is a large ask. |

**What option 2 costs.** Lowering `num_ctx` trades the crash for less room for the prompt and answer to share:

- **Budget at 4096.** Observed max prompt 3,140 tokens leaves 956 for the answer — comfortably over the observed max completion of 617. Theoretical worst case does not fit: ~3,866 tokens (1,116-token system prompt + 6 chunks × `chunk_max_tokens` 400 + tool schemas + question) would leave only ~230.
- **Failure mode if exceeded.** Ollama silently truncates the prompt — no error. On a compliance bot that means retrieved policy chunks or the system prompt itself can be dropped without warning, producing an answer that is ungrounded or missing citations. That's a correctness risk, not a performance one, which is why it's worth watching rather than filing away.
- **Levers if truncation shows up.** Reduce `reranker_top_n` (currently 6) or `chunk_max_tokens` (currently 400) to shrink the prompt, or raise `OLLAMA_NUM_CTX` from `.env` once the upstream issue closes or the host disables flash attention.

## 7 — Residual risk

`OLLAMA_NUM_CTX=4096` (client-side, §6) keeps condition 4 false no matter what the host does with flash attention, so the mechanism in §2 should no longer trigger through our traffic. Two risks remain. The old one, softened rather than closed: if `num_ctx` is ever raised back toward 8192 by someone who hasn't read this document — `config.py` carries the warning, but warnings get skipped — the crash comes back exactly as before, and without this document it would look like a brand-new mystery instead of a five-condition failure mode we already understand. The new one, introduced by the fix itself: capping the prompt+answer budget at 4096 tokens risks the silent truncation described in §6 — a correctness problem, not an availability one, and one we haven't hit yet.

---

**See also:** `CLAUDE.md` → Gotchas / Lessons Learned (short version, log signature); `docs/superpowers/specs/2026-09-16-scaling-audit.md` (host contention, MoE model profile, the `keep_alive`/eviction backdrop this incident sits on).
