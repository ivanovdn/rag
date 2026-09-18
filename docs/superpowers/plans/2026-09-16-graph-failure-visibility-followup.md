# Graph Failure Visibility — Follow-up Scoping Note

> **Status:** open questions, not a plan. No tasks, no steps — this records decisions to make
> before anyone writes code against them.

**Goal:** decide what the bot should do when Microsoft Graph stays broken.

Two open items came out of reviewing the Teams worker-queue branch (`feat/teams-worker-queue`,
plan `2026-09-16-scaling-to-30-users.md`). They look unrelated but they are the same policy
question asked at two levels: *inbound* (we cannot read Graph) and *outbound* (we cannot write to
it). Both were left deliberately unfixed there — that branch retried the transient case and made
the residual loud, which is as far as it could go without choosing a policy.

---

## 1. `_api_request` cannot distinguish "empty" from "broken"

**Where:** `channels/teams/bot.py:218` (`_api_request`), catch-all at `:252-254`; consumed at
`:502` and `:515`; error counter at `:579`.

`_api_request` catches every exception and returns `None`. `process_new_messages` reads that as
"no chats" (`chats = chats_data.get("value", []) if chats_data else []`) and quietly does nothing.
Because nothing raised, `run()` then resets `consecutive_errors = 0`.

So a dead refresh token, revoked consent, or a Graph outage is **indistinguishable from a quiet
afternoon** — indefinitely. `teams_max_consecutive_errors` never trips. The bot looks healthy in
every signal it emits while being completely dead. The bounded retry added in
`fix(teams): retry transient Graph failures` narrows the window for a blip but does not change
this: after the retries are exhausted the result is still `None`, still indistinguishable.

**Options to weigh:**

- Return a sentinel that separates "call failed" from "call succeeded, no data", so callers can
  tell the difference at all. Probably a prerequisite for any of the others.
- Let repeated hard failures reach the existing `consecutive_errors` counter instead of resetting
  it, so `teams_max_consecutive_errors` does its job.
- Exit non-zero and let Docker's restart policy take over (`docker-compose-remote.yml`).
- Distinguish **401/403** (credentials — will not self-heal, should be loud immediately, and a
  restart will not help) from **5xx** (wait it out). This is the distinction that makes any of the
  above proportionate rather than a hair-trigger.

**Open question:** loud-and-fail-fast vs quiet-and-keep-trying. A compliance bot that is silently
dead is arguably worse than one that restart-loops visibly, but that depends on who is watching
the logs, which is currently nobody by default.

---

## 2. Should an undelivered answer hold the watermark?

**Where:** `channels/teams/bot.py` — `_worker_loop`'s `finally`, and `_save_state`'s watermark
hold-back.

Today an answer whose reply POST fails after retries is logged as an ERROR, but its id is still
popped from `_inflight`; the watermark advances and the id stays in `processed_messages`. Nothing
re-sends it. The user has an ack and then silence.

Holding the id in `_inflight` instead would make a restart re-deliver it — consistent with the
at-least-once semantics the queue already chose. The cost: a **permanently** undeliverable message
(deleted chat, revoked access) would pin the persisted watermark indefinitely, so every restart
re-answers everything after that point, forever.

**What makes a bounded version possible:**

- Distinguishing permanent (4xx) from persistent-transient (5xx, timeouts) failures — the same
  distinction item 1 needs. A 4xx is never worth holding; a 5xx might be.
- A cap: hold at most N times, or at most M minutes, then give up and log.
- Note that `_load_state`'s 60-minute staleness clamp (`:131`, `teams_max_state_age_minutes`)
  already bounds the damage **across a restart** — a pinned watermark older than that is discarded.
  It does **not** bound it *within* a running process, where the watermark would stay pinned for as
  long as the process lives.

**Open question:** is a duplicate answer after restart cheaper than a lost one? The queue work
already answered "yes" for the crash case. Whether the same answer holds when delivery itself is
what failed is the decision to make.

---

## Incidental finding

The `[media/emoji]` guard in `_handle_inbound` (`channels/teams/bot.py:290`) is **unreachable**.
`process_new_messages` passes `clean_message` (`strip_html` output), which is empty for a media or
emoji message, so the empty-text guard immediately above it returns first. The literal
`"[media/emoji]"` only ever exists in the log line at `:540`. Harmless, but it is dead code that
reads like a live branch. Delete it, or make the caller actually pass the placeholder — not worth
its own change, so fold it into whichever of the above lands first.
