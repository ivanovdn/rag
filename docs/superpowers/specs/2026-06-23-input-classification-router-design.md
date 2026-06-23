# Pre-retrieval input classification router (ROUTER-1/2/3 + MSG-1)

**Date:** 2026-06-23
**Status:** Approved (design), pending implementation
**Author:** Dmytro Ivanov (with Claude Code)

A pre-retrieval gate that classifies every inbound Teams message and lets **only in-scope
policy questions** reach the RAG pipeline. Greetings, out-of-scope messages, and
unintelligible input get fixed deterministic replies and never trigger a search.

---

## Problem

Today every message that isn't the literal `start`/`help` command (`channels/teams/bot.py:206`)
or a pending rating goes straight into `_run_rag`:

- **Greetings** ("hi", "hello") run the full agent → `search_policies` → almost certainly
  `NO_RELEVANT_POLICY_FOUND` → escalation. A wasted 30–60s lookup and often a spurious escalation.
- **Out-of-scope** messages ("order me a pizza", "who is Sarah Connor") take the same route.
  Once escalation *delivery* (#1) is built, every off-topic message would file a Compliance ticket.
- **Unintelligible** input — most commonly English typed with a Cyrillic keyboard layout still
  active (e.g. "црфе ші …" = "what is …" on a Ukrainian layout) — also searches and escalates.

There is no place in the pipeline that decides "should this message be searched at all?"

## Goal

Classify each message **before retrieval** into exactly one of four categories and act
deterministically on the category. Only in-scope questions reach policy search. The classifier
must never be able to deny the product's core function: when it is uncertain or fails, the
message is treated as a real question and searched.

## Categories and actions

| Category | Action | Search? | Rating prompt? |
|---|---|---|---|
| `in_scope` | existing RAG flow (`LOADING_HTML` → `_run_rag` → answer / escalation / unavailable) | yes | yes (unchanged) |
| `greeting` | reply with `WELCOME_HTML` (reuse the existing welcome/disclaimer) | no | no |
| `out_of_scope` | reply with `OUT_OF_SCOPE_HTML` (fixed redirect) | no | no |
| `unintelligible` | reply with `UNINTELLIGIBLE_HTML` (ask to retype) | no | no |

Non-`in_scope` replies create no feedback row and set no pending rating — same treatment as the
existing "unavailable" outcome.

**Explicitly dropped for v1 (YAGNI):** an `escalation_request` category. An explicit "talk to a
human" message folds into `in_scope` — it gets searched, and the agent already escalates when no
policy matches. Revisit when escalation delivery (#1) lands.

## Decisions (locked)

- **Mechanism:** a single LLM classification call per message (not rules). The two hardest
  categories — out-of-scope and unintelligible/wrong-layout — require semantic judgment that
  keyword rules cannot provide, while the LLM handles all four uniformly.
- **Safe-default bias (the entire safety story):** the classifier can only ever *add*
  short-circuits for high-confidence non-questions. It can never refuse a real question:
  - `confidence < ROUTER_CONFIDENCE_FLOOR` → resolve to `in_scope` regardless of predicted category.
  - classifier LLM failure (after retries) or unparseable output → `in_scope`.
- **Temperature 0.0** for the classifier (project-wide determinism constraint).
- **Classifier uses a plain structured completion, no tools.** Unlike the main agent (where
  `response_format=json_object` kills tool-call emission — see CLAUDE.md gotcha), the classifier
  has no tools, so a JSON-only instruction + parse is correct and safe here.
- **Everything tuneable** — see the Tuning surface section. Knobs live in `.env`; multi-line
  text (prompt, messages) lives as editable code constants (a multi-line value in `.env` would
  hit the documented `\n`-sent-literally gotcha).

## Components

### `rag/router.py` (new)

```python
class Category(str, Enum):
    IN_SCOPE = "in_scope"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"
    UNINTELLIGIBLE = "unintelligible"

class RouterDecision(BaseModel):
    category: Category
    confidence: float          # 0..1, classifier's certainty
    fallback: bool = False      # True when set by the failure path, not the model
```

- `ROUTER_SYSTEM_PROMPT` — editable module constant (the tuning surface for classifier
  behavior; the same place the main agent keeps `SYSTEM_PROMPT`). Defines each category with
  examples, including the wrong-keyboard-layout case, and the explicit instruction: *when unsure
  whether a message is a real policy question, choose `in_scope`.* The prompt demands output of
  exactly one JSON object `{"category": "...", "confidence": <0..1>}` and nothing else.

- `classify_message(text: str) -> RouterDecision` — builds the classifier LLM
  (`ROUTER_LLM_MODEL` if set, else the main `get_llm()`), temperature 0, issues one completion
  with `ROUTER_SYSTEM_PROMPT` + the message, parses the JSON (tolerant extraction like
  `rag/response.py`) into `RouterDecision`. The LLM call is wrapped in `retry_transient` (reuses
  `rag/resilience.py`). On any transient-exhaustion, non-transient error, or parse/validation
  failure it returns `RouterDecision(category=IN_SCOPE, confidence=0.0, fallback=True)` — never
  raises.

- `resolve(decision: RouterDecision, floor: float) -> Category` — **pure** function (no LLM, unit
  tested):
  - if `decision.fallback` → `IN_SCOPE`
  - elif `decision.confidence < floor` → `IN_SCOPE`
  - else → `decision.category`

The split keeps the LLM call (integration/corpus-tested) separate from the floor/fallback logic
(pure, unit-tested without any service).

### `channels/teams/bot.py` — `_send_reply` wiring

The classifier runs **after** the existing deterministic fast-paths (`start`/`help` command,
`[media/emoji]`, pending-rating handling) and **before** `LOADING_HTML`. Revised order:

1. empty/whitespace → return *(unchanged)*
2. `start`/`help` → `WELCOME_HTML` *(unchanged)*
3. `[media/emoji]` → ignore *(unchanged)*
4. pending rating + valid rating → save feedback *(unchanged)*
5. clear pending state *(unchanged)*
6. **NEW — classify (only if `ROUTER_ENABLED`):**
   - `decision = classify_message(text)`
   - `category = resolve(decision, settings.router_confidence_floor)`
   - `record_classification(category.value, decision.confidence, fallback=(category != decision.category or decision.fallback))`
   - `greeting` → send `WELCOME_HTML`; return (no search, no rating)
   - `out_of_scope` → send `render_out_of_scope()`; return (no search, no rating)
   - `unintelligible` → send `render_unintelligible()`; return (no search, no rating)
   - `in_scope` → fall through to step 7
7. existing path: `LOADING_HTML` → `_run_rag(text)` → answer / escalation / unavailable + rating
   prompt *(unchanged)*

When `ROUTER_ENABLED` is false, step 6 is skipped entirely and every message flows to step 7 —
exactly today's behavior. The loading indicator now appears only for in-scope questions.

Imports of `rag.router` are deferred inside `_send_reply`/`_run_rag` like the existing RAG
imports, preserving the `init_observability()`-first rule.

### `channels/teams/renderer.py` — new messages

Add editable module constants + thin render functions, same pattern as `UNAVAILABLE_HTML` /
`render_unavailable()`:

- `OUT_OF_SCOPE_HTML` + `render_out_of_scope() -> str` — e.g. *"I can only answer questions about
  company policies. Ask me about a policy and I'll find the relevant section and clause."*
- `UNINTELLIGIBLE_HTML` + `render_unintelligible() -> str` — e.g. *"I couldn't read that — it may
  have been typed with a different keyboard layout. Please retype your question."*

Greeting reuses the existing `WELCOME_HTML` constant directly (no new constant, no render fn).

### `rag/observability.py` — classification signal

- `record_classification(category: str, confidence: float, fallback: bool) -> None` — opens/annotates
  a span with attributes `router_category`, `router_confidence`, `router_fallback`. Mirrors
  `record_infra_unavailable`. Makes the classification distribution and the safe-default fallback
  rate queryable in Phoenix (e.g. to spot a too-high floor starving real questions, or a category
  the classifier gets wrong).

### `config.py` + `.env.example` — knobs

- `ROUTER_ENABLED: bool = True` — kill switch. False → no classification, today's behavior.
- `ROUTER_LLM_MODEL: str = ""` — classifier model override; empty → use the main LLM.
- `ROUTER_CONFIDENCE_FLOOR: float = 0.6` — below this, resolve to `in_scope`.

## Tuning surface (everything tuneable)

| Knob | Where | Effect |
|---|---|---|
| `ROUTER_ENABLED` | `.env` | turn the whole router on/off |
| `ROUTER_LLM_MODEL` | `.env` | swap the classifier model (smaller/faster) |
| `ROUTER_CONFIDENCE_FLOOR` | `.env` | how aggressively to fall back to search |
| `ROUTER_SYSTEM_PROMPT` (incl. category definitions + examples) | `rag/router.py` constant | classifier behavior |
| `OUT_OF_SCOPE_HTML` | `renderer.py` constant | out-of-scope copy |
| `UNINTELLIGIBLE_HTML` | `renderer.py` constant | retype-prompt copy |
| greeting copy | existing `WELCOME_HTML` constant | greeting reply |

`.env` holds scalars/flags (runtime-tuneable, no redeploy); multi-line text lives as code
constants (avoids the `\n`-in-`.env` gotcha and matches how the main agent's `SYSTEM_PROMPT` is
handled). Prompt/message edits take effect on restart.

## Data flow

```
message → _send_reply
  → [command / media / rating fast-paths]      (unchanged)
  → ROUTER_ENABLED? ── no ──────────────────────────────────► _run_rag (today's behavior)
        │ yes
        ▼
     classify_message(text)  ── LLM (temp 0, retry_transient) ──► RouterDecision{category, confidence, fallback}
        ▼
     resolve(decision, floor) ── pure ──► Category
        ▼
     record_classification(...)  ──► Phoenix span
        ▼
   ┌── greeting        → WELCOME_HTML            (no search)
   ├── out_of_scope    → OUT_OF_SCOPE_HTML       (no search)
   ├── unintelligible  → UNINTELLIGIBLE_HTML     (no search)
   └── in_scope        → LOADING_HTML → _run_rag → answer/escalation/unavailable + rating
```

### In-scope path (unchanged — the router is purely additive in front of it)

`in_scope` falls through to today's exact pipeline; nothing in the answer path changes. Full trace:

1. `_send_reply` sends `LOADING_HTML` (now the *only* category that shows it), then calls
   `_run_rag(text)`.
2. `_run_rag` resets `sp._retrieval_unavailable = False`, then runs the agent under
   `retry_transient(lambda: asyncio.run(agent.run(user_msg=question)))`.
3. The `AgentWorkflow` (system prompt + 3 tools) calls **`search_policies`** first with the
   original question → `embed_query` → `vector_search` → `[BM25 RRF off]` → `rerank` → top
   `RERANKER_TOP_N` → `format_sources()` with `[Source N]` headers; optionally `get_section`;
   then emits the `ComplianceAnswer` JSON, or calls `escalate_to_compliance` when nothing matched.
   - transient blip inside the tool → sets `sp._retrieval_unavailable=True`, returns
     `POLICY_SEARCH_UNAVAILABLE`; empty retrieval → `NO_RELEVANT_POLICY_FOUND`.
4. `_run_rag` resolves the outcome: agent-LLM transient error → `record_infra_unavailable("llm")`
   → `{"status":"unavailable"}`; non-transient error → escalation dict with `reason=str(e)`;
   `sp._retrieval_unavailable` set → `{"status":"unavailable"}`; otherwise
   `parse_agent_response(...)` → `{answer, citations, escalation}`.
5. `_send_reply` branches: `status=="unavailable"` → `render_unavailable()` (no rating);
   `escalation.needed` → `render_escalation`; has `answer` → `render_answer`; else `render_error`.
   On a sent answer/escalation it then sends `RATING_PROMPT_HTML` and stores the pending-rating
   context.

The four resilience outcomes, escalation, citations, and the rating loop are all preserved. The
router only decides whether a message *enters* this pipeline; it never alters it.

## Error handling

- Classifier transient infra failure (after `retry_transient`): `classify_message` returns the
  `fallback=True` decision → `resolve` → `in_scope` → `_run_rag`. If the LLM is genuinely down,
  the downstream agent call surfaces the existing "unavailable" outcome — no duplicated logic.
- Classifier returns malformed/unparseable JSON or an unknown category: same `fallback=True`
  → `in_scope`.
- Net invariant: **the router can never refuse a real question through its own uncertainty or
  failure.** It only ever short-circuits high-confidence non-questions.

## Testing

**Unit (pure logic, no services):**
- `resolve`: confidence below floor → `IN_SCOPE` for every predicted category; `fallback=True`
  → `IN_SCOPE`; confidence at/above floor → predicted category passes through.
- Bot routing: each resolved category dispatches the correct reply (mock `classify_message`):
  greeting→`WELCOME_HTML`, out_of_scope→`render_out_of_scope`, unintelligible→
  `render_unintelligible`, in_scope→`_run_rag` path. Non-`in_scope` sets no pending rating.
- `ROUTER_ENABLED=false` bypasses classification entirely (`classify_message` not called).
- Renderer: `OUT_OF_SCOPE_HTML` and `UNINTELLIGIBLE_HTML` are safe Teams-limited HTML (same
  assertion style as `test_render_unavailable_is_safe_html`).

**Classifier accuracy (corpus-style, auto-skips without a live LLM):**
- A small committed labeled set of messages per category — greetings, real policy questions,
  out-of-scope ("order me a pizza", "who is Sarah Connor"), and unintelligible (the "црфе ші…"
  wrong-layout example, random characters) — asserting `classify_message` returns the expected
  category. Same Tier-A auto-skip pattern as `tests/docs/`. Treated as a tuning signal, not a
  hard gate (threshold-based, like retrieval-hit tests).

## What is explicitly unchanged

- The `start`/`help` command, `[media/emoji]` handling, and rating/feedback flow.
- `_run_rag` and the entire RAG pipeline (search, rerank, agent, response parsing, infra
  resilience). The router sits in front; it does not touch retrieval.
- Genuine `NO_RELEVANT_POLICY_FOUND` → escalation for in-scope questions.

## Out of scope (YAGNI)

- `escalation_request` category (folded into `in_scope`).
- Auto-decoding wrong-keyboard-layout text (we ask the user to retype).
- Multi-turn clarify loops / conversational state (the router is stateless per message).
- Escalation delivery (#1) — a separate feature; the router only reduces what reaches it.

## Risks

- **Classifier false negatives** (a real question tagged greeting/out-of-scope/unintelligible
  above the floor): the one error mode that denies the product's function. Mitigated by the
  safe-default bias (floor) + the prompt's "when unsure, in_scope" instruction + the Phoenix
  `router_fallback` signal to tune the floor + the `ROUTER_ENABLED` kill switch. The corpus
  accuracy set guards against regressions in the prompt.
- **Added latency** on in-scope questions: one extra classifier call before the 30–60s pipeline.
  Negligible relative to the pipeline, and it *saves* the full lookup on every non-question.
  `ROUTER_LLM_MODEL` lets a smaller model cut it further.
- **Classifier model drift / availability:** the classifier is just another LLM call on the same
  box; it shares the infra-resilience retry path and fails safe to `in_scope`.
