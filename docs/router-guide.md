# Compliance Q&A Bot — How It Triages Messages

*A plain-language guide to what the bot does with each message **before** it searches any
policy. Written for the Compliance/Legal reviewers and anyone evaluating the bot — no technical
background needed.*

---

## What changed

The bot now reads every incoming message and sorts it into one of **four kinds** before doing
anything else. Only genuine policy questions trigger a policy search. Everything else gets an
instant, fixed reply.

Why this matters:

- A "hi" or an off-topic message no longer triggers a slow (~30–60 second) policy lookup.
- Off-topic messages no longer risk being forwarded to the Compliance team as if they were
  unanswerable compliance questions.
- Every decision is **recorded for audit**: the exact message, the kind the bot chose, and how
  confident it was.

---

## The four kinds of message

All examples below are **real messages from testing**.

### 1. Greeting
A hello or a thank-you, with no question.

- Examples: `hola` · `привіт`
- **What the bot replies:** the welcome message — who it is and how to ask a question. *No search.*

### 2. Policy question (in scope)
Something an internal company policy could actually answer.

- Examples: `My antivirus slows down my laptop. Can I disable it while working?` ·
  `Does the company sell my personal data?`
- **What the bot replies:** it searches the approved policies and answers by pointing to the
  exact policy, section, and clause (quoting the text). If no policy covers the question, it
  escalates to the Compliance team. *This is the only kind that searches.*

### 3. Off-topic (out of scope)
A real request, but not something company policy covers.

- Examples: `i wanna order pizza` · `how is Liverpool fc coach?`
- **What the bot replies:** *"I can only answer questions about company policies. Ask me about a
  policy and I'll find the relevant section and clause."* *No search.*

### 4. Unreadable
Gibberish, random characters, or text typed with the wrong keyboard layout.

- Examples: `kjbfwjefgjwq` · `црфе ші` (this is English typed while the **Ukrainian keyboard
  layout** was still on)
- **What the bot replies:** *"I couldn't read that. It may have been typed with a different
  keyboard layout. Please retype your question."* *No search.*

---

## How the bot decides (the rules)

- It uses a quick **AI check** (not keyword matching), so it understands *meaning* — it can tell
  an off-topic question from a real policy one, and it can recognise wrong-keyboard-layout text.
- **When in doubt, it treats the message as a real policy question and searches.** This is
  deliberate and is the most important rule: the worst possible mistake would be to refuse a
  genuine compliance question, so the bot always errs toward searching. If its confidence is low,
  it searches.
- **If the AI check itself fails** (for any technical reason), the bot also defaults to searching.
  It never silently refuses a question.
- It treats **each message on its own** — it does not remember earlier messages, which is why the
  welcome asks you to put your whole question in one message.
- It only ever answers from **approved policy text**. It quotes policy; it does not give a
  personal opinion or interpretation. Judgment calls go to the Compliance team.

---

## When a backend is temporarily down

The bot depends on two background services: a **search database** and the **AI model**. If either
is briefly unreachable, the bot does **not** guess an answer and does **not** forward the question
to Compliance as if it were unanswerable. Instead it:

1. automatically **retries** for a few seconds, and
2. if the service is still unreachable, replies:
   **"⚠️ Policy service temporarily unavailable — I can't reach the policy database right now.
   Please try again in a moment."**

Examples:

- **Search database is down** — you ask *"Does the company sell my personal data?"*. The bot
  tries a few times, then replies with the temporary-unavailable message. Nothing is sent to
  Compliance; just ask again shortly.
- **AI model is down** — same outcome: the bot replies "temporarily unavailable" rather than
  erroring out or guessing.

These outages are also recorded, so the team can see exactly when a backend was unavailable.

---

## In short

| You send… | Bot does… |
|---|---|
| a greeting | shows the welcome — no search |
| a real policy question | searches and quotes the policy (or escalates if none applies) |
| an off-topic message | replies "I only answer company-policy questions" — no search |
| unreadable text | asks you to retype — no search |
| anything, while a backend is down | retries, then "temporarily unavailable — try again" |

When unsure, the bot always **searches** rather than refuse — and every decision is logged for audit.
