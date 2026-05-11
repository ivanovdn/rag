# Compliance Q&A Bot

An internal **Microsoft Teams bot** that answers employee compliance questions strictly from approved policy documents. If no policy applies, it escalates to the Compliance team with full context.

Everything runs on **self-hosted models** — no cloud APIs, no data leaves your infrastructure.

---

## What it does

Employees send 1:1 Teams messages like:

> *"Can I install software on my work laptop?"*
> *"What happens if I move to another country without informing HR?"*
> *"A journalist contacted me asking about an incident — can I respond?"*

The bot returns the **exact policy text** that answers the question, with **document name, section, and clause number** cited verbatim. After each answer, users rate quality (`-1` / `0` / `1` / `2`), and feedback is persisted.

Answers are **grounded** — the bot quotes policy text directly, never paraphrases, never invents rules. If retrieval finds nothing relevant, the bot escalates instead of guessing.

---

## How it works

```
Teams 1:1 message
       │
       ▼  poll Microsoft Graph every 5s
┌─────────────────────────────────────────────────────────┐
│  Teams bot (channels/teams/)                            │
│                                                         │
│   ┌────────────────────────────────────────────────┐    │
│   │  RAG pipeline (direct Python import)           │    │
│   │                                                │    │
│   │  1. embed query → Qdrant vector search         │    │
│   │  2. rerank candidates with Qwen3-Reranker      │    │
│   │  3. format top-N as [Source N] blocks          │    │
│   │  4. LlamaIndex agent (ReAct, 3 tools)          │    │
│   │  5. structured JSON: answer + citations[]      │    │
│   └────────────────────────────────────────────────┘    │
│                                                         │
│   render HTML → send reply → ask for rating             │
│   user rates → save (JSONL + SQLite)                    │
└─────────────────────────────────────────────────────────┘
```

### Grounding guarantees

- Agent **must** call `search_policies` before answering — no answers from general knowledge
- Citations must come from retrieved chunks (no hallucinations)
- Quote text is verbatim from policy documents
- If retrieval returns `NO_RELEVANT_POLICY_FOUND` → escalation, not a guessed answer
- `temperature=0.0` everywhere — deterministic outputs

### Feedback loop

After each answer the bot asks: *"Was this helpful? Reply -1 (should have been escalated), 0 (wrong), 1 (partially), or 2 (correct)"*

Detection is strict — only `"-1"`, `"0"`, `"1"`, `"2"` exactly. Anything else (typos, sentences) is treated as a new question. Ratings go to both JSONL (append-only) and SQLite (indexed by rating + timestamp).

---

## Stack

| Layer | Tech | Notes |
|---|---|---|
| LLM | Ollama **or** llama-server / vLLM (OpenAI-compatible) | Switchable via `LLM_BACKEND` |
| Embedding | HuggingFace (`nemotron` 2048d) or Ollama (`embeddinggemma` 768d, `qwen3-embedding` 4096d) | `EMBEDDING_SOURCE` flag |
| Vector store | Qdrant | Local or remote |
| Search | Vector search; optional BM25 + Reciprocal Rank Fusion | |
| Reranker | Qwen3-Reranker-4B | llama-server `/v1/rerank` (local) or vLLM `/v1/score` (remote), with Qwen3 chat-template wrapping for 0.99 vs 0.0003 score discrimination |
| Agent | LlamaIndex `AgentWorkflow` — 3 tools (search / get_section / escalate), structured JSON output | |
| Document parsing | `python-docx` + custom `NumberingResolver` | Resolves Word auto-numbering across multiple `numId` groups |
| Observability | Arize Phoenix | Traces every LLM / tool / retrieval / rerank call |
| Channels | Microsoft Teams (Graph API polling) | Direct Python import — no HTTP layer between bot and pipeline |
| Feedback | JSONL + SQLite | -1 / 0 / 1 / 2 ratings, indexed |
| Deployment | Docker Compose | Local: Qdrant + Phoenix. Remote: bot + Phoenix, models on a separate host |

### Document ingestion

DOCX parsing is **structure-aware** — chunks never split by token count. The parser:
- Detects Heading 1/2/3 styles for section hierarchy
- Resolves Word's auto-numbering (the "4.7." prefix is in `numbering.xml`, not in `para.text`) — including continuing counters across multiple `numId` groups, which is how Word renders sequential sections
- Extracts bold clause labels (`Software Installation` from `**Software Installation:** ...`)
- Converts tables to `"Header: Value | Header: Value"` rows
- Keeps chunks between 50 and 400 tokens, splits at sentence boundaries when needed

Current corpus: **52 documents, ~1602 chunks**.

---

## Docs

- **[SETUP.md](SETUP.md)** — step-by-step setup, configuration reference, troubleshooting
- **[CLAUDE.md](CLAUDE.md)** — architecture deep-dive, design decisions, common pitfalls
