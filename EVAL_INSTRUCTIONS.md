# Evaluation System — Phoenix Datasets & Experiments

## Overview

We use **Arize Phoenix** for all evaluation. The previous approach (raw JSON + local `run_eval.py` with span logging) has been **replaced** by Phoenix Datasets + Experiments API.

**What changed:**
- Test cases live in **Phoenix Datasets** (not just local JSON files)
- Evaluation runs are **Phoenix Experiments** — grouped runs with aggregate metrics, visible in the Phoenix dashboard
- All evaluator functions are **shared** across tiers via `eval/evaluators.py`
- Agent tool calls are **instrumented** — we capture the agent's actual search queries and retrieved chunks, not a separate standalone search
- Results are tracked over time — each experiment is a named run you can compare against previous runs

**What didn't change:**
- Source of truth for test cases is still XLSX → JSON (via `scripts/convert_eval_xlsx.py`)
- JSON files still live in `eval/datasets/`
- The evaluator logic (hit matching, answer coverage, MRR) is the same

## Architecture

```
eval/datasets/*.json          ← Source of truth (from XLSX)
        ↓
scripts/make_dataset.py       ← Uploads to Phoenix as a Dataset
        ↓
Phoenix Dataset               ← Stores examples (input/output/metadata)
        ↓
eval/run_experiment.py         ← Runs task + evaluators against dataset
        ↓
Phoenix Experiment             ← Stores results, metrics, traces
```

## File Structure

```
eval/
├── __init__.py
├── evaluators.py          # All evaluator functions, grouped by tier
├── agent_wrapper.py       # Instrumented agent (logs tool calls)
├── run_experiment.py      # CLI: runs experiments for any tier
└── datasets/
    ├── retrieval_test.json      # Tier 1: 70 retrieval test cases
    ├── e2e_test.json            # Tier 2: 25 end-to-end test cases
    ├── escalation_test.json     # Tier 3: escalation test cases
    └── chatbot_test_cases.json  # Tier 4: 61 chatbot Q&A test cases

scripts/
├── convert_eval_xlsx.py   # XLSX → JSON converter (4 sheets → 4 JSONs)
└── make_dataset.py        # JSON → Phoenix Dataset uploader
```

## Evaluation Tiers

### Tier 1 — Retrieval Quality
**What it tests:** Does `hybrid_search(question)` return the correct policy chunk?
**LLM involved:** No — pure search evaluation, fast.
**Dataset:** `retrieval-test-v1` (70 examples)
**Task:** Calls `hybrid_search(question, top_k=6)`, returns raw search results.

**Evaluators:**
| Evaluator | What it measures |
|---|---|
| `hit_evaluator` | Did ANY result match expected doc+section+clause? (1.0 or 0.0) |
| `mrr_evaluator` | 1/rank of first match (1.0=rank1, 0.5=rank2, 0.33=rank3...) |
| `retrieval_doc_hit` | At least the right document? (loose check) |
| `retrieval_section_hit` | Right document + section? |

**Run:**
```bash
python eval/run_experiment.py --tier tier1 --name baseline-hybrid-v1
```

### Tier 2 — End-to-End Agent
**What it tests:** Does the full agent pipeline produce correct answers with proper citations?
**LLM involved:** Yes — calls Ollama for each question. Slow (~15-30s per question).
**Dataset:** `e2e-test-v1` (25 examples)
**Task:** Builds a fresh instrumented agent per question. Captures:
  - Agent's actual search queries (may differ from user's question)
  - Chunks returned by each search call
  - get_section calls and whether they succeeded
  - Whether the agent escalated

**Evaluators:**
| Evaluator | Layer | What it measures |
|---|---|---|
| `hit_evaluator` | Retrieval | Did agent's searches return the right chunk? |
| `mrr_evaluator` | Retrieval | At what rank? |
| `retrieval_doc_hit` | Retrieval | At least the right document? |
| `retrieval_section_hit` | Retrieval | Right doc + section? |
| `answer_coverage` | Generation | Fraction of expected answer items found in response |
| `citation_in_answer` | Generation | Did the LLM mention the expected doc/section name? |
| `agent_search_count` | Behavior | How many searches? (0 = hallucinated, 1-3 = good, 4+ = thrashing) |
| `agent_used_get_section` | Behavior | Did agent fetch full section for precise citation? |

**Run:**
```bash
python eval/run_experiment.py --tier tier2 --name baseline-e2e-v1
```

### Chatbot (Tier 4) — Realistic User Questions
**What it tests:** Same as Tier 2 but with realistic, conversational user questions.
**LLM involved:** Yes.
**Dataset:** `chatbot-test-v1` (61 examples)
**Evaluators:** Same as Tier 2.

**Run:**
```bash
python eval/run_experiment.py --tier chatbot --name baseline-chatbot-v1
```

### Tier 3 — Escalation (TODO)
**What it tests:** Does the bot refuse to answer out-of-scope questions?
**Status:** Dataset structure defined, evaluators not yet implemented.

## Evaluator Design

### Unified evaluators across tiers
All retrieval evaluators (`hit_evaluator`, `mrr_evaluator`, etc.) work identically for Tier 1, 2, and Chatbot. The `_extract_expected()` helper normalizes the different expected formats:

- **Tier 1** expected: `{"expected_doc": "...", "expected_section": "...", "expected_clause": "..."}`
- **Tier 2 / Chatbot** expected: `{"expected_citations": [{"doc_id": "...", "section": "...", "clause": "..."}]}`

This means you can compare retrieval metrics directly across tiers.

### Retrieval source differs by tier
- **Tier 1:** `output["search_results"]` comes from `hybrid_search(user_question)` — a single controlled search with the raw user query.
- **Tier 2 / Chatbot:** `output["search_results"]` comes from the agent's actual `search_policies` tool calls — the agent may reformulate queries and search multiple times. Results are deduplicated across all searches.

This is intentional. Tier 1 measures search quality. Tier 2 measures the agent's search strategy.

### answer_coverage scoring
For each expected answer item, checks if ≥50% of "significant" words (longer than 3 characters) appear anywhere in the agent's response. This is a fuzzy word-overlap check, not exact string matching. Handles both list (Tier 2) and string (Chatbot) expected answers.

## Agent Instrumentation

The agent's 4 tools are wrapped in `eval/agent_wrapper.py`:

```
search_policies  → logged_search_policies  (logs query + raw hybrid_search results)
get_section      → logged_get_section      (logs doc_id, section, found?)
escalate         → logged_escalate         (logs reason)
ask_clarification → logged_clarify         (logs question)
```

**Critical:** Tool `name=` parameter MUST match the system prompt references:
```python
FunctionTool.from_defaults(fn=logged_search_policies, name="search_policies")
```
If names don't match, the LLM won't recognize the tools and will hallucinate answers without searching.

**Fresh agent per question:** `build_instrumented_agent()` is called for every question to avoid state/memory leakage between evaluation runs.

## Workflow

### First time setup
```bash
# 1. Convert XLSX to JSON (if updated)
python scripts/convert_eval_xlsx.py eval_dataset.xlsx

# 2. Upload to Phoenix
python scripts/make_dataset.py eval/datasets/retrieval_test.json
python scripts/make_dataset.py eval/datasets/e2e_test.json
python scripts/make_dataset.py eval/datasets/chatbot_test_cases.json

# 3. Run baseline experiments
python eval/run_experiment.py --tier tier1 --name baseline-hybrid-v1
python eval/run_experiment.py --tier tier2 --name baseline-e2e-v1
python eval/run_experiment.py --tier chatbot --name baseline-chatbot-v1
```

### After making changes (e.g., new embedding model, re-ranker, prompt change)
```bash
# Just re-run with a new experiment name
python eval/run_experiment.py --tier tier1 --name reranker-cohere-v1
python eval/run_experiment.py --tier tier2 --name prompt-v2-e2e

# Phoenix shows all experiments side-by-side for comparison
```

### After updating test cases
```bash
# 1. Re-convert XLSX
python scripts/convert_eval_xlsx.py eval_dataset_v2.xlsx

# 2. Re-upload (overwrite)
python scripts/make_dataset.py eval/datasets/retrieval_test.json --overwrite
python scripts/make_dataset.py eval/datasets/e2e_test.json --overwrite

# 3. Re-run experiments
python eval/run_experiment.py --tier tier1 --name baseline-hybrid-v2
```

## Phoenix Dataset Mapping

| Phoenix field | Tier 1 | Tier 2 / Chatbot | Tier 3 |
|---|---|---|---|
| `input` | `{"question": "..."}` | `{"question": "..."}` | `{"question": "..."}` |
| `output` | `{"expected_doc": "...", "expected_section": "...", "expected_clause": "..."}` | `{"expected_answer": [...], "expected_citations": [...]}` | `{"should_escalate": true, "reason": "..."}` |
| `metadata` | `{"test_id": "RET-001", "tier": "retrieval"}` | `{"test_id": "E2E-001", "tier": "e2e"}` | `{"test_id": "ESC-001", "tier": "escalation"}` |

## Known Issues

### Agent skipping tools (hallucination)
Qwen2.5-14B sometimes skips tool calls entirely and generates a hallucinated answer (often in Thai/non-English). The `agent_search_count` evaluator catches this (score=0 when num_searches=0). Mitigations:
- Stronger system prompt enforcement ("NEVER answer without searching first")
- Larger model (Qwen2.5-32B or 72B)
- Post-generation validation step that rejects answers without tool calls

### nest_asyncio requirement
The LlamaIndex AgentWorkflow is async. Phoenix `run_experiment` is sync. We bridge this with `nest_asyncio.apply()`. This must be called before any async code runs. The `run_experiment.py` script handles this automatically.

### Event loop conflicts in Jupyter
If running in Jupyter, put `import nest_asyncio; nest_asyncio.apply()` in the first cell after kernel restart. The CLI script handles this automatically.

## Baseline Results (as of 2026-03-24)

### Tier 1 — Retrieval
| Metric | Score |
|---|---|
| hit_evaluator | 0.81 |
| mrr_evaluator | 0.67 |
| retrieval_doc_hit | 0.95 |
| retrieval_section_hit | 0.81 |

### Tier 2 — E2E Agent
| Metric | Score |
|---|---|
| answer_coverage | 0.88 |
| citation_in_answer | 0.96 |
| no_hallucination_check | 1.00 |

(Tier 2 retrieval metrics from instrumented agent pending re-run.)

## Adding New Evaluators

1. Add the function to `eval/evaluators.py`
2. Add it to the appropriate `*_EVALUATORS` list
3. Re-run the experiment — Phoenix will show the new metric column automatically

Evaluator function signature:
```python
def my_evaluator(output: dict, expected: dict) -> dict:
    """Must return {"score": float, "label": str, "explanation": str}."""
    ...
```
