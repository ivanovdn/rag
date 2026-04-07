# Reranker Integration — Implementation Guide

## Overview

Add a reranking stage to the retrieval pipeline using **Qwen3-Reranker-0.6B** running locally via **llama-server** (llama.cpp). The reranker rescores vector search candidates to improve ranking quality before passing results to the agent.

**Current pipeline:**
```
Query → Vector Search (top-k) → Agent
```

**Target pipeline:**
```
Query → Vector Search (top 20) → Reranker (scores all 20) → Top-k → Agent
```

---

## Decision Log

| Decision | Choice | Reason |
|----------|--------|--------|
| Reranker model | Qwen3-Reranker-0.6B | Best MTEB-R score (65.80) at 0.6B size, instruction-aware, outperforms BGE-reranker-v2-m3 |
| Quantization | Q8_0 (609 MB) | Minimal quality loss vs F16, fast on M4 Pro |
| Runtime | llama-server (llama.cpp) | Faster than Transformers on Apple Silicon, lighter memory, clean HTTP API |
| GGUF source | `Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp` | Only correctly converted GGUF — includes `cls.output.weight` classifier tensor. Other community GGUFs produce garbage scores. |
| Ollama | NOT usable for reranking | No `/api/rerank` endpoint, no logprob extraction — confirmed as of early 2026 |
| Custom instruction | Yes, prepend to query | Tested: boosts top-result confidence from 0.87 → 0.98 |

---

## Prerequisites

### Install llama-server

```bash
# macOS via Homebrew
brew install llama.cpp
```

### Download the model (automatic on first run)

```bash
llama-server -hf Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp:Q8_0 \
  --reranking --pooling rank --embedding --port 8081
```

This downloads the GGUF to `~/Library/Caches/llama.cpp/` (~609 MB, one-time).

---

## Step 1 — Add Configuration

### Add to `.env`

```bash
# Reranker (llama-server)
RERANKER_ENABLED=true
RERANKER_URL=http://localhost:8081
RERANKER_TOP_N=6
RERANKER_CANDIDATES=20
RERANKER_INSTRUCTION=Given an employee compliance question, retrieve the internal policy clause that answers it
```

### Add to `config.py`

Add these fields to the `Settings` class:

```python
# Reranker
reranker_enabled: bool = True
reranker_url: str = "http://localhost:8081"
reranker_top_n: int = 6
reranker_candidates: int = 20
reranker_instruction: str = "Given an employee compliance question, retrieve the internal policy clause that answers it"
```

---

## Step 2 — Create Reranker Module

### Create `rag/reranker.py`

This module:
- Calls llama-server's `/v1/rerank` endpoint
- Prepends the custom instruction to every query
- Returns results sorted by relevance score
- Falls back to original ranking if reranker is unavailable

**API contract** — llama-server `/v1/rerank` request:
```json
{
  "query": "<Instruct>: {instruction}\n<Query>: {user_question}",
  "documents": ["chunk text 1", "chunk text 2", ...],
  "top_n": 6
}
```

**API response:**
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.978},
    {"index": 2, "relevance_score": 0.079},
    {"index": 1, "relevance_score": 0.0002}
  ]
}
```

- `index` refers to position in the input `documents` array
- `relevance_score` is 0.0–1.0 (probability that the document is relevant)
- Results are returned sorted by score descending

**Key implementation details:**
- Build the query as: `f"<Instruct>: {instruction}\n<Query>: {user_question}"`
- Use `httpx` (already in requirements) with a short timeout (5s)
- If reranker call fails (server down, timeout), log warning and return candidates in original order — never block the pipeline
- Preserve all original metadata from the search results; only reorder them based on reranker scores
- Add a `rerank_score` field to each result alongside the existing `vector_score`

---

## Step 3 — Integrate into Search Pipeline

### Modify `rag/hybrid_search.py` (or `rag/vector_store.py`)

The reranker sits AFTER vector search, BEFORE returning results to the agent.

**Current flow in search:**
```python
results = vector_search(query, top_k=settings.retrieval_top_k)
# apply confidence threshold
# return results
```

**New flow:**
```python
if settings.reranker_enabled:
    # Step 1: Get more candidates than final top_k
    candidates = vector_search(query, top_k=settings.reranker_candidates)  # 20
    # Step 2: Rerank all candidates
    results = rerank(query, candidates, top_n=settings.reranker_top_n)  # best 6
else:
    # Fallback: original behavior
    results = vector_search(query, top_k=settings.retrieval_top_k)
```

**Important changes when reranker is enabled:**
- Vector search retrieves `reranker_candidates` (20) instead of `retrieval_top_k` (6)
- The confidence threshold (`MIN_CONFIDENCE_SCORE=0.45`) should NOT be applied before the reranker — it would filter out candidates the reranker might score highly
- After reranking, the reranker score becomes the primary relevance signal
- The `NO_RELEVANT_POLICY_FOUND` logic should check if the top reranker score is below a threshold (to be determined empirically via eval — start with no threshold and add one later if needed)

---

## Step 4 — Add Observability

### Add Phoenix tracing to reranker calls

In `rag/reranker.py`, wrap the rerank call in a custom span (similar to existing `hybrid_search` spans):

```python
# Use the tracer from rag/observability.py
# Span should capture:
#   - query (with instruction)
#   - number of candidates in
#   - number of results out
#   - top rerank score
#   - latency
#   - any errors
```

This ensures reranker performance is visible in Phoenix UI at http://localhost:6006.

---

## Step 5 — Update Evaluation

### No changes to eval datasets or evaluators needed

The eval system measures retrieval quality (hit rate, MRR, citation accuracy). The reranker changes the ranking but not the data format — evaluators still compare against the same expected doc/section/clause.

### Run comparative experiments

```bash
# Before reranker (baseline already exists)
# python eval/run_experiment.py --tier tier1 --name baseline-vector-only

# After reranker integration
python eval/run_experiment.py --tier tier1 --name reranker-qwen3-0.6b-v1
python eval/run_experiment.py --tier tier2 --name reranker-e2e-v1
python eval/run_experiment.py --tier chatbot --name reranker-chatbot-v1
```

### Key metrics to watch

| Metric | Expect improvement? | Why |
|--------|-------------------|-----|
| `hit_evaluator` (Hit Rate) | Maybe small | Reranker reorders but doesn't find new chunks |
| `mrr_evaluator` (MRR) | **Yes, main target** | Pushes best match to rank 1 — current MRR 0.67 has room to grow |
| `answer_coverage` | Maybe | Better-ranked chunks → agent sees better context first |
| `citation_doc_accuracy` | Stable | Same docs, just reordered |

---

## Step 6 — Starting the Reranker Server

### Manual start (development)

```bash
llama-server -hf Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp:Q8_0 \
  --reranking --pooling rank --embedding --port 8081
```

### Add to `docker-compose.yml` (optional, for consistency)

The reranker runs via llama-server natively (not Docker), but you could document the start command alongside Docker services. Alternatively, add a script:

### Create `scripts/start_reranker.sh`

```bash
#!/bin/bash
# Start the Qwen3 Reranker via llama-server
# Model is cached at ~/Library/Caches/llama.cpp/ after first download

llama-server \
  -hf Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp:Q8_0 \
  --reranking \
  --pooling rank \
  --embedding \
  --port 8081
```

```bash
chmod +x scripts/start_reranker.sh
```

---

## File Changes Summary

| File | Action | What |
|------|--------|------|
| `.env` / `.env.example` | Modify | Add `RERANKER_*` variables |
| `config.py` | Modify | Add reranker settings to `Settings` class |
| `rag/reranker.py` | **Create** | Reranker module — calls llama-server `/v1/rerank` |
| `rag/hybrid_search.py` or `rag/vector_store.py` | Modify | Insert reranker step after vector search |
| `rag/observability.py` | Modify (minor) | Ensure tracer is available for reranker spans |
| `scripts/start_reranker.sh` | **Create** | Convenience script to start llama-server |
| `SETUP.md` | Modify | Add reranker setup instructions |
| `CLAUDE.md` | Modify | Document reranker in architecture and config sections |

---

## Testing Checklist

1. **llama-server is running:** `curl http://localhost:8081/health` returns OK
2. **Rerank endpoint works:**
   ```bash
   curl -s http://localhost:8081/v1/rerank \
     -H "Content-Type: application/json" \
     -d '{
       "query": "<Instruct>: Given an employee compliance question, retrieve the internal policy clause that answers it\n<Query>: What is the policy on software installation?",
       "documents": [
         "Team Members are forbidden to install any software on corporate workstations without prior approval.",
         "Annual leave must be requested 14 days in advance."
       ]
     }' | python3 -m json.tool
   ```
   Expected: first doc scores ~0.97, second doc scores ~0.0002
3. **Fallback works:** Stop llama-server, run a query — should still work with vector-only results (with warning log)
4. **Agent still returns valid JSON:** Run `python scripts/test_query.py -q "What is the policy on software installation?"` and verify `ComplianceAnswer` schema
5. **Eval passes:** Run Tier 1 experiment, compare MRR against baseline
6. **Phoenix traces:** Open http://localhost:6006, verify reranker spans appear in query traces

---

## Constraints

- **No external API calls** — llama-server runs locally on localhost:8081
- **temperature=0.0 on agent LLM** — unchanged, reranker is a separate model
- **Reranker must never block the pipeline** — if server is down, fall back to vector-only
- **Do not remove the confidence threshold logic** — just skip it when reranker is enabled (it's still used in vector-only fallback mode)
