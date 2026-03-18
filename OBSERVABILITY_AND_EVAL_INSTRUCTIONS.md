# TASK: Integrate Observability & Evaluation Harness into Compliance Q&A Bot

## Context

This is a compliance Q&A bot built with:
- **LlamaIndex** `AgentWorkflow` (ReAct agent with 4 tools)
- **Ollama** `qwen2.5:14b` for LLM, `nomic-embed-text` for embeddings
- **Qdrant** vector store (Docker, port 6333)
- **Python**, pydantic-settings config, project root: `compliance-bot/`

The existing project structure:
```
compliance-bot/
├── config.py                      # pydantic-settings, singleton via @lru_cache
├── ingest/
│   ├── chunk_models.py            # PolicyChunk pydantic model
│   ├── docx_parser.py            # DOCX → PolicyChunk list
│   └── pipeline.py               # parse → embed → upsert orchestrator
├── rag/
│   ├── embeddings.py             # Ollama nomic-embed-text wrapper
│   ├── vector_store.py           # Qdrant client + operations
│   ├── bm25_index.py             # BM25 keyword index
│   ├── hybrid_search.py          # RRF fusion (vector + BM25)
│   ├── agent.py                  # ReAct agent + system prompt
│   └── tools/
│       ├── search_policies.py    # Tool 1: semantic/hybrid search
│       ├── get_section.py        # Tool 2: exact clause fetch
│       ├── clarify.py            # Tool 3: ask user for clarity
│       └── escalate.py           # Tool 4: escalate to compliance
├── scripts/
│   ├── ingest_all.py
│   └── test_query.py
├── docker-compose.yml            # Qdrant container
├── .env
└── requirements.txt
```

You need to add:
1. Arize Phoenix for observability (tracing every query through the full pipeline)
2. Evaluation harness with 4 tiers (retrieval, e2e, escalation, chatbot positive/negative)
3. XLSX → JSON converter scripts for evaluation datasets
4. CLI script to run all evaluations and log results to Phoenix

---

## PART 1: Add Phoenix to Docker & Dependencies

### 1.1 Update docker-compose.yml

Add a Phoenix service alongside the existing Qdrant service:

```yaml
services:
  qdrant:
    # ... existing qdrant config stays unchanged ...

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "${PHOENIX_PORT:-6006}:6006"
    volumes:
      - phoenix_data:/data
    environment:
      - PHOENIX_WORKING_DIR=/data
    restart: unless-stopped

volumes:
  qdrant_data:    # existing
  phoenix_data:   # NEW — persists all traces, experiments, eval results
```

IMPORTANT: The `phoenix_data` Docker volume ensures all observability data (traces, experiments, evaluation scores) persists across container restarts, rebuilds, and even `docker-compose down` (volumes survive unless you run `docker-compose down -v`).

### 1.2 Update requirements.txt

Add these packages:

```
arize-phoenix>=8.0
openinference-instrumentation-llama-index>=3.0
opentelemetry-api>=1.0
opentelemetry-sdk>=1.0
phoenix-client>=1.0
openpyxl>=3.1
```

Do NOT add `openinference-instrumentation-openai` — we're using Ollama via LlamaIndex, not OpenAI directly. The LlamaIndex instrumentor handles everything.

### 1.3 Update config.py

Add these fields to the `Settings` class:

```python
    # Observability (Phoenix)
    phoenix_enabled: bool = True
    phoenix_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "compliance-bot"

    # Evaluation
    eval_dataset_path: str = "eval/datasets"
    eval_confidence_threshold: float = 0.45  # should match min_confidence_score
```

Add to `.env`:

```
PHOENIX_ENABLED=true
PHOENIX_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=compliance-bot
EVAL_DATASET_PATH=eval/datasets
```

---

## PART 2: Instrument the Application with Phoenix Tracing

### 2.1 Create `rag/observability.py`

This module initializes Phoenix tracing. It MUST be called once at application startup, BEFORE any LlamaIndex calls.

```python
"""
Phoenix observability integration.

Initializes OpenTelemetry tracing to Phoenix for the entire application.
Must be called once at startup before any LlamaIndex or Ollama calls.

What gets traced automatically (via LlamaIndex instrumentor):
- Every agent ReAct iteration (Thought → Action → Observation)
- Every tool call (search_policies, get_section, clarify, escalate)
- Every LLM generation (prompt, response, tokens, latency)
- Every embedding call

What we trace manually (via custom spans):
- Hybrid search breakdown (vector score, BM25 score, RRF fusion)
- Retrieval confidence gate decisions
- Escalation events
- Citation validation results
"""

import logging
from config import settings

logger = logging.getLogger(__name__)

_initialized = False


def init_observability() -> None:
    """
    Initialize Phoenix tracing. Safe to call multiple times (idempotent).
    
    Call this at the top of:
    - scripts/test_query.py
    - scripts/run_eval.py
    - Your FastAPI app startup (when you build it)
    """
    global _initialized
    if _initialized:
        return
    
    if not settings.phoenix_enabled:
        logger.info("Phoenix observability disabled (PHOENIX_ENABLED=false)")
        _initialized = True
        return
    
    try:
        import phoenix as px
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        # Connect to Phoenix server
        px.register(
            endpoint=settings.phoenix_endpoint,
            project_name=settings.phoenix_project_name,
        )
        
        # Auto-instrument all LlamaIndex calls
        LlamaIndexInstrumentor().instrument()
        
        logger.info(
            f"Phoenix observability initialized: "
            f"endpoint={settings.phoenix_endpoint}, "
            f"project={settings.phoenix_project_name}"
        )
        _initialized = True
        
    except ImportError:
        logger.warning(
            "Phoenix packages not installed. Run: "
            "pip install arize-phoenix openinference-instrumentation-llama-index"
        )
        _initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize Phoenix: {e}. Continuing without observability.")
        _initialized = True


def get_tracer():
    """
    Get an OpenTelemetry tracer for manual span creation.
    
    Usage:
        tracer = get_tracer()
        with tracer.start_as_current_span("hybrid_search") as span:
            span.set_attribute("query", query)
            span.set_attribute("vector_top_score", 0.87)
    """
    if not settings.phoenix_enabled:
        from opentelemetry import trace
        return trace.get_tracer("noop")
    
    from opentelemetry import trace
    return trace.get_tracer("compliance-bot")
```

### 2.2 Add custom tracing to `rag/hybrid_search.py`

In the `hybrid_search()` function, wrap the search logic with a custom span:

```python
from rag.observability import get_tracer

def hybrid_search(query: str, top_k: int = 6, ...) -> list[HybridResult]:
    tracer = get_tracer()
    with tracer.start_as_current_span("hybrid_search") as span:
        span.set_attribute("input.query", query)
        span.set_attribute("input.top_k", top_k)
        
        # ... existing vector search code ...
        span.set_attribute("retrieval.vector_count", len(vector_results))
        span.set_attribute("retrieval.vector_top_score", 
                          vector_results[0].score if vector_results else 0.0)
        
        # ... existing BM25 search code ...
        span.set_attribute("retrieval.bm25_count", len(bm25_results))
        span.set_attribute("retrieval.bm25_top_score",
                          bm25_results[0][1] if bm25_results else 0.0)
        
        # ... existing RRF fusion code ...
        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        
        span.set_attribute("retrieval.fused_count", len(fused))
        span.set_attribute("retrieval.top_result_doc", 
                          fused[0].doc_title if fused else "none")
        span.set_attribute("retrieval.top_result_clause",
                          fused[0].clause_number if fused else "none")
        
        return fused[:top_k]
```

### 2.3 Add custom tracing to `rag/tools/escalate.py`

```python
from rag.observability import get_tracer

def escalate_to_compliance(reason: str, unanswered_question: str, ...) -> str:
    tracer = get_tracer()
    with tracer.start_as_current_span("escalation") as span:
        span.set_attribute("escalation.reason", reason)
        span.set_attribute("escalation.question", unanswered_question)
        span.set_attribute("escalation.search_attempted", search_attempted)
        
        # ... existing escalation logic ...
        
        span.set_attribute("escalation.ticket_id", ticket_id)
        return result
```

### 2.4 Update `scripts/test_query.py`

Add observability initialization at the very top, BEFORE any LlamaIndex imports:

```python
# THIS MUST BE THE FIRST IMPORT after standard library
from rag.observability import init_observability
init_observability()

# Then your existing imports
from rag.agent import build_agent
# ... rest of the script ...
```

---

## PART 3: Evaluation Dataset Structure

### 3.1 Create directory structure

```
compliance-bot/
├── eval/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── retrieval_test.json      # Tier 1: retrieval-only tests
│   │   ├── e2e_test.json            # Tier 2: end-to-end Q&A tests
│   │   ├── escalation_test.json     # Tier 3: should-escalate tests
│   │   └── chatbot_test_cases.json  # Tier 4: positive/negative paired tests
│   └── results/                     # auto-generated eval run outputs
```

### 3.2 JSON Formats

The evaluation JSONs are generated from XLSX files using converter scripts (see Part 5).
Here are the exact formats the evaluation harness expects.

#### Tier 1 — Retrieval (`retrieval_test.json`)

```json
{
  "metadata": {
    "name": "Retrieval Quality Test Set",
    "version": "1.0",
    "total_cases": 50
  },
  "test_cases": [
    {
      "id": "RET-001",
      "question": "What is the gift reporting threshold?",
      "expected_doc_id": "conflicts-of-interest-policy",
      "expected_section_contains": "4.3",
      "expected_clause": "4.3.1",
      "expected_text_contains": "report any gift"
    }
  ]
}
```

IMPORTANT: `expected_text_contains` can be a string OR a list of strings.
When it is a list, the retrieval check should pass if ANY item from the list
appears in the retrieved chunk text. Example:

```json
{
  "expected_text_contains": [
    "developing and maintaining HIPAA policies",
    "overseeing the use of PHI",
    "providing HIPAA training to staff"
  ]
}
```

#### Tier 2 — E2E (`e2e_test.json`)

```json
{
  "metadata": {
    "name": "End-to-End Q&A Test Set",
    "version": "1.0",
    "total_cases": 30
  },
  "test_cases": [
    {
      "id": "E2E-001",
      "question": "What is the gift reporting threshold?",
      "expected_answer": "Employees must report any gift valued at $100 or more within 5 business days.",
      "expected_citations": [
        { "doc_id": "conflicts-of-interest-policy", "section": "4.3", "clause": "4.3.1" }
      ]
    }
  ]
}
```

IMPORTANT: `expected_answer` can be a string OR a list of strings.
When it is a list, the evaluation should check how many items from the list
appear in the bot's answer (fact coverage score). Example:

```json
{
  "expected_answer": [
    "developing and maintaining HIPAA policies",
    "overseeing the use of PHI",
    "providing HIPAA training to staff"
  ]
}
```

#### Tier 3 — Escalation (`escalation_test.json`)

```json
{
  "metadata": {
    "name": "Escalation Test Set",
    "version": "1.0",
    "total_cases": 20
  },
  "test_cases": [
    {
      "id": "ESC-001",
      "question": "Can we accept cryptocurrency payments from clients?",
      "reason": "No cryptocurrency policy exists",
      "category": "policy-gap",
      "should_escalate": true
    }
  ]
}
```

#### Tier 4 — Chatbot Positive/Negative (`chatbot_test_cases.json`)

```json
{
  "metadata": {
    "name": "Chatbot Test Cases (Positive & Negative)",
    "version": "1.0",
    "total_positive": 20,
    "total_negative": 20,
    "total_cases": 40
  },
  "positive_cases": [
    {
      "id": "TC-001-POS",
      "question": "My company laptop is being repaired. Can I use my personal laptop?",
      "expected_answer": "Team Members must use corporate-issued workstations for work activities.",
      "policy": "Acceptable Use Policy",
      "policy_section": "Corporate Workstation and Software Use",
      "policy_rule": "Team Members must use corporate-issued workstations for work.",
      "user_goal": "Use personal laptop for work",
      "type": "positive"
    }
  ],
  "negative_cases": [
    {
      "id": "TC-001-NEG",
      "question": "My company laptop is being repaired. Can I use my personal laptop?",
      "incorrect_answer": "Yes, you can use your personal laptop if it's just temporary.",
      "expected_behavior": "Bot must NOT give an answer similar to the negative example.",
      "policy": "Acceptable Use Policy",
      "policy_section": "Corporate Workstation and Software Use",
      "policy_rule": "Team Members must use corporate-issued workstations for work.",
      "user_goal": "Use personal laptop for work",
      "type": "negative"
    }
  ]
}
```

---

## PART 4: XLSX → JSON Converter Scripts

Create two converter scripts. Users fill in Excel spreadsheets, then run these to generate JSON.

### 4.1 Create `scripts/convert_eval_xlsx.py`

Converts the 3-tier evaluation XLSX into JSON datasets.

Expected XLSX structure (3 sheets, headers in row 1, data from row 2):
- Sheet 1 (Tier 1): ID | Question | Expected Document | Expected Section | Expected Clause | Key Phrase from Clause
- Sheet 2 (Tier 2): ID | Question | Expected Answer | Expected Document | Expected Section | Expected Clause
- Sheet 3 (Tier 3): ID | Question | Why It Should Escalate | Category | Should Escalate

CRITICAL RULES:
- Skip rows where column A value is "ID" (header row read as data)
- Skip rows where column B value is "Question" (header row read as data)
- Skip rows where column B (Question) is empty
- The pipe character `|` is used as a list separator in cells
- If a cell contains `|` pipes, parse it into a JSON array of strings
- If no pipes, keep it as a single string
- This applies to: `expected_text_contains` (Tier 1), `expected_answer` (Tier 2)
- The script auto-detects sheets by order (first 3 non-"Instructions" sheets)

```python
"""
Convert evaluation XLSX (3 sheets) into JSON datasets.

Usage:
    python scripts/convert_eval_xlsx.py eval_dataset.xlsx

Output:
    eval/datasets/retrieval_test.json
    eval/datasets/e2e_test.json
    eval/datasets/escalation_test.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

try:
    from openpyxl import load_workbook
except ImportError:
    print("ERROR: pip install openpyxl")
    sys.exit(1)


def cell_val(cell) -> str:
    if cell is None:
        return ""
    return str(cell).strip()


def parse_pipe_list(text: str):
    """If text contains | pipes -> list of strings. Otherwise -> string."""
    if not text:
        return ""
    if "|" in text:
        return [item.strip() for item in text.split("|") if item.strip()]
    return text


def read_rows(ws, start_row: int = 2) -> list[list[str]]:
    rows = []
    for row in ws.iter_rows(min_row=start_row, values_only=True):
        cells = [cell_val(c) for c in row]
        while len(cells) < 6:
            cells.append("")
        if not cells[1]:
            continue
        # Skip header rows read as data
        if cells[0].lower() == "id" or cells[1].lower() in ("question",):
            continue
        rows.append(cells)
    return rows


def convert_tier1(ws) -> dict:
    rows = read_rows(ws)
    test_cases = []
    for cells in rows:
        tc = {
            "id": cells[0] or f"RET-{len(test_cases)+1:03d}",
            "question": cells[1],
            "expected_doc_id": cells[2],
            "expected_section_contains": cells[3],
            "expected_clause": cells[4],
            "expected_text_contains": parse_pipe_list(cells[5]) if len(cells) > 5 else "",
        }
        test_cases.append(tc)
    return {
        "metadata": {
            "name": "Retrieval Quality Test Set",
            "version": "1.0",
            "description": "Tests whether the correct policy chunk is retrieved.",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_cases": len(test_cases),
        },
        "test_cases": test_cases,
    }


def convert_tier2(ws) -> dict:
    rows = read_rows(ws)
    test_cases = []
    for cells in rows:
        expected_answer = parse_pipe_list(cells[2])
        citations = []
        if cells[3]:
            cit = {"doc_id": cells[3]}
            if cells[4]:
                cit["section"] = cells[4]
            if cells[5]:
                cit["clause"] = cells[5]
            citations.append(cit)
        tc = {
            "id": cells[0] or f"E2E-{len(test_cases)+1:03d}",
            "question": cells[1],
            "expected_answer": expected_answer,
            "expected_citations": citations,
        }
        test_cases.append(tc)
    return {
        "metadata": {
            "name": "End-to-End Q&A Test Set",
            "version": "1.0",
            "description": "Tests full pipeline: retrieval + LLM generation.",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_cases": len(test_cases),
        },
        "test_cases": test_cases,
    }


def convert_tier3(ws) -> dict:
    rows = read_rows(ws)
    test_cases = []
    for cells in rows:
        tc = {
            "id": cells[0] or f"ESC-{len(test_cases)+1:03d}",
            "question": cells[1],
            "reason": cells[2],
            "category": cells[3] or "policy-gap",
            "should_escalate": str(cells[4]).upper() == "TRUE" if cells[4] else True,
        }
        test_cases.append(tc)
    return {
        "metadata": {
            "name": "Escalation Test Set",
            "version": "1.0",
            "description": "Questions the bot should NOT answer.",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_cases": len(test_cases),
        },
        "test_cases": test_cases,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_eval_xlsx.py <filled_template.xlsx>")
        sys.exit(1)
    xlsx_path = Path(sys.argv[1])
    if not xlsx_path.exists():
        print(f"ERROR: File not found: {xlsx_path}")
        sys.exit(1)
    output_dir = Path("eval/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    data_sheets = [s for s in wb.sheetnames if "instruct" not in s.lower()]
    converters = [
        ("retrieval_test.json", convert_tier1),
        ("e2e_test.json", convert_tier2),
        ("escalation_test.json", convert_tier3),
    ]
    for i, (output_name, converter) in enumerate(converters):
        if i >= len(data_sheets):
            print(f"WARNING: No sheet found for {output_name}, skipping")
            continue
        ws = wb[data_sheets[i]]
        data = converter(ws)
        count = data["metadata"]["total_cases"]
        output_path = output_dir / output_name
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  {output_name}: {count} test cases -> {output_path}")
    wb.close()
    # Print first entry from each file
    print()
    for fname in ["retrieval_test.json", "e2e_test.json", "escalation_test.json"]:
        fpath = output_dir / fname
        if fpath.exists():
            data = json.loads(fpath.read_text())
            cases = data.get("test_cases", [])
            if cases:
                print(f"--- {fname} (first entry) ---")
                print(json.dumps(cases[0], indent=2, ensure_ascii=False))
                print()


if __name__ == "__main__":
    main()
```

### 4.2 Create `scripts/convert_chatbot_xlsx.py`

Converts the chatbot positive/negative test cases XLSX into JSON.

Expected XLSX structure (1 sheet, headers can be in any row — auto-detected):
- Column A: Policy
- Column B: Policy Section Reference
- Column C: Policy Rule
- Column D: User Goal
- Column E: Example User Question
- Column F: ✅ Positive Examples (correct answer)
- Column G: ❌ Negative Examples (wrong answer)

CRITICAL RULES:
- Auto-skip header rows (detected by checking for keywords like "Policy Section Reference", "User Goal", etc.)
- Skip rows where column E (question) is empty
- Skip instruction/description rows (rows without a question)
- Each row produces TWO test cases: one positive (TC-XXX-POS) and one negative (TC-XXX-NEG)
- Clean up non-breaking spaces and collapse multiple whitespace

```python
"""
Convert chatbot test cases XLSX into evaluation JSON.

Usage:
    python scripts/convert_chatbot_xlsx.py chatbot_cases.xlsx
    python scripts/convert_chatbot_xlsx.py chatbot_cases.xlsx --sheet "Sheet1"

Output:
    eval/datasets/chatbot_test_cases.json
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

try:
    from openpyxl import load_workbook
except ImportError:
    print("ERROR: pip install openpyxl")
    sys.exit(1)


def cell_val(cell) -> str:
    if cell is None:
        return ""
    val = str(cell).strip()
    val = val.replace("\xa0", " ")
    val = re.sub(r"\s+", " ", val)
    return val


def is_header_row(cells: list[str]) -> bool:
    combined = " ".join(c.lower() for c in cells if c)
    header_markers = ["policy section reference", "user goal", "example user question",
                      "positive examples", "negative examples", "policy rule"]
    return any(marker in combined for marker in header_markers)


def read_rows(ws) -> list[list[str]]:
    rows = []
    for row in ws.iter_rows(min_row=1, values_only=True):
        cells = [cell_val(c) for c in row]
        while len(cells) < 7:
            cells.append("")
        # Skip empty rows (no question in col E)
        if not cells[4]:
            continue
        # Skip headers
        if is_header_row(cells):
            continue
        rows.append(cells)
    return rows


def convert(ws) -> dict:
    rows = read_rows(ws)
    positive_cases = []
    negative_cases = []

    for i, cells in enumerate(rows):
        policy = cells[0]
        section_ref = cells[1]
        policy_rule = cells[2]
        user_goal = cells[3]
        question = cells[4]
        positive_answer = cells[5]
        negative_answer = cells[6]
        base_id = f"TC-{i+1:03d}"

        if positive_answer:
            positive_cases.append({
                "id": f"{base_id}-POS",
                "question": question,
                "expected_answer": positive_answer,
                "policy": policy,
                "policy_section": section_ref,
                "policy_rule": policy_rule,
                "user_goal": user_goal,
                "type": "positive",
            })
        if negative_answer:
            negative_cases.append({
                "id": f"{base_id}-NEG",
                "question": question,
                "incorrect_answer": negative_answer,
                "expected_behavior": "Bot must NOT give an answer similar to the negative example.",
                "policy": policy,
                "policy_section": section_ref,
                "policy_rule": policy_rule,
                "user_goal": user_goal,
                "type": "negative",
            })

    return {
        "metadata": {
            "name": "Chatbot Test Cases (Positive & Negative)",
            "version": "1.0",
            "description": "Paired test cases: correct answer vs incorrect answer.",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_positive": len(positive_cases),
            "total_negative": len(negative_cases),
            "total_cases": len(positive_cases) + len(negative_cases),
        },
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_chatbot_xlsx.py <chatbot_cases.xlsx> [--sheet SheetName]")
        sys.exit(1)
    xlsx_path = Path(sys.argv[1])
    if not xlsx_path.exists():
        print(f"ERROR: File not found: {xlsx_path}")
        sys.exit(1)
    sheet_name = None
    if "--sheet" in sys.argv:
        idx = sys.argv.index("--sheet")
        if idx + 1 < len(sys.argv):
            sheet_name = sys.argv[idx + 1]
    output_dir = Path("eval/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name:
        ws = wb[sheet_name]
    else:
        ws = wb[wb.sheetnames[0]]
        print(f"Using sheet: '{wb.sheetnames[0]}'")
    data = convert(ws)
    wb.close()
    output_path = output_dir / "chatbot_test_cases.json"
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n  chatbot_test_cases.json:")
    print(f"    Positive cases: {data['metadata']['total_positive']}")
    print(f"    Negative cases: {data['metadata']['total_negative']}")
    print(f"    Saved to:       {output_path}")
    if data["positive_cases"]:
        print(f"\n--- Sample positive case ---")
        print(json.dumps(data["positive_cases"][0], indent=2, ensure_ascii=False))
    if data["negative_cases"]:
        print(f"\n--- Sample negative case ---")
        print(json.dumps(data["negative_cases"][0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

---

## PART 5: Evaluation Harness Script

### 5.1 Create `scripts/run_eval.py`

This is the main evaluation CLI. It runs test datasets against the retrieval pipeline and/or full agent, computes metrics, and logs everything to Phoenix.

```
Usage:
    python scripts/run_eval.py --tier retrieval  --tag "baseline"
    python scripts/run_eval.py --tier e2e        --tag "baseline"
    python scripts/run_eval.py --tier escalation --tag "baseline"
    python scripts/run_eval.py --tier chatbot    --tag "baseline"
    python scripts/run_eval.py --tier all        --tag "baseline"
```

The script must implement these 4 evaluation tiers:

#### Tier 1: Retrieval Evaluation (`run_retrieval_eval`)

- For each test case, run hybrid_search (or vector-only) with the question
- Check if the expected chunk appears in top-k results by matching:
  - `expected_doc_id` must match result's doc_id
  - `expected_clause` must match result's clause_number (if specified)
  - `expected_section_contains` must appear in result's section_display (if specified)
  - `expected_text_contains`: if it's a STRING, check if it appears in result text (case-insensitive). If it's a LIST, check if ANY item from the list appears in result text.
- Record: hit (bool), hit_rank, top_score
- Log each test as a Phoenix span with attributes: `eval.test_id`, `eval.question`, `eval.hit`, `eval.hit_rank`, `eval.top_score`
- Does NOT need the LLM — fast

Metrics to compute:
- `hit_rate_at_k`: fraction of queries where expected chunk was in top-k
- `mrr`: Mean Reciprocal Rank (average of 1/rank for hits)
- `avg_top_score`: average similarity score of the #1 result

#### Tier 2: End-to-End Evaluation (`run_e2e_eval`)

- For each test case, send question to the full agent
- Check citation accuracy: does the answer reference the expected doc_id and section?
- Check fact coverage: if `expected_answer` is a LIST, count how many items appear in the bot's answer. If it's a STRING, compute text similarity.
- Measure latency per query
- Log each test as a Phoenix span with attributes: `eval.test_id`, `eval.citation_correct`, `eval.fact_coverage`, `eval.latency_seconds`

Metrics:
- `citation_accuracy`: fraction with correct citations
- `fact_coverage`: average fraction of expected facts found
- `avg_latency_seconds`

#### Tier 3: Escalation Evaluation (`run_escalation_eval`)

- For each test case, send question to the full agent
- Check if the bot escalated (look for "escalat", "ESC-", "unable to find", "cannot confirm" in response)
- A bot that ANSWERS these questions is FAILING — it should escalate
- Log: `eval.was_escalated`, `eval.correctly_escalated`, `eval.false_answer`

Metrics:
- `correct_escalation_rate`: fraction that correctly escalated
- `false_answer_rate`: fraction that wrongly answered instead of escalating

#### Tier 4: Chatbot Evaluation (`run_chatbot_eval`)

- Group positive and negative cases by question text
- Run the agent ONCE per unique question
- Compare the bot's answer against BOTH the positive and negative examples:
  - Positive check: if `expected_answer` is a LIST, count items found in bot answer (score = items_found / total_items). If STRING, use SequenceMatcher similarity.
  - Negative check: compute SequenceMatcher similarity between bot answer and `incorrect_answer`
- Check policy citation: do key words from the policy name appear in the answer?
- Pass/fail logic: PASS if positive_score > negative_score AND negative_score < 0.4
- Log: `eval.positive_score`, `eval.negative_score`, `eval.passed`, `eval.policy_cited`

Metrics:
- `pass_rate`: fraction of questions that passed
- `positive_wins_rate`: fraction where positive_score > negative_score
- `negative_avoided_rate`: fraction where negative_score < 0.4
- `policy_citation_rate`: fraction where policy was cited
- `avg_positive_score`, `avg_negative_score`
- `policy_breakdown`: pass rate grouped by policy name

#### General requirements for all tiers:

- Initialize Phoenix observability at script start (before any LlamaIndex imports)
- Every test case gets its own Phoenix span with `eval.*` attributes
- Save results to `eval/results/{tier}_{tag}_{timestamp}.json` containing: metadata (tier, tag, timestamp, config snapshot), metrics dict, and full results list
- Print a formatted metrics summary to console
- The `--tag` argument is for labeling experiments (e.g., "hybrid-search-v1", "vector-only", "bge-base")
- Default dataset paths if `--dataset` is not specified:
  - retrieval → `eval/datasets/retrieval_test.json`
  - e2e → `eval/datasets/e2e_test.json`
  - escalation → `eval/datasets/escalation_test.json`
  - chatbot → `eval/datasets/chatbot_test_cases.json`
- The config snapshot in results should include: bm25_enabled, embedding_model, llm_model, min_confidence_score, retrieval_top_k

---

## PART 6: Verification Checklist

After implementing everything, verify:

### 6.1 Infrastructure
- [ ] `docker-compose up -d` starts both Qdrant AND Phoenix
- [ ] Phoenix UI accessible at http://localhost:6006
- [ ] Phoenix data persists across `docker-compose restart`

### 6.2 Tracing (no eval needed)
- [ ] Run `python scripts/test_query.py "What is the gift reporting threshold?"`
- [ ] Open Phoenix UI → see a trace with spans for: LLM call, tool calls, embeddings
- [ ] Custom spans appear for hybrid_search with vector/bm25 score attributes

### 6.3 XLSX Converters
- [ ] `python scripts/convert_eval_xlsx.py eval_tiers.xlsx` → creates 3 JSON files
- [ ] `python scripts/convert_chatbot_xlsx.py chatbot_cases.xlsx` → creates 1 JSON file
- [ ] No header rows in the JSON output
- [ ] Pipe-separated fields are parsed into arrays
- [ ] First entry of each file is printed and looks correct

### 6.4 Evaluation
- [ ] `python scripts/run_eval.py --tier retrieval --tag "test"` runs, prints metrics, saves JSON
- [ ] `python scripts/run_eval.py --tier chatbot --tag "test"` runs, prints metrics, saves JSON
- [ ] `python scripts/run_eval.py --tier all --tag "test"` runs all 4 tiers
- [ ] Results JSON saved to `eval/results/` with config snapshot
- [ ] Eval traces visible in Phoenix with `eval.*` attributes

### 6.5 Experiment Comparison
- [ ] Run eval with `--tag "baseline"`, note metrics
- [ ] Change config (disable BM25), run with `--tag "no-bm25"`
- [ ] Both results saved as separate JSON files in `eval/results/`
- [ ] Can compare metrics between the two runs

---

## PART 7: Complete File List

New files to CREATE:
1. `rag/observability.py` — Phoenix initialization + tracer helper (Part 2.1)
2. `eval/__init__.py` — empty file
3. `scripts/convert_eval_xlsx.py` — 3-tier XLSX → JSON converter (Part 4.1)
4. `scripts/convert_chatbot_xlsx.py` — chatbot XLSX → JSON converter (Part 4.2)
5. `scripts/run_eval.py` — evaluation harness CLI with 4 tiers (Part 5.1)

Existing files to MODIFY:
1. `docker-compose.yml` — add Phoenix service + volume (Part 1.1)
2. `requirements.txt` — add Phoenix + openpyxl packages (Part 1.2)
3. `config.py` — add phoenix_* and eval_* settings (Part 1.3)
4. `rag/hybrid_search.py` — add custom tracing spans (Part 2.2)
5. `rag/tools/escalate.py` — add custom tracing span (Part 2.3)
6. `scripts/test_query.py` — add `init_observability()` at top (Part 2.4)

Directories to CREATE:
1. `eval/` — evaluation module root
2. `eval/datasets/` — JSON test datasets (generated by converters)
3. `eval/results/` — evaluation run outputs (auto-generated)

---

## PART 8: How to Use

### Step 1: Convert XLSX files to JSON

```bash
# Convert 3-tier evaluation dataset
python scripts/convert_eval_xlsx.py eval_tiers.xlsx

# Convert chatbot positive/negative test cases
python scripts/convert_chatbot_xlsx.py chatbot_cases.xlsx
```

### Step 2: Start services

```bash
docker-compose up -d   # starts Qdrant + Phoenix
ollama serve           # starts LLM server
```

### Step 3: Quick smoke test (just tracing, no eval)

```bash
python scripts/test_query.py "What is the gift reporting threshold?"
# Open http://localhost:6006 — see the full trace
```

### Step 4: Run evaluations

```bash
# Fast — retrieval only, no LLM needed:
python scripts/run_eval.py --tier retrieval --tag "baseline"

# Full pipeline — needs Ollama running:
python scripts/run_eval.py --tier e2e --tag "baseline"
python scripts/run_eval.py --tier escalation --tag "baseline"
python scripts/run_eval.py --tier chatbot --tag "baseline"

# Or all at once:
python scripts/run_eval.py --tier all --tag "baseline"
```

### Step 5: Compare configurations

```bash
# 1. Baseline
python scripts/run_eval.py --tier retrieval --tag "hybrid-baseline"

# 2. Disable BM25 (edit .env: BM25_ENABLED=false)
python scripts/run_eval.py --tier retrieval --tag "vector-only"

# 3. Swap embedding model (edit config)
python scripts/run_eval.py --tier retrieval --tag "bge-base"

# Compare JSON results in eval/results/
# Compare traces in Phoenix UI at localhost:6006
```
