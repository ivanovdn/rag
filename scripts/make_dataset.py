#!/usr/bin/env python3
"""
Create a Phoenix dataset from a JSON evaluation file.

Supports all tiers:
  - Tier 1 (retrieval):  retrieval_test.json
  - Tier 2 (e2e):        e2e_test.json
  - Tier 3 (escalation): escalation_test.json
  - Tier 4 (chatbot):    chatbot_test_cases.json

Usage:
    python scripts/make_dataset.py eval/datasets/retrieval_test.json
    python scripts/make_dataset.py eval/datasets/e2e_test.json
    python scripts/make_dataset.py eval/datasets/chatbot_test_cases.json
    python scripts/make_dataset.py eval/datasets/escalation_test.json

    # Custom dataset name:
    python scripts/make_dataset.py eval/datasets/e2e_test.json --name my-custom-name

    # Overwrite existing dataset:
    python scripts/make_dataset.py eval/datasets/e2e_test.json --overwrite

Auto-detection rules:
  - Has "expected_doc_id"   → Tier 1 (retrieval)
  - Has "should_escalate"   → Tier 3 (escalation)
  - Has "expected_answer" + "expected_citations" → Tier 2/4 (e2e or chatbot)

Phoenix dataset mapping:
  ┌─────────────────┬────────────────────────────────────────────────┐
  │ Phoenix field    │ What goes in it                                │
  ├─────────────────┼────────────────────────────────────────────────┤
  │ input           │ {"question": "..."}                            │
  │ output          │ Ground truth (expected doc/section/answer)     │
  │ metadata        │ {"test_id": "RET-001", "tier": "retrieval"}   │
  └─────────────────┴────────────────────────────────────────────────┘
"""

import argparse
import json
import sys
from pathlib import Path


def detect_tier(test_cases: list[dict]) -> str:
    """Auto-detect the tier from the test case structure."""
    if not test_cases:
        print("ERROR: No test cases found in JSON file.")
        sys.exit(1)

    sample = test_cases[0]

    if "expected_doc_id" in sample:
        return "retrieval"
    if "should_escalate" in sample:
        return "escalation"
    if "expected_answer" in sample and "expected_citations" in sample:
        # Distinguish e2e vs chatbot by ID prefix or expected_answer type
        test_id = sample.get("id", "")
        if test_id.startswith("CB"):
            return "chatbot"
        if isinstance(sample.get("expected_answer"), list):
            return "e2e"
        return "chatbot"

    print(f"ERROR: Cannot detect tier. Keys found: {list(sample.keys())}")
    sys.exit(1)


def map_retrieval(test_cases: list[dict]) -> tuple[list, list, list]:
    """Map Tier 1 (retrieval) test cases to Phoenix format."""
    inputs = []
    outputs = []
    metadata = []

    for tc in test_cases:
        inputs.append({"question": tc["question"]})
        outputs.append({
            "expected_doc": tc.get("expected_doc_id", ""),
            "expected_section": tc.get("expected_section_contains", ""),
            "expected_clause": tc.get("expected_clause", ""),
        })
        metadata.append({"test_id": tc.get("id", ""), "tier": "retrieval"})

    return inputs, outputs, metadata


def map_e2e(test_cases: list[dict]) -> tuple[list, list, list]:
    """Map Tier 2 (e2e) test cases to Phoenix format.
    expected_answer is a list of items to check coverage against.
    """
    inputs = []
    outputs = []
    metadata = []

    for tc in test_cases:
        expected_answer = tc.get("expected_answer", "")
        # Normalize to list if string
        if isinstance(expected_answer, str):
            expected_answer = [expected_answer] if expected_answer else []

        inputs.append({"question": tc["question"]})
        outputs.append({
            "expected_answer": expected_answer,
            "expected_citations": tc.get("expected_citations", []),
        })
        metadata.append({"test_id": tc.get("id", ""), "tier": "e2e"})

    return inputs, outputs, metadata


def map_chatbot(test_cases: list[dict]) -> tuple[list, list, list]:
    """Map Tier 4 (chatbot) test cases to Phoenix format.
    Same structure as e2e but expected_answer is typically a single string.
    Normalized to list for consistent evaluator interface.
    """
    inputs = []
    outputs = []
    metadata = []

    for tc in test_cases:
        expected_answer = tc.get("expected_answer", "")
        # Normalize to list — chatbot usually has a single string answer
        if isinstance(expected_answer, str):
            expected_answer = [expected_answer] if expected_answer else []

        inputs.append({"question": tc["question"]})
        outputs.append({
            "expected_answer": expected_answer,
            "expected_citations": tc.get("expected_citations", []),
        })
        metadata.append({"test_id": tc.get("id", ""), "tier": "chatbot"})

    return inputs, outputs, metadata


def map_escalation(test_cases: list[dict]) -> tuple[list, list, list]:
    """Map Tier 3 (escalation) test cases to Phoenix format."""
    inputs = []
    outputs = []
    metadata = []

    for tc in test_cases:
        inputs.append({"question": tc["question"]})
        outputs.append({
            "should_escalate": tc.get("should_escalate", True),
            "reason": tc.get("reason", ""),
        })
        metadata.append({
            "test_id": tc.get("id", ""),
            "tier": "escalation",
            "category": tc.get("category", ""),
        })

    return inputs, outputs, metadata


TIER_MAPPERS = {
    "retrieval": map_retrieval,
    "e2e": map_e2e,
    "chatbot": map_chatbot,
    "escalation": map_escalation,
}

TIER_DESCRIPTIONS = {
    "retrieval": "Tier 1: Retrieval quality — does search return the correct policy chunk?",
    "e2e": "Tier 2: End-to-end — correct answer + citations from full agent pipeline?",
    "chatbot": "Tier 4: Chatbot Q&A — answer quality for realistic user questions.",
    "escalation": "Tier 3: Escalation — does the bot refuse to answer out-of-scope questions?",
}


def derive_dataset_name(json_path: Path, tier: str) -> str:
    """Derive a Phoenix dataset name from the filename.
    
    retrieval_test.json     → retrieval-test-v1
    e2e_test.json           → e2e-test-v1
    chatbot_test_cases.json → chatbot-test-v1
    escalation_test.json    → escalation-test-v1
    """
    stem = json_path.stem  # e.g. "retrieval_test" or "chatbot_test_cases"
    # Simplify: use tier name
    return f"{tier}-test-v1"


def main():
    parser = argparse.ArgumentParser(
        description="Create a Phoenix dataset from evaluation JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/make_dataset.py eval/datasets/retrieval_test.json
  python scripts/make_dataset.py eval/datasets/e2e_test.json --name my-e2e-v2
  python scripts/make_dataset.py eval/datasets/chatbot_test_cases.json --overwrite
        """,
    )
    parser.add_argument("json_file", type=Path, help="Path to the JSON evaluation file")
    parser.add_argument("--name", type=str, default=None, help="Custom dataset name (default: auto-derived from tier)")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing dataset with same name before creating")
    parser.add_argument("--phoenix-url", type=str, default=None, help="Phoenix server URL (default: http://localhost:6006)")
    args = parser.parse_args()

    # --- Load JSON ---
    if not args.json_file.exists():
        print(f"ERROR: File not found: {args.json_file}")
        sys.exit(1)

    with open(args.json_file) as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    json_metadata = data.get("metadata", {})

    # --- Detect tier ---
    tier = detect_tier(test_cases)
    print(f"  Detected tier: {tier}")
    print(f"  Description:   {TIER_DESCRIPTIONS[tier]}")
    print(f"  Test cases:    {len(test_cases)}")

    # --- Map to Phoenix format ---
    mapper = TIER_MAPPERS[tier]
    inputs, outputs, metadata = mapper(test_cases)

    # --- Derive dataset name ---
    dataset_name = args.name or derive_dataset_name(args.json_file, tier)
    print(f"  Dataset name:  {dataset_name}")

    # --- Connect to Phoenix ---
    from phoenix.client import Client

    client_kwargs = {}
    if args.phoenix_url:
        client_kwargs["endpoint"] = args.phoenix_url
    client = Client(**client_kwargs)

    # --- Handle overwrite ---
    if args.overwrite:
        try:
            existing = client.datasets.get_dataset(dataset=dataset_name)
            print(f"  Deleting existing dataset '{dataset_name}'...")
            client.datasets.delete_dataset(dataset=existing)
            print(f"  Deleted.")
        except Exception:
            pass  # Dataset doesn't exist, nothing to delete

    # --- Create dataset ---
    try:
        dataset = client.datasets.create_dataset(
            name=dataset_name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        print(f"\n  Dataset created: {dataset.name}")
        print(f"  Examples:        {len(dataset)}")
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower() or "409" in error_msg:
            print(f"\n  ERROR: Dataset '{dataset_name}' already exists.")
            print(f"  Use --overwrite to replace it, or --name to use a different name.")
            sys.exit(1)
        raise

    # --- Print sample ---
    print(f"\n  --- Sample (first entry) ---")
    print(f"  Input:    {json.dumps(inputs[0], ensure_ascii=False)}")
    print(f"  Output:   {json.dumps(outputs[0], ensure_ascii=False)[:200]}...")
    print(f"  Metadata: {json.dumps(metadata[0], ensure_ascii=False)}")

    # --- Summary ---
    print(f"\n  View in Phoenix: http://localhost:6006/datasets")
    print(f"  Done.")


if __name__ == "__main__":
    main()
