"""
Test the vanilla RAG pipeline.

Usage:
    PYTHONPATH=. python scripts/test_pipeline.py
    PYTHONPATH=. python scripts/test_pipeline.py -q "Can I install software?"
    PYTHONPATH=. python scripts/test_pipeline.py -i  # interactive mode
"""

import argparse
import json
import time

from rag.pipeline import run_query


def main():
    parser = argparse.ArgumentParser(description="Test vanilla RAG pipeline")
    parser.add_argument("-q", "--question", help="Single question")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.question:
        _run_one(args.question)
    elif args.interactive:
        print("Vanilla RAG Pipeline — Interactive Mode")
        print("Type 'quit' to exit.\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if question:
                _run_one(question)
    else:
        # Default test questions
        questions = [
            "What is the policy on software installation?",
            "Can visitors walk around the office alone?",
            "What is the company policy on cryptocurrency mining?",
        ]
        for q in questions:
            _run_one(q)
            print()


def _run_one(question: str):
    print(f"Q: {question}")
    start = time.time()
    result = run_query(question)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s")
    print(json.dumps(result, indent=2))
    print("-" * 60)


if __name__ == "__main__":
    main()
