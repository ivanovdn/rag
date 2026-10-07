"""CLI: test the compliance agent with a query."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()  # Must be before any LlamaIndex imports

from rag.agent import build_agent
from rag.search_first import compose_agent_input, prefetch
from rag.vector_store import preflight_sparse_config


async def run_query(query: str) -> str:
    pre = prefetch(query)
    if pre.status != "ok":
        return f"[{pre.status}] no sources retrieved for: {query}"
    agent = build_agent()
    response = await agent.run(user_msg=compose_agent_input(query, pre.sources))
    return str(response)


async def interactive_mode():
    print("Compliance Q&A Bot — Interactive Mode")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query:
            continue

        print("Searching policies...\n")
        pre = prefetch(query)
        if pre.status != "ok":
            print(f"[{pre.status}] no sources retrieved for: {query}\n")
            continue
        agent = build_agent()
        response = await agent.run(user_msg=compose_agent_input(query, pre.sources))
        print(f"Bot: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Test the compliance agent")
    parser.add_argument("-q", "--query", type=str, help="Single query to run")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )
    args = parser.parse_args()

    if args.interactive or args.query:
        # Before the first query, not after it. BM25 on against a collection with
        # no sparse vector is not a transient error, so search_policies re-raises
        # it and prefetch hands this CLI a bare "Not existing vector name error"
        # traceback — from the entry point a person reaches for to find out what
        # is wrong. Skipped on the --help path so that stays offline.
        preflight_sparse_config()

    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.query:
        result = asyncio.run(run_query(args.query))
        print(f"Response:\n{result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
