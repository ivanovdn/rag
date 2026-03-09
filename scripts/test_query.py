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


async def run_query(query: str) -> str:
    agent = build_agent()
    response = await agent.run(query)
    return str(response)


async def interactive_mode():
    agent = build_agent()
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
        response = await agent.run(query)
        print(f"Bot: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Test the compliance agent")
    parser.add_argument("-q", "--query", type=str, help="Single query to run")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.query:
        result = asyncio.run(run_query(args.query))
        print(f"Response:\n{result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
