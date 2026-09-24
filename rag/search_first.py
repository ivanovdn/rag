"""Retrieval that happens before the agent, not because the agent asked.

The agent used to own `search_policies` as a tool and chose whether to call it:
21 calls across 25 production runs. A compliance answer produced without retrieval
violates the project's first constraint, so the choice is removed — retrieval runs
here, and its formatted sources are handed to the agent as part of the question.

Shared by channels/teams/bot.py and eval/agent_wrapper.py so eval measures what
production does.
"""

from dataclasses import dataclass

import rag.tools.search_policies as sp


@dataclass(frozen=True)
class PrefetchResult:
    """status is one of "ok" | "unavailable" | "no_match". sources is empty unless ok."""

    status: str
    sources: str = ""


def prefetch(question: str) -> PrefetchResult:
    """Run retrieval for `question` and classify the outcome.

    The question is passed verbatim: the retrieval stack is tuned for natural
    language, and rewriting it into keywords measurably hurts.
    """
    sp._retrieval_unavailable = False
    text = sp.search_policies(question)

    if sp._retrieval_unavailable or text == sp.UNAVAILABLE:
        # search_policies emits the Phoenix span but never logs, so this is the
        # only container-log record that retrieval — not the LLM — was the
        # failing component.
        print(
            "[worker] Unavailable (retrieval): search_policies flagged the "
            "backend unavailable (embeddings/qdrant)"
        )
        return PrefetchResult("unavailable")

    if text == sp.NO_MATCH or text.endswith(f"\n\n{sp.NO_MATCH}"):
        return PrefetchResult("no_match")

    return PrefetchResult("ok", text)


def compose_agent_input(question: str, sources: str) -> str:
    """The agent's user message: the question, then the sources.

    `sources` already carries its own "=== RETRIEVED POLICY SOURCES ===" header
    from format_sources, so nothing is added around it.
    """
    return f"{question}\n\n{sources}"
