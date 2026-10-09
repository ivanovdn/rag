"""What a run was made of, as content hashes -- shared by eval and production.

An experiment is only worth keeping if a later reader can tell what it measured,
and a production trace is only explainable if it can be matched to one. Both
record the same values from here, under the same names, so "is production
running the prompt that experiment approved?" is a string comparison.

Hashes, not version numbers: a hash identifies the exact bytes, committed or
not, and cannot be forgotten when someone edits the text. Two runs with the same
sha12 sent the model byte-identical instructions.

Nothing here is cached. Each call re-reads the module attributes, so it always
reports what the next request will actually use -- and hashing a few KB costs
microseconds next to a multi-second answer.
"""

import hashlib

import rag.agent as agent
import rag.query_rewrite as query_rewrite
import rag.router as router
import rag.search_first as search_first
import rag.tools.search_policies as sp
from config import settings
from rag.model_digest import ollama_model_digest

# A fixed input to render the agent's user message from. The system prompt is
# only half of what the model reads: the other half is the question plus the
# [Source N] blocks, laid out by compose_agent_input and format_sources. A change
# to that layout -- how a source is labelled, what metadata it shows -- changes
# what the model does as surely as a prompt edit, and would be invisible to
# system_prompt_sha12. Two sources, one with every optional field and one with
# none, so both branches of format_sources are in the rendering.
_SAMPLE_QUESTION = "Can I use a personal laptop for work?"
_SAMPLE_RESULTS = [
    {
        "doc_title": "Sample Policy A",
        "section": "Scope",
        "clause_number": "1.2",
        "clause": "Personal devices",
        "text": "Sample clause text.",
    },
    {
        "doc_title": "Sample Policy B",
        "section": "General",
        "clause_number": "",
        "clause": "",
        "text": "Another sample clause.",
    },
]


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def prompt_identity() -> dict[str, str]:
    """Hashes of everything the ANSWERING model is told, other than the question.

    The router's prompt is separate (router_prompt_sha12): eval never runs the
    router, and recording its hash on an experiment would suggest it was measured.
    """
    rendered = search_first.compose_agent_input(
        _SAMPLE_QUESTION, sp.format_sources(_SAMPLE_RESULTS)
    )
    return {
        "system_prompt_sha12": sha12(agent.SYSTEM_PROMPT),
        "agent_input_sha12": sha12(rendered),
    }


def rewrite_identity() -> dict[str, str]:
    """Which query-rewrite mode ran, and the exact prompt template it used.

    The template, not the rendered prompt: `multi_titles` fills in titles read
    from the collection, and the collection is already in the metadata. Reading
    it here would put a Qdrant call on every request's span.
    """
    mode = settings.query_rewrite
    prompt = query_rewrite.REWRITE_PROMPTS.get(mode, "")
    return {"query_rewrite": mode, "rewrite_prompt_sha12": sha12(prompt) if prompt else ""}


def router_prompt_sha12() -> str:
    return sha12(router.ROUTER_SYSTEM_PROMPT)


def llm_digest12() -> str:
    """The first 12 hex digits of the weights LLM_MODEL resolves to, or "".

    "" when the backend has no digest to read (openai-compatible) or the host did
    not answer. Never raises: this labels a run, and a model host that is slow to
    list its tags must not cost one.
    """
    if settings.llm_backend != "ollama":
        return ""
    try:
        digest = ollama_model_digest(settings.active_ollama_url, settings.llm_model)
    except Exception:
        return ""
    return digest.lower().removeprefix("sha256:")[:12]
