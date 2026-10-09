"""Rephrase a question for retrieval only (design 2026-10-09).

The verbatim question is always searched; this adds variants that use the
policy's vocabulary where the user did not. Nothing here reaches the reranker
or the agent -- both keep the user's own words.

One plain-text LLM call at temperature 0: no tools and no structured output,
because constrained decoding is one of the five conditions of the Ollama MoE
crash (docs/superpowers/specs/2026-09-17-ollama-moe-cuda-crash.md). Never
raises: whatever goes wrong, the search runs with the original question alone
and the span says why.
"""

import re
import time
from dataclasses import asdict, dataclass

from llama_index.core.llms import ChatMessage, MessageRole
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from config import settings
from rag.agent import get_llm
from rag.observability import get_tracer
from rag.vector_store import policy_titles

_MULTI = """\
You rewrite an employee's question so it can be searched against internal company policy documents.

Write exactly 2 alternative versions of the question. Each version must:
- keep the original meaning and every specific detail (who, what, which system or situation);
- use the formal wording a written company policy would use for the same thing \
(for example "Team Member", "corporate workstation", "personal data breach", "approval", "prohibited");
- be a single line: a question or a search phrase.

Do not answer the question. Do not add facts, conditions, policies or details that are not in the question.
Output only the 2 lines, with no numbering and nothing else."""

_TITLES = """

The policy documents that can be searched are:
{titles}
Use their vocabulary where it fits. Do not name a policy unless the question is clearly about it."""

_HYDE = """\
Write one short paragraph (2-3 sentences) in the style of an internal company policy clause \
that would answer the employee's question. Use formal policy wording \
(for example "Team Members must ...", "... is prohibited unless approved by ...").
It is used only to search for the real clause: do not hedge, do not mention that it is \
hypothetical, and do not address the employee. Output only the paragraph."""

REWRITE_PROMPTS = {
    "multi": _MULTI,
    "multi_titles": _MULTI + _TITLES,
    "hyde": _HYDE,
}

_MAX_REPHRASINGS = 2
_MAX_QUERY_CHARS = 300
_MAX_PASSAGE_CHARS = 1000
_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


@dataclass(frozen=True)
class RewriteResult:
    mode: str
    queries: tuple[str, ...] = ()  # extra queries only -- the original is never in here
    fallback: bool = False  # rewriting was on but produced nothing usable
    error: str = ""
    latency_ms: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["queries"] = list(self.queries)
        return d


def parse_rephrasings(raw: str, original: str) -> tuple[str, ...]:
    """At most 2 distinct lines, minus list markers, quotes, preambles and the original."""
    seen = {original.strip().casefold()}
    out = []
    for line in raw.splitlines():
        line = _MARKER.sub("", line).strip().strip("\"'“”‘’").strip()
        if not line or line.endswith(":"):
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(line[:_MAX_QUERY_CHARS])
        if len(out) == _MAX_REPHRASINGS:
            break
    return tuple(out)


def parse_passage(raw: str) -> tuple[str, ...]:
    passage = " ".join(raw.split())
    return (passage[:_MAX_PASSAGE_CHARS],) if passage else ()


def rewrite_query(question: str) -> RewriteResult:
    mode = settings.query_rewrite
    if mode == "off":
        return RewriteResult("off")

    tracer = get_tracer()
    with tracer.start_as_current_span(
        "query_rewrite",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "query_rewrite.mode": mode,
        },
    ) as span:
        started = time.monotonic()
        try:
            system = REWRITE_PROMPTS[mode]
            if mode == "multi_titles":
                system = system.format(titles="\n".join(f"- {t}" for t in policy_titles()))
            llm = get_llm(timeout=settings.query_rewrite_timeout)
            response = llm.chat(
                [
                    ChatMessage(role=MessageRole.SYSTEM, content=system),
                    ChatMessage(role=MessageRole.USER, content=question),
                ]
            )
            raw = str(response.message.content or "")
            queries = parse_passage(raw) if mode == "hyde" else parse_rephrasings(raw, question)
            error = "" if queries else "no usable rephrasing"
        except Exception as exc:  # never blocks: any failure means "search the original alone"
            queries, error = (), f"{type(exc).__name__}: {exc}"[:200]
        result = RewriteResult(
            mode=mode,
            queries=queries,
            fallback=not queries,
            error=error,
            latency_ms=round((time.monotonic() - started) * 1000),
        )
        span.set_attribute("query_rewrite.queries", list(result.queries))
        span.set_attribute("query_rewrite.fallback", result.fallback)
        span.set_attribute("query_rewrite.latency_ms", result.latency_ms)
        if result.error:
            span.set_attribute("query_rewrite.error", result.error)
        return result
