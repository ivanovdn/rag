"""
Phoenix observability integration.

Initializes OpenTelemetry tracing to Phoenix for the entire application.
Must be called once at startup before any LlamaIndex or Ollama calls.

What gets traced automatically (via LlamaIndex instrumentor):
- The agent's one LLM call (FunctionAgent, tool-free — no tool-call loop)
- The pre-agent retrieval call (search_policies, via rag/search_first.py)
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


def init_observability(
    project_name: str | None = None, endpoint: str | None = None
) -> None:
    """
    Initialize Phoenix tracing. Safe to call multiple times (idempotent).

    Both arguments default to the configured values. They exist so an eval run can
    write its spans to its own project rather than the bot's — the bot's project is
    a record of what real users asked, and a 61-question gate run dropped into it
    reads as production traffic to anyone looking later. See eval/run_experiment.py.

    PHOENIX_ENABLED=false still wins over both, so the kill switch keeps meaning
    what it says even when a caller passes arguments.

    Idempotent via a module-level flag, which means a SECOND call with DIFFERENT
    arguments is silently ignored — the first call's project is the one that
    sticks. No entry point calls this twice today.

    Call this at the top of:
    - scripts/test_query.py
    - scripts/run_eval.py
    """
    global _initialized
    if _initialized:
        return

    if not settings.phoenix_enabled:
        logger.info("Phoenix observability disabled (PHOENIX_ENABLED=false)")
        _initialized = True
        return

    project = project_name or settings.phoenix_project_name
    target = endpoint or settings.phoenix_endpoint

    try:
        from phoenix.otel import register

        # Connect to Phoenix server and auto-instrument all OpenInference libraries
        register(endpoint=target, project_name=project, auto_instrument=True)

        logger.info(
            f"Phoenix observability initialized: endpoint={target}, project={project}"
        )
        _initialized = True

    except ImportError:
        logger.warning(
            "Phoenix packages not installed. Run: "
            "pip install arize-phoenix openinference-instrumentation-llama-index"
        )
        _initialized = True  # don't retry
    except Exception as e:
        logger.warning(f"Failed to initialize Phoenix: {e}. Continuing without observability.")
        _initialized = True


def get_tracer():
    """
    Get an OpenTelemetry tracer for manual span creation.

    Usage:
        tracer = get_tracer()
        with tracer.start_as_current_span("search_vectors") as span:
            span.set_attribute("query", query)
            span.set_attribute("vector_top_score", 0.87)
            # ... do work ...
    """
    from opentelemetry import trace

    if not settings.phoenix_enabled:
        return trace.get_tracer("noop")

    return trace.get_tracer("compliance-bot")


def record_infra_unavailable(failed_component: str, error_type: str, retries_attempted: int) -> None:
    """Emit a Phoenix span marking a transient backend-unavailable event.

    failed_component: "embeddings" | "qdrant" | "llm"
    Makes infra-down events filterable in Phoenix, distinct from content escalations.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("infra_unavailable") as span:
        span.set_attribute("infra_unavailable", True)
        span.set_attribute("failed_component", failed_component)
        span.set_attribute("error_type", error_type)
        span.set_attribute("retries_attempted", retries_attempted)


def record_classification(category: str, confidence: float, fallback: bool, message: str) -> None:
    """Emit a Phoenix span for a pre-retrieval classification decision.

    category: the resolved Category value acted on
              ("in_scope" | "greeting" | "out_of_scope" | "unintelligible").
    fallback: True when the safe default (IN_SCOPE) overrode the model or the classifier failed.
    message: the user's message text, recorded in full for audit (this is a compliance bot —
             every classification must be auditable against the message that produced it).
    Makes the classification distribution and safe-default fallback rate queryable in Phoenix.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("classification") as span:
        span.set_attribute("router_category", category)
        span.set_attribute("router_confidence", confidence)
        span.set_attribute("router_fallback", fallback)
        span.set_attribute("router_message", message)


def record_floor_rejection(top_score: float, threshold: float) -> None:
    """Emit a Phoenix span for a search rejected by the relevance floor.

    Makes the rejection rate countable in production, which is how the threshold
    gets tuned and how a too-high floor is caught before users feel it.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("retrieval_floor_rejected") as span:
        span.set_attribute("retrieval.floor_rejected", True)
        span.set_attribute("retrieval.top_score", top_score)
        span.set_attribute("retrieval.floor_threshold", threshold)
