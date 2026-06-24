"""
Phoenix observability integration.

Initializes OpenTelemetry tracing to Phoenix for the entire application.
Must be called once at startup before any LlamaIndex or Ollama calls.

What gets traced automatically (via LlamaIndex instrumentor):
- Every agent ReAct iteration (Thought -> Action -> Observation)
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
    """
    global _initialized
    if _initialized:
        return

    if not settings.phoenix_enabled:
        logger.info("Phoenix observability disabled (PHOENIX_ENABLED=false)")
        _initialized = True
        return

    try:
        from phoenix.otel import register

        # Connect to Phoenix server and auto-instrument all OpenInference libraries
        register(
            endpoint=settings.phoenix_endpoint,
            project_name=settings.phoenix_project_name,
            auto_instrument=True,
        )

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
        _initialized = True  # don't retry
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
