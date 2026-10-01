#!/usr/bin/env python3
"""
Run a Phoenix evaluation experiment.

Usage:
    python eval/run_experiment.py --tier tier1 --name baseline-hybrid-v1
    python eval/run_experiment.py --tier tier2 --name baseline-e2e-v1
    python eval/run_experiment.py --tier chatbot --name baseline-chatbot-v1
"""

import argparse
import hashlib
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Safe at module level: config pulls no LlamaIndex or Ollama, so it does not
# compete with the init_observability() ordering the local imports below protect.
# _tracing_target needs it, and so does the test that patches it here.
# init_observability itself is imported inside main(), not here: rag.observability
# is just as import-light, but test_eval_metadata.py guards this file against any
# top-level `rag.*` import, and satisfying that guard costs nothing.
from config import settings

# Eval spans go to their own Phoenix project. The bot's project records what real
# users asked; a 61-question gate run dropped into it reads as production traffic
# to whoever looks at it later.
EVAL_PROJECT_NAME = "compliance-bot-eval"


def _tracing_target(args) -> tuple[str, str] | None:
    """(project, endpoint) for this run's spans, or None when tracing is off.

    The endpoint follows --phoenix-url when that is given. That flag moves the
    client which writes the experiment; leaving traces on the configured endpoint
    would put the experiment on one Phoenix and its spans on another, with nothing
    anywhere reporting the split.
    """
    if args.no_trace:
        return None
    project = args.phoenix_project or EVAL_PROJECT_NAME
    if args.phoenix_url:
        endpoint = f"{args.phoenix_url.rstrip('/')}/v1/traces"
    else:
        endpoint = settings.phoenix_endpoint
    return project, endpoint


def setup_async():
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("WARNING: pip install nest_asyncio")


def make_tier1_task(top_k: int):
    # Local imports: init_observability() must run before any LlamaIndex/Ollama
    # import, and this module is imported by the CLI entry point below.
    from config import settings
    from rag.embeddings import embed_query
    from rag.vector_store import search_chunks

    retrieve_k = settings.reranker_candidates if settings.reranker_enabled else top_k

    def retrieval_task(input):
        # One path for both modes: Qdrant fuses server-side, so dense-only and
        # dense+sparse both come back as ScoredPoints with the same payload.
        vector = embed_query(input["question"])
        raw = search_chunks(input["question"], vector, top_k=retrieve_k)
        results = [
            {
                "doc_title": r.payload.get("doc_title", ""),
                "section": r.payload.get("section", ""),
                "clause": r.payload.get("clause", ""),
                "clause_number": r.payload.get("clause_number", ""),
                "text": r.payload.get("text", ""),
                "retrieval_score": round(r.score, 4),
            }
            for r in raw
        ]

        if settings.reranker_enabled and results:
            from rag.reranker import rerank

            results = rerank(input["question"], results, top_n=settings.reranker_top_n)

        return {
            "search_results": [
                {
                    "doc_title": r["doc_title"],
                    "section": r["section"],
                    "clause": r.get("clause", ""),
                    "clause_number": r.get("clause_number", ""),
                    "retrieval_score": r.get("retrieval_score", 0),
                    "rerank_score": r.get("rerank_score"),
                    "original_rank": r.get("original_rank"),
                }
                for r in results
            ]
        }

    return retrieval_task


def make_agent_task(verbose: bool = False):
    from eval.agent_wrapper import (
        build_instrumented_agent,
        clear_log,
        compose_agent_input,
        get_log,
        parse_agent_response,
        prefetch_logged,
    )

    async def _run_fresh_agent(agent_input, verbose):
        agent = build_instrumented_agent(verbose=verbose)
        return await agent.run(agent_input)

    def e2e_task(input):
        question = input["question"]
        clear_log()

        pre = prefetch_logged(question)
        tool_calls = list(get_log())
        search_queries = [c["query"] for c in tool_calls if c["tool"] == "search_policies"]

        def _agent_metadata(escalation):
            # Escalation is read from the JSON field (or synthesized below on a
            # short-circuit), which is the only escalation path production has
            # ever had — the tool that used to be counted here was never called.
            return {
                "search_queries": search_queries,
                "num_searches": len(search_queries),
                "escalated": bool(escalation.get("needed")),
                "escalation_reason": escalation.get("reason") or None,
            }

        # Mirrors channels/teams/bot.py::_run_rag's branch shape exactly, so the
        # two cannot drift: an infra failure never reaches the agent and never
        # reads as a content escalation, and a no-match escalates on the same
        # fixed reason text without spending an LLM call to reach it.
        if pre.status == "unavailable":
            escalation = {"needed": False, "reason": ""}
            return {
                "status": "unavailable",
                "answer": "",
                "citations": [],
                "escalation": escalation,
                "parse_success": False,
                "raw_response": "",
                "search_results": [],
                "agent_metadata": _agent_metadata(escalation),
            }

        if pre.status == "no_match":
            escalation = {
                "needed": True,
                "reason": "No relevant policy was found for this question.",
            }
            return {
                "status": "no_match",
                "answer": "",
                "citations": [],
                "escalation": escalation,
                "parse_success": True,
                "raw_response": "",
                "search_results": [],
                "agent_metadata": _agent_metadata(escalation),
            }

        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            _run_fresh_agent(compose_agent_input(question, pre.sources), verbose)
        )

        parsed = parse_agent_response(str(response))

        agent_search_results = []
        for call in tool_calls:
            if call["tool"] == "search_policies":
                agent_search_results.extend(call["results"])

        seen = set()
        unique_results = []
        for r in agent_search_results:
            key = (r["doc_title"], r["section"], r["clause"])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return {
            "status": "ok",
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "escalation": parsed["escalation"],
            "parse_success": parsed["parse_success"],
            "raw_response": parsed["raw_response"],
            "search_results": unique_results,
            "agent_metadata": _agent_metadata(parsed["escalation"]),
        }
    return e2e_task


def _prompt_meta() -> dict:
    """Identify the exact SYSTEM_PROMPT this run used.

    Imported locally, like every other rag/ import in this module, so importing
    run_experiment does not drag LlamaIndex in.

    The hash is what makes prompt experiments comparable: two runs with the same
    sha12 used byte-identical prompts, committed or not. chars and tokens make the
    context-budget cost of a prompt edit visible alongside its quality effect —
    the whole point of the audit this branch came out of.
    """
    from rag.agent import FIXED_OVERHEAD_TOKENS, SYSTEM_PROMPT

    return {
        "system_prompt_sha12": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "fixed_overhead_tokens": FIXED_OVERHEAD_TOKENS,
    }


PROMPT_REGISTRY_NAME = "compliance-system-prompt"


def _mirror_prompt_to_registry(client) -> dict:
    """Mirror SYSTEM_PROMPT into Phoenix's prompt registry and return its version id.

    The registry is a MIRROR, never a source. rag/agent.py stays authoritative:
    the prompt is this bot's entire safety surface (never answer ungrounded, quote
    verbatim, escalate when uncertain), and fetching it at runtime would mean a
    Phoenix outage or a stray Playground edit could silently change how a
    compliance bot behaves. Nothing reads this back.

    What it buys: the actual text visible beside each experiment, and Phoenix's
    diff between versions — so when a sweep moves a number you can see what the
    prompt change was, instead of diffing 2,350 characters by hand.

    Idempotent: versions are tagged with the prompt's sha12, so re-running with an
    unchanged prompt reuses the existing version instead of piling up duplicates.

    Never raises. An eval run costs real GPU time on a shared host; a registry
    hiccup must degrade to "no version id in the metadata", not lose the run.
    """
    from rag.agent import SYSTEM_PROMPT

    sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    try:
        existing = client.prompts.get(prompt_identifier=PROMPT_REGISTRY_NAME, tag=sha)
        return {"system_prompt_version_id": existing.id}
    except Exception:
        pass  # not registered yet (or the registry is unreachable) — try to create

    try:
        from config import settings
        from phoenix.client.types import PromptVersion

        version = client.prompts.create(
            name=PROMPT_REGISTRY_NAME,
            prompt_description=(
                "Read-only mirror of rag/agent.py SYSTEM_PROMPT. Nothing reads this at "
                "runtime — editing it here changes nothing. Edit rag/agent.py."
            ),
            version=PromptVersion(
                [{"role": "system", "content": SYSTEM_PROMPT}],
                model_name=settings.llm_model,
                model_provider="OLLAMA",
                # NONE, not MUSTACHE/F_STRING: the prompt embeds a literal JSON
                # block with { } braces. Any templating format would try to
                # interpolate them and mangle the output contract.
                template_format="NONE",
                description=f"sha12={sha} · {len(SYSTEM_PROMPT)} chars",
            ),
        )
        try:
            client.prompts.tags.create(
                prompt_version_id=version.id,
                name=sha,
                description="sha12 of rag/agent.py SYSTEM_PROMPT at the time of this run",
            )
        except Exception:
            pass  # the version exists either way; the tag is only for idempotency
        return {"system_prompt_version_id": version.id}
    except Exception as exc:
        print(f"  WARNING: could not mirror the prompt into Phoenix "
              f"({type(exc).__name__}: {str(exc)[:80]}); the run continues without it")
        return {}


TIER_CONFIG = {
    "tier1": {"default_dataset": "retrieval-test-v1", "description": "Retrieval: hybrid search"},
    "tier2": {"default_dataset": "e2e-test-v1", "description": "E2E: full agent + structured JSON"},
    "chatbot": {"default_dataset": "chatbot-test-v1", "description": "Chatbot: realistic user questions"},
}


def main():
    parser = argparse.ArgumentParser(description="Run Phoenix evaluation experiment.")
    parser.add_argument("--tier", choices=["tier1", "tier2", "chatbot"], required=True)
    parser.add_argument("--name", default=None, help="Experiment name (auto-generated from config if omitted)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval_top_k from .env")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--phoenix-url", default=None)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not emit Phoenix traces for this run",
    )
    parser.add_argument(
        "--phoenix-project",
        default=None,
        help=f"Phoenix project for this run's traces (default: {EVAL_PROJECT_NAME})",
    )
    args = parser.parse_args()

    # Before setup_async() and every local import below it: make_agent_task() pulls
    # eval/agent_wrapper.py, which imports llama_index at module level, and Phoenix's
    # instrumentors must be installed before that happens or the agent's own spans
    # are never recorded. This entry point had no such call at all until now, which
    # is why eval runs produced no spans of any kind.
    trace_target = _tracing_target(args)
    if trace_target is None:
        print("tracing:      off (--no-trace)")
    else:
        from rag.observability import init_observability

        print(f"tracing:      project={trace_target[0]} endpoint={trace_target[1]}")
        init_observability(project_name=trace_target[0], endpoint=trace_target[1])

    setup_async()
    from phoenix.client import Client
    from eval.evaluators import TIER1_EVALUATORS, TIER2_EVALUATORS, CHATBOT_EVALUATORS
    from config import settings
    from rag.vector_store import preflight_sparse_config

    # BM25_ENABLED=true against a collection with no sparse vector fails every
    # query with "Not existing vector name error". That is not transient, so
    # every question escalates and hit_evaluator comes out near zero -- which
    # reads exactly like the sparse half failing on its merits, and the rollout's
    # response to a low gate score is a 100-line fallback encoder. Both
    # QDRANT_COLLECTION and BM25_ENABLED have to be set for the gate run; this
    # turns forgetting the first into a message instead of a plausible zero.
    preflight_sparse_config()

    top_k = args.top_k if args.top_k is not None else settings.retrieval_top_k
    tier_cfg = TIER_CONFIG[args.tier]
    dataset_name = args.dataset or tier_cfg["default_dataset"]
    description = args.description or tier_cfg["description"]

    # Auto-generate experiment name from config if not provided
    if not args.name:
        embed_short = settings.embedding_model.split("/")[-1]
        search = "hybrid" if settings.bm25_enabled else "vector"
        if settings.reranker_enabled:
            reranker_short = settings.reranker_model.replace("/", "-")
            args.name = f"agentic_{args.tier}_{embed_short}_{search}_cand{settings.reranker_candidates}_{reranker_short}_top{settings.reranker_top_n}"
        else:
            args.name = f"agentic_{args.tier}_{embed_short}_{search}_top{top_k}"

    client_kwargs = {}
    if args.phoenix_url:
        # base_url, not endpoint: phoenix.client.Client takes
        # (base_url, api_key, headers, http_client). "endpoint" was the old
        # kwarg and raises TypeError on arize-phoenix >= 13. Never caught
        # because locally Phoenix is at the default localhost:6006, so this
        # flag is only reached when running from inside a container.
        client_kwargs["base_url"] = args.phoenix_url
    client = Client(**client_kwargs)

    try:
        dataset = client.datasets.get_dataset(dataset=dataset_name)
    except Exception:
        print(f"ERROR: Dataset '{dataset_name}' not found.")
        print(f"  Create: python scripts/make_dataset.py eval/datasets/<file>.json")
        sys.exit(1)

    print(f"  Tier:        {args.tier}")
    print(f"  Dataset:     {dataset.name} ({len(dataset)} examples)")
    print(f"  Experiment:  {args.name}")
    print(f"  Embedding:   {settings.embedding_model}")
    print(f"  LLM:         {settings.llm_model}")
    print(f"  Ollama:      {settings.active_ollama_url} ({'remote' if settings.use_remote_ollama else 'local'}, timeout={settings.active_request_timeout}s)")
    print(f"  top_k:       {top_k} (from {'--top-k' if args.top_k is not None else '.env'})")
    print(f"  BM25:        {'on' if settings.bm25_enabled else 'off'}")
    print(f"  Reranker:    {settings.reranker_model if settings.reranker_enabled else 'off'}" + (f" (candidates={settings.reranker_candidates}, top_n={settings.reranker_top_n})" if settings.reranker_enabled else ""))

    search_type = "hybrid_rrf" if settings.bm25_enabled else "vector_only"
    reranker_info = settings.reranker_model if settings.reranker_enabled else "none"
    infra = "remote" if settings.use_remote_ollama else "local"
    infra_meta = {
        "infra": infra,
        "llm_backend": settings.llm_backend,
        "llm_url": settings.active_ollama_url,
        "embedding_source": settings.embedding_source,
        "embedding_url": settings.ollama_embedding_url if settings.embedding_source == "ollama" else "local",
        "qdrant_url": settings.active_qdrant_url,
        # The collection is a retrieval parameter now: v1 and v2 hold different
        # indexes, so a run that does not name it cannot be compared later.
        "qdrant_collection": settings.qdrant_collection,
        # Which Phoenix project holds this run's spans. Without it, finding a
        # six-week-old run's traces means guessing.
        "phoenix_project": trace_target[0] if trace_target else None,
        "bm25_enabled": settings.bm25_enabled,
        "bm25_avg_len": settings.bm25_avg_len,
        "reranker_backend": settings.reranker_backend if settings.reranker_enabled else "none",
        "reranker_url": settings.reranker_url if settings.reranker_enabled else "none",
    }

    if args.tier == "tier1":
        task = make_tier1_task(top_k=top_k)
        evaluators = TIER1_EVALUATORS
        metadata = {**infra_meta, "search_type": search_type, "embedding_model": settings.embedding_model,
                     "reranker": reranker_info, "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "reranker_min_score": settings.reranker_min_score if settings.reranker_enabled else None,
                     "min_confidence_score": settings.min_confidence_score if (not settings.reranker_enabled and not settings.bm25_enabled) else None,
                     "top_k": top_k, "tier": "tier1"}
    else:
        task = make_agent_task(verbose=args.verbose)
        evaluators = TIER2_EVALUATORS if args.tier == "tier2" else CHATBOT_EVALUATORS
        # Every knob that changes the result belongs here: an experiment whose
        # parameters are only recoverable from its NAME cannot be compared against
        # another six weeks later. reranker_min_score in particular gates retrieval
        # entirely when it fires, and embedding_model was missing from the agent
        # tiers although it decides what is retrievable at all.
        metadata = {**infra_meta, "llm": settings.llm_model, "search_type": search_type,
                     "embedding_model": settings.embedding_model,
                     "reranker": reranker_info,
                     "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "reranker_min_score": settings.reranker_min_score if settings.reranker_enabled else None,
                     "min_confidence_score": settings.min_confidence_score if (not settings.reranker_enabled and not settings.bm25_enabled) else None,
                     "num_ctx": settings.ollama_num_ctx,
                     "temperature": settings.llm_temperature,
                     # The prompt is a parameter like any other, and the one most
                     # likely to be edited between runs. The hash identifies it
                     # exactly (two runs with the same hash used the same prompt,
                     # committed or not); chars and tokens make the context-budget
                     # cost of a prompt change visible in the comparison.
                     **_prompt_meta(), **_mirror_prompt_to_registry(client),
                     "agent_type": "function-agent-toolfree", "top_k": top_k, "tier": args.tier,
                     "structured_output": True}

    print(f"  Evaluators:  {[e.__name__ for e in evaluators]}")

    client.experiments.run_experiment(
        dataset=dataset, task=task, evaluators=evaluators,
        experiment_name=args.name, experiment_description=description,
        experiment_metadata=metadata,
    )
    print(f"\n  Done: {args.name}")
    print(f"  View: http://localhost:6006/datasets")


if __name__ == "__main__":
    main()
