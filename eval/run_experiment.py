#!/usr/bin/env python3
"""
Run a Phoenix evaluation experiment.

Usage:
    python eval/run_experiment.py --tier tier1 --name baseline-hybrid-v1
    python eval/run_experiment.py --tier tier2 --name baseline-e2e-v1
    python eval/run_experiment.py --tier chatbot --name baseline-chatbot-v1
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_async():
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("WARNING: pip install nest_asyncio")


def make_tier1_task(top_k: int):
    from config import settings

    def _to_result_dicts(raw_results, is_hybrid: bool):
        """Convert raw search results to a common dict format."""
        if is_hybrid:
            return [
                {
                    "doc_title": r["doc_title"],
                    "section": r["section"],
                    "clause": r.get("clause", ""),
                    "clause_number": r.get("clause_number", ""),
                    "text": r.get("text", ""),
                    "retrieval_score": round(r["rrf_score"], 4),
                }
                for r in raw_results
            ]
        else:
            return [
                {
                    "doc_title": r.payload.get("doc_title", ""),
                    "section": r.payload.get("section", ""),
                    "clause": r.payload.get("clause", ""),
                    "clause_number": r.payload.get("clause_number", ""),
                    "text": r.payload.get("text", ""),
                    "retrieval_score": round(r.score, 4),
                }
                for r in raw_results
            ]

    retrieve_k = settings.reranker_candidates if settings.reranker_enabled else top_k

    if settings.bm25_enabled:
        from rag.hybrid_search import hybrid_search

        def retrieval_task(input):
            raw = hybrid_search(input["question"], top_k=retrieve_k)
            results = _to_result_dicts(raw, is_hybrid=True)

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
    else:
        from rag.embeddings import embed_query
        from rag.vector_store import search_vectors

        def retrieval_task(input):
            vector = embed_query(input["question"])
            raw = search_vectors(vector, top_k=retrieve_k)
            results = _to_result_dicts(raw, is_hybrid=False)

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
    from eval.agent_wrapper import build_instrumented_agent, get_log, clear_log, parse_agent_response

    async def _run_fresh_agent(question, verbose):
        agent = build_instrumented_agent(verbose=verbose)
        return await agent.run(question)

    def e2e_task(input):
        question = input["question"]
        clear_log()

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_run_fresh_agent(question, verbose))

        parsed = parse_agent_response(str(result))
        tool_calls = list(get_log())

        agent_search_results, search_queries = [], []
        for call in tool_calls:
            if call["tool"] == "search_policies":
                search_queries.append(call["query"])
                agent_search_results.extend(call["results"])

        seen = set()
        unique_results = []
        for r in agent_search_results:
            key = (r["doc_title"], r["section"], r["clause"])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        section_calls = [c for c in tool_calls if c["tool"] == "get_section"]
        escalation_calls = [c for c in tool_calls if c["tool"] == "escalate_to_compliance"]

        return {
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "escalation": parsed["escalation"],
            "parse_success": parsed["parse_success"],
            "raw_response": parsed["raw_response"],
            "search_results": unique_results,
            "agent_metadata": {
                "search_queries": search_queries,
                "num_searches": len(search_queries),
                "num_section_fetches": len(section_calls),
                "section_fetches": [
                    {"doc_id": c["doc_id"], "section": c["section_name"], "found": c["found"]}
                    for c in section_calls
                ],
                "escalated": len(escalation_calls) > 0,
                "escalation_reason": escalation_calls[0]["reason"] if escalation_calls else None,
            },
        }
    return e2e_task


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
    args = parser.parse_args()

    setup_async()
    from phoenix.client import Client
    from eval.evaluators import TIER1_EVALUATORS, TIER2_EVALUATORS, CHATBOT_EVALUATORS
    from config import settings

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
        client_kwargs["endpoint"] = args.phoenix_url
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
        "reranker_backend": settings.reranker_backend if settings.reranker_enabled else "none",
        "reranker_url": settings.reranker_url if settings.reranker_enabled else "none",
    }

    if args.tier == "tier1":
        task = make_tier1_task(top_k=top_k)
        evaluators = TIER1_EVALUATORS
        metadata = {**infra_meta, "search_type": search_type, "embedding_model": settings.embedding_model,
                     "reranker": reranker_info, "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "top_k": top_k, "tier": "tier1"}
    else:
        task = make_agent_task(verbose=args.verbose)
        evaluators = TIER2_EVALUATORS if args.tier == "tier2" else CHATBOT_EVALUATORS
        metadata = {**infra_meta, "llm": settings.llm_model, "search_type": search_type,
                     "reranker": reranker_info,
                     "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "agent_type": "react", "top_k": top_k, "tier": args.tier,
                     "structured_output": True}

    print(f"  Evaluators:  {[e.__name__ for e in evaluators]}")

    experiment = client.experiments.run_experiment(
        dataset=dataset, task=task, evaluators=evaluators,
        experiment_name=args.name, experiment_description=description,
        experiment_metadata=metadata,
    )
    print(f"\n  Done: {args.name}")
    print(f"  View: http://localhost:6006/datasets")


if __name__ == "__main__":
    main()
