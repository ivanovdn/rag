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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Safe at module level: config pulls no LlamaIndex or Ollama, so it does not
# compete with the init_observability() ordering the local imports below protect.
# _tracing_endpoint needs it, and so does the test that patches it here.
# init_observability itself is imported inside main(), not here: rag.observability
# is just as import-light, but test_eval_metadata.py guards this file against any
# top-level `rag.*` import, and satisfying that guard costs nothing.
from config import settings


def _tracing_endpoint(args) -> str | None:
    """Where this run's spans go, or None when tracing is off.

    Deliberately does not choose a PROJECT. Phoenix's run_experiment files task
    spans under its own per-experiment project and overrides whatever was
    registered, so an override here would be inert — measured 2026-10-02, when a
    registered `compliance-bot-eval` was never created while the spans sat in
    `Experiment-<hash>`. Phoenix already records the real project on the
    experiment, and already keeps it out of the bot's project unaided.

    The endpoint follows --phoenix-url when that is given. That flag moves the
    client which writes the experiment; leaving traces on the configured endpoint
    would put the experiment on one Phoenix and its spans on another, with nothing
    anywhere reporting the split.
    """
    if args.no_trace:
        return None
    if args.phoenix_url:
        return f"{args.phoenix_url.rstrip('/')}/v1/traces"
    return settings.phoenix_endpoint


def setup_async():
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("WARNING: pip install nest_asyncio")


def make_tier1_task(top_k: int):
    # Local import: init_observability() must run before any LlamaIndex/Ollama
    # import, and search_policies pulls llama_index.
    import rag.tools.search_policies as sp

    def retrieval_task(input):
        # The same call production makes -- rewrite, embed, fused search, rerank,
        # floors. Tier1 used to carry its own copy of that path, which stopped
        # being the shipped one the moment a step (the query rewrite) was added
        # inside search_policies. `top_k` only matters with the reranker off.
        sp.search_policies(input["question"], top_k=top_k)
        if sp._retrieval_unavailable:
            raise RuntimeError("retrieval unavailable (embeddings/qdrant) -- not a retrieval miss")
        return {
            "search_results": [dict(r) for r in sp._last_search_results],
            "rewrite": dict(sp._last_rewrite),
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
                "rewrite": next((c.get("rewrite", {}) for c in tool_calls if c["tool"] == "search_policies"), {}),
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

        # The key is exactly what evaluators._match_result can distinguish: it
        # reads doc_title, section and clause and never clause_number, so two
        # chunks of one clause are one hit opportunity and collapsing them cannot
        # move a score. Widening the key would count a clause twice; narrowing it
        # would merge clauses the evaluators tell apart.
        #
        # The consequence is that output["search_results"] is NOT what the agent
        # saw — production does no dedup, format_sources shows all of them. On
        # chatbot-test-v1 this turned 6 reranked sources into 5 for 12 of 61
        # questions while reranker.results_out said 6 on every span, which reads
        # exactly like the reranker dropping one. It is not. Hence the count
        # below: the output has to say what it did to itself.
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
            "search_results_before_dedup": len(agent_search_results),
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
    from rag.run_identity import prompt_identity

    # prompt_identity also carries agent_input_sha12: the layout of the question
    # and its [Source N] blocks, which the model reads as closely as the prompt.
    # The production bot stamps the same two hashes on every compliance_request
    # span, so an experiment and a trace can be matched by value.
    return {
        **prompt_identity(),
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "fixed_overhead_tokens": FIXED_OVERHEAD_TOKENS,
    }


def _git_commit() -> str:
    """The commit of the code this run executes.

    Asks git first, because on the VM the eval container mounts a worktree's
    source over the image, and the image's baked GIT_COMMIT names the DEPLOYED
    commit, not the one being measured. Inside that container there is no .git
    to ask, so it falls back to GIT_COMMIT -- which is right only when the
    runbook's `-e GIT_COMMIT=...` override is passed (SETUP.md Step 11).
    """
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return settings.git_commit
    return out.stdout.strip() or settings.git_commit


def _llm_digest_meta() -> dict:
    """The model's digest, plus a printed warning when it is not the expected one.

    Closes follow-up #12: a run labelled only `qwen3.6:latest` names a tag that
    moves on a host this project does not own, so the weights it measured could
    not be recovered later.
    """
    from rag.run_identity import llm_digest12

    digest = llm_digest12()
    expected = settings.llm_model_digest.strip().lower()
    print(f"  LLM digest:  {digest or 'unavailable'}"
          + (f" (expected {expected})" if expected else ""))
    if expected and not digest:
        print("  WARNING: could not confirm the model digest -- the check did not "
              "run, which is not the same as passing")
    elif expected and not digest.startswith(expected[:12]):
        print(f"  WARNING: {settings.llm_model} resolves to {digest}, not the expected "
              f"{expected} (LLM_MODEL_DIGEST): this run measures different weights "
              "from the ones production was verified against")
    return {"llm_digest": digest}


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
    """
    from rag.agent import SYSTEM_PROMPT

    return _mirror_to_registry(
        client,
        name=PROMPT_REGISTRY_NAME,
        text=SYSTEM_PROMPT,
        source="rag/agent.py SYSTEM_PROMPT",
        id_key="system_prompt_version_id",
    )


REWRITE_REGISTRY_PREFIX = "compliance-rewrite-prompt-"


def _mirror_rewrite_prompt_to_registry(client) -> dict:
    """Mirror the active QUERY_REWRITE prompt template; {} when the rewrite is off.

    One registry entry per mode, so Phoenix's version diff compares edits of one
    prompt and never `multi` against `hyde`. The TEMPLATE is mirrored -- the same
    text rewrite_prompt_sha12 hashes -- so `multi_titles` keeps its literal
    {titles} placeholder (template_format NONE) rather than one day's titles.
    """
    from rag.query_rewrite import REWRITE_PROMPTS

    mode = settings.query_rewrite
    if mode not in REWRITE_PROMPTS:
        return {}
    return _mirror_to_registry(
        client,
        name=f"{REWRITE_REGISTRY_PREFIX}{mode}",
        text=REWRITE_PROMPTS[mode],
        source=f"rag/query_rewrite.py REWRITE_PROMPTS['{mode}']",
        id_key="rewrite_prompt_version_id",
    )


def _mirror_to_registry(client, *, name: str, text: str, source: str, id_key: str) -> dict:
    """Store `text` as a version of registry prompt `name`; return {id_key: version id}.

    Idempotent: versions are tagged with the text's sha12, so re-running with an
    unchanged prompt reuses the existing version instead of piling up duplicates.

    Never raises. An eval run costs real GPU time on a shared host; a registry
    hiccup must degrade to "no version id in the metadata", not lose the run.
    """
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    try:
        existing = client.prompts.get(prompt_identifier=name, tag=sha)
        return {id_key: existing.id}
    except Exception:
        pass  # not registered yet (or the registry is unreachable) — try to create

    try:
        from phoenix.client.types import PromptVersion

        version = client.prompts.create(
            name=name,
            prompt_description=(
                f"Read-only mirror of {source}. Nothing reads this at runtime — "
                "editing it here changes nothing. Edit the source file."
            ),
            version=PromptVersion(
                [{"role": "system", "content": text}],
                model_name=settings.llm_model,
                model_provider="OLLAMA",
                # NONE, not MUSTACHE/F_STRING: SYSTEM_PROMPT embeds a literal JSON
                # block and the rewrite template a literal {titles}. Any
                # templating format would try to interpolate them.
                template_format="NONE",
                description=f"sha12={sha} · {len(text)} chars",
            ),
        )
        try:
            client.prompts.tags.create(
                prompt_version_id=version.id,
                name=sha,
                description=f"sha12 of {source} at the time of this run",
            )
        except Exception:
            pass  # the version exists either way; the tag is only for idempotency
        return {id_key: version.id}
    except Exception as exc:
        print(f"  WARNING: could not mirror {source} into Phoenix "
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
    parser.add_argument(
        "--candidates",
        "--top-k",
        dest="candidates",
        type=int,
        default=None,
        help="Override RERANKER_CANDIDATES for this run (sizes both prefetches and the fused limit)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--phoenix-url", default=None)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not emit Phoenix traces for this run",
    )
    args = parser.parse_args()

    # Before setup_async() and every local import below it: make_agent_task() pulls
    # eval/agent_wrapper.py, which imports llama_index at module level, and Phoenix's
    # instrumentors must be installed before that happens or the agent's own spans
    # are never recorded. This entry point had no such call at all until now, which
    # is why eval runs produced no spans of any kind.
    trace_endpoint = _tracing_endpoint(args)
    if trace_endpoint is None:
        print("tracing:      off (--no-trace)")
    else:
        from rag.observability import init_observability

        print(f"tracing:      endpoint={trace_endpoint}")
        print("              spans land in Phoenix's per-experiment project, "
              "reachable from the experiment -- NOT from the Projects page")
        init_observability(endpoint=trace_endpoint)

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

    # Write the override INTO settings rather than threading it as a parameter.
    # Only tier1 builds its own retrieval; tier2/chatbot go through
    # rag.tools.search_policies, which reads settings.reranker_candidates
    # directly — so a threaded parameter reached exactly one of the three tiers
    # and silently did nothing on the one the gate is measured from. Every
    # reader must see the same number or the flag is decoration.
    if args.candidates is not None:
        settings.reranker_candidates = args.candidates
    candidates = settings.reranker_candidates
    tier_cfg = TIER_CONFIG[args.tier]
    dataset_name = args.dataset or tier_cfg["default_dataset"]
    description = args.description or tier_cfg["description"]

    # Auto-generate experiment name from config if not provided
    if not args.name:
        embed_short = settings.embedding_model.split("/")[-1]
        search = "hybrid" if settings.bm25_enabled else "vector"
        if settings.reranker_enabled:
            reranker_short = settings.reranker_model.replace("/", "-")
            args.name = f"agentic_{args.tier}_{embed_short}_{search}_cand{candidates}_{reranker_short}_top{settings.reranker_top_n}"
        else:
            args.name = f"agentic_{args.tier}_{embed_short}_{search}_cand{candidates}"

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
    print(f"  Candidates:  {candidates} (from {'--candidates' if args.candidates is not None else 'RERANKER_CANDIDATES in .env'}) — sizes both prefetches and the fused limit")
    print(f"  BM25:        {'on' if settings.bm25_enabled else 'off'}")
    print(f"  Reranker:    {settings.reranker_model if settings.reranker_enabled else 'off'}" + (f" (top_n={settings.reranker_top_n})" if settings.reranker_enabled else ""))
    print(f"  Rewrite:     {settings.query_rewrite}")

    search_type = "hybrid_rrf" if settings.bm25_enabled else "vector_only"
    reranker_info = settings.reranker_model if settings.reranker_enabled else "none"
    infra = "remote" if settings.use_remote_ollama else "local"
    from rag.run_identity import rewrite_identity

    git_commit = _git_commit()
    print(f"  Commit:      {git_commit}")
    infra_meta = {
        # Which code ran. The prompt hashes below are exact for the prompt; this
        # covers everything else (retrieval, parsing, evaluators).
        "git_commit": git_commit,
        # Applies to every tier: tier1 now retrieves through search_policies,
        # which is where the rewrite runs.
        **rewrite_identity(),
        **_mirror_rewrite_prompt_to_registry(client),
        "infra": infra,
        "llm_backend": settings.llm_backend,
        "llm_url": settings.active_ollama_url,
        "embedding_source": settings.embedding_source,
        "embedding_url": settings.ollama_embedding_url if settings.embedding_source == "ollama" else "local",
        "qdrant_url": settings.active_qdrant_url,
        # The collection is a retrieval parameter now: v1 and v2 hold different
        # indexes, so a run that does not name it cannot be compared later.
        "qdrant_collection": settings.qdrant_collection,
        "bm25_enabled": settings.bm25_enabled,
        "bm25_avg_len": settings.bm25_avg_len,
        "reranker_backend": settings.reranker_backend if settings.reranker_enabled else "none",
        "reranker_url": settings.reranker_url if settings.reranker_enabled else "none",
    }

    if args.tier == "tier1":
        task = make_tier1_task(top_k=candidates)
        evaluators = TIER1_EVALUATORS
        metadata = {**infra_meta, "search_type": search_type, "embedding_model": settings.embedding_model,
                     "reranker": reranker_info, "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "reranker_min_score": settings.reranker_min_score if settings.reranker_enabled else None,
                     "min_confidence_score": settings.min_confidence_score if settings.cosine_floor_applies else None,
                     "candidates": candidates, "tier": "tier1"}
    else:
        task = make_agent_task(verbose=args.verbose)
        evaluators = TIER2_EVALUATORS if args.tier == "tier2" else CHATBOT_EVALUATORS
        # Every knob that changes the result belongs here: an experiment whose
        # parameters are only recoverable from its NAME cannot be compared against
        # another six weeks later. reranker_min_score in particular gates retrieval
        # entirely when it fires, and embedding_model was missing from the agent
        # tiers although it decides what is retrievable at all.
        metadata = {**infra_meta, "llm": settings.llm_model, **_llm_digest_meta(),
                     "search_type": search_type,
                     "embedding_model": settings.embedding_model,
                     "reranker": reranker_info,
                     "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
                     "reranker_candidates": settings.reranker_candidates if settings.reranker_enabled else None,
                     "reranker_min_score": settings.reranker_min_score if settings.reranker_enabled else None,
                     "min_confidence_score": settings.min_confidence_score if settings.cosine_floor_applies else None,
                     "num_ctx": settings.ollama_num_ctx,
                     "temperature": settings.llm_temperature,
                     # The prompt is a parameter like any other, and the one most
                     # likely to be edited between runs. The hash identifies it
                     # exactly (two runs with the same hash used the same prompt,
                     # committed or not); chars and tokens make the context-budget
                     # cost of a prompt change visible in the comparison.
                     **_prompt_meta(), **_mirror_prompt_to_registry(client),
                     "agent_type": "function-agent-toolfree", "candidates": candidates, "tier": args.tier,
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
