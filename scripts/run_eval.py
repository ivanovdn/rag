"""
Evaluation harness for the Compliance Q&A Bot.

Runs test datasets against the retrieval pipeline and/or full agent,
computes metrics, and logs everything to Phoenix.

Usage:
    python scripts/run_eval.py --tier retrieval  --tag "baseline"
    python scripts/run_eval.py --tier e2e        --tag "baseline"
    python scripts/run_eval.py --tier escalation --tag "baseline"
    python scripts/run_eval.py --tier chatbot    --tag "baseline"
    python scripts/run_eval.py --tier all        --tag "baseline"
"""

# Phoenix must be initialized before any LlamaIndex imports
from rag.observability import init_observability

init_observability()

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from config import settings
from rag.observability import get_tracer
from rag.search_first import compose_agent_input, prefetch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("eval")

RESULTS_DIR = Path("eval/results")
DATASETS_DIR = Path(settings.eval_dataset_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_results(data: dict, tier: str, tag: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{tier}_{tag}_{ts}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def config_snapshot() -> dict:
    return {
        "bm25_enabled": settings.bm25_enabled,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "min_confidence_score": settings.min_confidence_score,
        "retrieval_top_k": settings.retrieval_top_k,
        "hybrid_vector_candidates": settings.hybrid_vector_candidates,
        "hybrid_bm25_candidates": settings.hybrid_bm25_candidates,
    }



def similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fact_coverage(expected, actual: str) -> float:
    """
    If expected is a list, return fraction of items found in actual.
    If expected is a string, return SequenceMatcher similarity.
    """
    if isinstance(expected, list):
        if not expected:
            return 1.0
        hits = sum(1 for item in expected if item.lower() in actual.lower())
        return hits / len(expected)
    return similarity(expected, actual)


async def run_agent_query(agent, agent_input: str) -> str:
    """Run a (question + retrieved sources) input through the agent and return the response string."""
    response = await agent.run(user_msg=agent_input)
    return str(response)


# ---------------------------------------------------------------------------
# Tier 1: Retrieval Evaluation
# ---------------------------------------------------------------------------


def run_retrieval_eval(dataset_path: Path, tag: str) -> dict:
    from rag.embeddings import embed_query
    from rag.vector_store import search_vectors

    tracer = get_tracer()
    data = load_dataset(dataset_path)
    test_cases = data["test_cases"]
    top_k = settings.retrieval_top_k

    results = []
    hits = 0
    reciprocal_ranks = []
    top_scores = []

    logger.info(f"Running retrieval eval: {len(test_cases)} cases, top_k={top_k}")

    for tc in test_cases:
        with tracer.start_as_current_span("eval.retrieval") as span:
            span.set_attribute("eval.test_id", tc["id"])
            span.set_attribute("eval.question", tc["question"])

            # Run search
            if settings.bm25_enabled:
                from rag.hybrid_search import hybrid_search

                raw_results = hybrid_search(tc["question"], top_k=top_k)
                search_results = []
                for r in raw_results:
                    search_results.append(
                        {
                            "doc_id": r["doc_id"],
                            "doc_title": r["doc_title"],
                            "section": r.get("section", ""),
                            "section_number": r.get("section_number", ""),
                            "clause": r.get("clause", ""),
                            "clause_number": r.get("clause_number", ""),
                            "section_display": r.get("section_display", ""),
                            "text": r["text"],
                            "score": r["rrf_score"],
                        }
                    )
            else:
                vector = embed_query(tc["question"])
                raw_results = search_vectors(vector, top_k=top_k)
                search_results = []
                for r in raw_results:
                    p = r.payload
                    search_results.append(
                        {
                            "doc_id": p.get("doc_id", ""),
                            "doc_title": p.get("doc_title", ""),
                            "section": p.get("section", ""),
                            "section_number": p.get("section_number", ""),
                            "clause": p.get("clause", ""),
                            "clause_number": p.get("clause_number", ""),
                            "section_display": p.get("section_display", ""),
                            "text": p.get("text", ""),
                            "score": r.score,
                        }
                    )

            top_score = search_results[0]["score"] if search_results else 0.0
            top_scores.append(top_score)

            # Check for hit
            hit = False
            hit_rank = None
            expected_doc = tc.get("expected_doc_id", "").strip()
            expected_section = tc.get("expected_section_contains", "").strip()
            expected_clause = tc.get("expected_clause", "").strip()

            for rank, sr in enumerate(search_results, start=1):
                # Document: case-insensitive exact match against doc_title
                doc_match = (
                    not expected_doc
                    or expected_doc.lower() == sr.get("doc_title", "").lower()
                )

                # Section: match against section name
                section_match = (
                    not expected_section
                    or expected_section.lower() in sr.get("section", "").lower()
                )

                # Clause: match against clause name
                clause_match = (
                    not expected_clause
                    or expected_clause.lower() in sr.get("clause", "").lower()
                )

                if doc_match and section_match and clause_match:
                    hit = True
                    hit_rank = rank
                    break

            if hit:
                hits += 1
                reciprocal_ranks.append(1.0 / hit_rank)
            else:
                reciprocal_ranks.append(0.0)

            span.set_attribute("eval.hit", hit)
            span.set_attribute("eval.hit_rank", hit_rank or 0)
            span.set_attribute("eval.top_score", top_score)
            span.set_attribute("eval.expected_doc", expected_doc)
            span.set_attribute("eval.expected_section", expected_section)
            span.set_attribute("eval.expected_clause", expected_clause)

            # Log ALL returned results so you can inspect in Phoenix
            for ri, sr in enumerate(search_results):
                span.set_attribute(f"eval.result.{ri}.doc_title", sr.get("doc_title", ""))
                span.set_attribute(f"eval.result.{ri}.section", sr.get("section", ""))
                span.set_attribute(f"eval.result.{ri}.clause", sr.get("clause", ""))
                span.set_attribute(f"eval.result.{ri}.score", sr.get("score", 0.0))

            results.append(
                {
                    "id": tc["id"],
                    "question": tc["question"],
                    "hit": hit,
                    "hit_rank": hit_rank,
                    "top_score": top_score,
                    "expected_doc": expected_doc,
                    "expected_section": expected_section,
                    "expected_clause": expected_clause,
                    "matched_doc": search_results[hit_rank - 1]["doc_title"] if hit else "",
                    "matched_section": search_results[hit_rank - 1].get("section", "") if hit else "",
                    "matched_clause": search_results[hit_rank - 1].get("clause", "") if hit else "",
                    "search_results": [
                        {
                            "rank": ri + 1,
                            "doc_title": sr.get("doc_title", ""),
                            "section": sr.get("section", ""),
                            "clause": sr.get("clause", ""),
                            "score": round(sr.get("score", 0.0), 4),
                        }
                        for ri, sr in enumerate(search_results)
                    ],
                }
            )

            if hit:
                matched = search_results[hit_rank - 1]
                logger.info(
                    f"  [{tc['id']}] HIT@{hit_rank} (score={top_score:.3f}) "
                    f"doc={matched.get('doc_title', '')[:30]} | sec={matched.get('section', '')[:20]} | cls={matched.get('clause', '')[:20]}"
                )
            else:
                top = search_results[0] if search_results else {}
                logger.info(
                    f"  [{tc['id']}] MISS (score={top_score:.3f}) "
                    f"expected: doc={expected_doc[:30]} sec={expected_section[:20]} cls={expected_clause[:20]} | "
                    f"got: doc={top.get('doc_title', '')[:30]} sec={top.get('section', '')[:20]} cls={top.get('clause', '')[:20]}"
                )

    # Compute metrics
    n = len(test_cases)
    metrics = {
        "hit_rate_at_k": hits / n if n else 0,
        "mrr": sum(reciprocal_ranks) / n if n else 0,
        "avg_top_score": sum(top_scores) / n if n else 0,
        "total_cases": n,
        "hits": hits,
        "misses": n - hits,
    }

    output = {
        "metadata": {
            "tier": "retrieval",
            "tag": tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config_snapshot(),
            "dataset": str(dataset_path),
        },
        "metrics": metrics,
        "results": results,
    }

    path = save_results(output, "retrieval", tag)

    print("\n=== Retrieval Evaluation Results ===")
    print(f"  Hit Rate@{top_k}:  {metrics['hit_rate_at_k']:.1%} ({hits}/{n})")
    print(f"  MRR:              {metrics['mrr']:.3f}")
    print(f"  Avg Top Score:    {metrics['avg_top_score']:.3f}")
    print(f"  Results saved to: {path}")

    return output


# ---------------------------------------------------------------------------
# Tier 2: End-to-End Evaluation
# ---------------------------------------------------------------------------


async def run_e2e_eval(dataset_path: Path, tag: str) -> dict:
    from rag.agent import build_agent

    tracer = get_tracer()
    data = load_dataset(dataset_path)
    test_cases = data["test_cases"]

    results = []
    citation_correct_count = 0
    fact_coverages = []
    latencies = []
    unavailable_count = 0

    logger.info(f"Running e2e eval: {len(test_cases)} cases")

    for tc in test_cases:
        with tracer.start_as_current_span("eval.e2e") as span:
            span.set_attribute("eval.test_id", tc["id"])
            span.set_attribute("eval.question", tc["question"])

            # Timer starts before prefetch: retrieval used to run INSIDE
            # agent.run, so avg_latency_seconds included it. It still must,
            # now that retrieval happens in front of the agent instead of
            # inside it — otherwise the metric quietly narrows to
            # agent-only time and a comparison against an older run reads as
            # a speedup that never happened.
            start = time.time()
            pre = prefetch(tc["question"])
            if pre.status == "unavailable":
                unavailable_count += 1
                logger.info(f"  [{tc['id']}] UNAVAILABLE; retrieval backend down, case dropped")
                continue

            # "no_match" is a real outcome — retrieval ran and correctly found
            # nothing relevant. Score it as a no-answer, like production does
            # (_run_rag short-circuits the same way), instead of dropping it.
            if pre.status == "no_match":
                answer = "No relevant policy was found for this question."
            else:
                agent = build_agent()
                answer = await run_agent_query(agent, compose_agent_input(tc["question"], pre.sources))
            latency = time.time() - start
            latencies.append(latency)

            # Check citation accuracy
            citation_correct = False
            expected_cits = tc.get("expected_citations", [])
            if expected_cits:
                answer_lower = answer.lower()
                for cit in expected_cits:
                    doc_id = cit.get("doc_id", "")
                    # Check if doc name words appear in answer
                    doc_words = [
                        w
                        for w in doc_id.lower().replace("-", " ").split()
                        if len(w) > 3
                    ]
                    if doc_words and any(w in answer_lower for w in doc_words):
                        citation_correct = True
                        break
            else:
                citation_correct = True  # no citation requirement

            if citation_correct:
                citation_correct_count += 1

            # Fact coverage
            fc = fact_coverage(tc.get("expected_answer", ""), answer)
            fact_coverages.append(fc)

            span.set_attribute("eval.citation_correct", citation_correct)
            span.set_attribute("eval.fact_coverage", fc)
            span.set_attribute("eval.latency_seconds", latency)

            results.append(
                {
                    "id": tc["id"],
                    "question": tc["question"],
                    "answer": answer[:500],
                    "citation_correct": citation_correct,
                    "fact_coverage": round(fc, 3),
                    "latency_seconds": round(latency, 2),
                }
            )

            logger.info(
                f"  [{tc['id']}] cit={'OK' if citation_correct else 'MISS'} "
                f"fc={fc:.2f} latency={latency:.1f}s"
            )

    n = len(test_cases)
    n_scored = n - unavailable_count
    metrics = {
        "citation_accuracy": citation_correct_count / n_scored if n_scored else 0,
        "fact_coverage": sum(fact_coverages) / n_scored if n_scored else 0,
        "avg_latency_seconds": sum(latencies) / n_scored if n_scored else 0,
        "total_cases": n,
        "unavailable_cases": unavailable_count,
    }

    output = {
        "metadata": {
            "tier": "e2e",
            "tag": tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config_snapshot(),
            "dataset": str(dataset_path),
        },
        "metrics": metrics,
        "results": results,
    }

    path = save_results(output, "e2e", tag)

    print("\n=== End-to-End Evaluation Results ===")
    print(f"  Unavailable:       {unavailable_count}/{n} (dropped, not scored)")
    print(f"  Citation Accuracy: {metrics['citation_accuracy']:.1%}")
    print(f"  Fact Coverage:     {metrics['fact_coverage']:.1%}")
    print(f"  Avg Latency:       {metrics['avg_latency_seconds']:.1f}s")
    print(f"  Results saved to:  {path}")

    return output


# ---------------------------------------------------------------------------
# Tier 3: Escalation Evaluation
# ---------------------------------------------------------------------------


async def run_escalation_eval(dataset_path: Path, tag: str) -> dict:
    from rag.agent import build_agent

    tracer = get_tracer()
    data = load_dataset(dataset_path)
    test_cases = data["test_cases"]

    results = []
    correct_escalations = 0
    false_answers = 0
    unavailable_count = 0

    escalation_markers = [
        "escalat",
        "esc-",
        "unable to find",
        "cannot confirm",
        "forwarded to",
        "compliance team",
        "no relevant policy",
        "could not find",
    ]

    logger.info(f"Running escalation eval: {len(test_cases)} cases")

    for tc in test_cases:
        with tracer.start_as_current_span("eval.escalation") as span:
            span.set_attribute("eval.test_id", tc["id"])
            span.set_attribute("eval.question", tc["question"])

            pre = prefetch(tc["question"])
            if pre.status == "unavailable":
                unavailable_count += 1
                logger.info(f"  [{tc['id']}] UNAVAILABLE; retrieval backend down, case dropped")
                continue

            # "no_match" IS the escalation outcome here — retrieval ran and
            # correctly found nothing relevant, which is exactly what this
            # tier exists to measure. Score it against should_escalate like
            # any other case, without ever building or calling the agent.
            if pre.status == "no_match":
                answer = "No relevant policy was found for this question."
                was_escalated = True
            else:
                agent = build_agent()
                answer = await run_agent_query(agent, compose_agent_input(tc["question"], pre.sources))
                was_escalated = any(m in answer.lower() for m in escalation_markers)

            correctly_escalated = was_escalated == tc.get("should_escalate", True)
            false_answer = not was_escalated and tc.get("should_escalate", True)

            if correctly_escalated:
                correct_escalations += 1
            if false_answer:
                false_answers += 1

            span.set_attribute("eval.was_escalated", was_escalated)
            span.set_attribute("eval.correctly_escalated", correctly_escalated)
            span.set_attribute("eval.false_answer", false_answer)

            results.append(
                {
                    "id": tc["id"],
                    "question": tc["question"],
                    "answer": answer[:500],
                    "was_escalated": was_escalated,
                    "correctly_escalated": correctly_escalated,
                    "false_answer": false_answer,
                    "category": tc.get("category", ""),
                }
            )

            status = "ESCALATED" if was_escalated else "ANSWERED"
            ok = "OK" if correctly_escalated else "FAIL"
            logger.info(f"  [{tc['id']}] {status} ({ok}) - {tc.get('category', '')}")

    n = len(test_cases)
    n_scored = n - unavailable_count
    metrics = {
        "correct_escalation_rate": correct_escalations / n_scored if n_scored else 0,
        "false_answer_rate": false_answers / n_scored if n_scored else 0,
        "total_cases": n,
        "unavailable_cases": unavailable_count,
        "correct_escalations": correct_escalations,
        "false_answers": false_answers,
    }

    output = {
        "metadata": {
            "tier": "escalation",
            "tag": tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config_snapshot(),
            "dataset": str(dataset_path),
        },
        "metrics": metrics,
        "results": results,
    }

    path = save_results(output, "escalation", tag)

    print("\n=== Escalation Evaluation Results ===")
    print(f"  Unavailable:             {unavailable_count}/{n} (dropped, not scored)")
    print(
        f"  Correct Escalation Rate: {metrics['correct_escalation_rate']:.1%} ({correct_escalations}/{n_scored})"
    )
    print(f"  False Answer Rate:       {metrics['false_answer_rate']:.1%} ({false_answers}/{n_scored})")
    print(f"  Results saved to:        {path}")

    return output


# ---------------------------------------------------------------------------
# Tier 4: Chatbot Evaluation (same format as E2E)
# ---------------------------------------------------------------------------


async def run_chatbot_eval(dataset_path: Path, tag: str) -> dict:
    from rag.agent import build_agent

    tracer = get_tracer()
    data = load_dataset(dataset_path)
    test_cases = data["test_cases"]

    results = []
    citation_correct_count = 0
    fact_coverages = []
    latencies = []
    unavailable_count = 0

    logger.info(f"Running chatbot eval: {len(test_cases)} cases")

    for tc in test_cases:
        with tracer.start_as_current_span("eval.chatbot") as span:
            span.set_attribute("eval.test_id", tc["id"])
            span.set_attribute("eval.question", tc["question"])

            # Timer starts before prefetch — see run_e2e_eval for why: it must
            # span the whole path a user waits on, not just the agent call.
            start = time.time()
            pre = prefetch(tc["question"])
            if pre.status == "unavailable":
                unavailable_count += 1
                logger.info(f"  [{tc['id']}] UNAVAILABLE; retrieval backend down, case dropped")
                continue

            # "no_match" is a real outcome — see run_e2e_eval.
            if pre.status == "no_match":
                answer = "No relevant policy was found for this question."
            else:
                agent = build_agent()
                answer = await run_agent_query(agent, compose_agent_input(tc["question"], pre.sources))
            latency = time.time() - start
            latencies.append(latency)

            # Check citation accuracy
            citation_correct = False
            expected_cits = tc.get("expected_citations", [])
            if expected_cits:
                answer_lower = answer.lower()
                for cit in expected_cits:
                    doc_id = cit.get("doc_id", "")
                    doc_words = [
                        w
                        for w in doc_id.lower().replace("-", " ").split()
                        if len(w) > 3
                    ]
                    if doc_words and any(w in answer_lower for w in doc_words):
                        citation_correct = True
                        break
            else:
                citation_correct = True

            if citation_correct:
                citation_correct_count += 1

            # Fact coverage
            fc = fact_coverage(tc.get("expected_answer", ""), answer)
            fact_coverages.append(fc)

            span.set_attribute("eval.citation_correct", citation_correct)
            span.set_attribute("eval.fact_coverage", fc)
            span.set_attribute("eval.latency_seconds", latency)

            results.append(
                {
                    "id": tc["id"],
                    "question": tc["question"],
                    "answer": answer[:500],
                    "citation_correct": citation_correct,
                    "fact_coverage": round(fc, 3),
                    "latency_seconds": round(latency, 2),
                }
            )

            logger.info(
                f"  [{tc['id']}] cit={'OK' if citation_correct else 'MISS'} "
                f"fc={fc:.2f} latency={latency:.1f}s"
            )

    n = len(test_cases)
    n_scored = n - unavailable_count
    metrics = {
        "citation_accuracy": citation_correct_count / n_scored if n_scored else 0,
        "fact_coverage": sum(fact_coverages) / n_scored if n_scored else 0,
        "avg_latency_seconds": sum(latencies) / n_scored if n_scored else 0,
        "total_cases": n,
        "unavailable_cases": unavailable_count,
    }

    output = {
        "metadata": {
            "tier": "chatbot",
            "tag": tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config_snapshot(),
            "dataset": str(dataset_path),
        },
        "metrics": metrics,
        "results": results,
    }

    path = save_results(output, "chatbot", tag)

    print("\n=== Chatbot Evaluation Results ===")
    print(f"  Unavailable:       {unavailable_count}/{n} (dropped, not scored)")
    print(f"  Citation Accuracy: {metrics['citation_accuracy']:.1%}")
    print(f"  Fact Coverage:     {metrics['fact_coverage']:.1%}")
    print(f"  Avg Latency:       {metrics['avg_latency_seconds']:.1f}s")
    print(f"  Results saved to:  {path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compliance Bot Evaluation Harness")
    parser.add_argument(
        "--tier",
        required=True,
        choices=["retrieval", "e2e", "escalation", "chatbot", "all"],
        help="Which evaluation tier to run",
    )
    parser.add_argument(
        "--tag", default="eval", help="Tag for this eval run (e.g., 'baseline', 'hybrid-v1')"
    )
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    args = parser.parse_args()

    tiers = (
        ["retrieval", "e2e", "escalation", "chatbot"]
        if args.tier == "all"
        else [args.tier]
    )

    default_datasets = {
        "retrieval": DATASETS_DIR / "retrieval_test.json",
        "e2e": DATASETS_DIR / "e2e_test.json",
        "escalation": DATASETS_DIR / "escalation_test.json",
        "chatbot": DATASETS_DIR / "chatbot_test_cases.json",
    }

    for tier in tiers:
        dataset_path = Path(args.dataset) if args.dataset else default_datasets[tier]
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Running {tier} evaluation (tag={args.tag})")
        print(f"Dataset: {dataset_path}")
        print(f"{'='*50}")

        if tier == "retrieval":
            run_retrieval_eval(dataset_path, args.tag)
        elif tier == "e2e":
            asyncio.run(run_e2e_eval(dataset_path, args.tag))
        elif tier == "escalation":
            asyncio.run(run_escalation_eval(dataset_path, args.tag))
        elif tier == "chatbot":
            asyncio.run(run_chatbot_eval(dataset_path, args.tag))


if __name__ == "__main__":
    main()
