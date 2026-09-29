import asyncio

from llama_index.core.tools import FunctionTool

from config import settings
from rag.embeddings import embed_query
from rag.observability import record_floor_rejection, record_infra_unavailable
from rag.reranker import rerank
from rag.resilience import RETRY_BACKOFFS, is_transient, retry_transient
from rag.vector_store import search_chunks

_last_search_results: list[dict] = []
_retrieval_unavailable: bool = False

NO_MATCH = "NO_RELEVANT_POLICY_FOUND"
UNAVAILABLE = "POLICY_SEARCH_UNAVAILABLE"


def search_policies(query: str, top_k: int = 6) -> str:
    """
    Search approved compliance policy documents for information relevant to the query.
    Uses hybrid search (semantic + keyword matching) for best accuracy.

    Args:
        query: Natural language search query describing what policy info you need.
               Be specific — include relevant terms, clause numbers, or policy names.
        top_k: Number of most relevant policy sections to return (default 6).

    Returns:
        Formatted policy excerpts with document name, section, clause, clause number,
        and full text. Returns "NO_RELEVANT_POLICY_FOUND" if no policies match.
    """
    global _last_search_results, _retrieval_unavailable
    _retrieval_unavailable = False

    # How many candidates to retrieve (more when the reranker will rescore)
    retrieve_k = settings.reranker_candidates if settings.reranker_enabled else top_k

    # Step 1: Retrieve candidates — dense only, or dense + sparse fused by Qdrant.
    # One path for both: fusion is server-side, so both return ScoredPoints.
    try:
        query_vector = retry_transient(lambda: embed_query(query))
    except Exception as exc:
        if is_transient(exc):
            _last_search_results = []
            _retrieval_unavailable = True
            record_infra_unavailable("embeddings", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    try:
        raw = retry_transient(lambda: search_chunks(query, query_vector, top_k=retrieve_k))
    except Exception as exc:
        if is_transient(exc):
            _last_search_results = []
            _retrieval_unavailable = True
            record_infra_unavailable("qdrant", type(exc).__name__, len(RETRY_BACKOFFS))
            return UNAVAILABLE
        raise

    if not raw:
        _last_search_results = []
        return NO_MATCH

    # min_confidence_score is a COSINE threshold (0.45). It may only judge a score
    # that IS a cosine similarity — reranker off AND no RRF fusion. An RRF score is
    # ~0.016 (1/(k + rank), with k pinned to 60 in rag/vector_store.py), so dropping
    # the bm25 half of this condition would return NO_MATCH for every question in the
    # corpus, with no error raised anywhere. That total failure is the diagnosable
    # one: at Qdrant's default k of 2, RRF scores land in the same range as cosine
    # ones and the same mistake would fail only for some questions. Same class of bug
    # as the rerank_score-presence guard in Step 3b: the guard is on what the number
    # MEANS, not on which component produced it.
    if (
        not settings.reranker_enabled
        and not settings.bm25_enabled
        and raw[0].score < settings.min_confidence_score
    ):
        _last_search_results = []
        return NO_MATCH

    results = [
        {
            "doc_title": r.payload["doc_title"],
            "doc_id": r.payload["doc_id"],
            "section": r.payload.get("section", ""),
            "clause": r.payload.get("clause", ""),
            "clause_number": r.payload.get("clause_number", ""),
            "text": r.payload["text"],
            "retrieval_score": r.score,
            "score_type": "rrf" if settings.bm25_enabled else "cosine",
        }
        for r in raw
    ]

    # Step 2: Rerank (if enabled)
    if settings.reranker_enabled and results:
        results = rerank(query, results, top_n=settings.reranker_top_n)

    # Step 3: Capture structured results for eval logging.
    # score_type travels with retrieval_score because it is that number's unit:
    # 0.016 (RRF) and 0.85 (cosine) are not comparable, and this dict is read by
    # people, in eval result JSON, often long after the run. Recording the number
    # without its scale is how a score gets judged against the wrong threshold —
    # the failure this file's min_confidence_score guard exists to prevent.
    _last_search_results = [
        {
            "doc_title": r["doc_title"],
            "section": r.get("section", ""),
            "clause": r.get("clause", ""),
            "clause_number": r.get("clause_number", ""),
            "rerank_score": round(r.get("rerank_score", 0), 4),
            "retrieval_score": round(r.get("retrieval_score", 0), 4),
            "score_type": r.get("score_type", ""),
        }
        for r in results
    ]

    # Step 3b: Relevance floor (spec D8).
    # Guarded on the key being PRESENT, not on its value: the reranker's fallback
    # path returns results with no rerank_score at all, and defaulting that to 0.0
    # would reject every question the moment the reranker degraded.
    if settings.reranker_min_score > 0 and results and "rerank_score" in results[0]:
        top_score = results[0]["rerank_score"]
        if top_score < settings.reranker_min_score:
            record_floor_rejection(top_score, settings.reranker_min_score)
            # _last_search_results deliberately left populated — see
            # test_a_rejected_search_still_reports_what_it_found.
            return NO_MATCH

    # Step 4: Format for the agent
    return format_sources(results)


def format_sources(search_results: list[dict]) -> str:
    """Format search results for the agent. No scores, no doc_id — just policy content."""
    if not search_results:
        return f"=== RETRIEVED POLICY SOURCES ===\n\n{NO_MATCH}"

    lines = ["=== RETRIEVED POLICY SOURCES ==="]

    for i, r in enumerate(search_results):
        lines.append("")  # blank line between sources
        lines.append(f"[Source {i + 1}] {r['doc_title']}")
        lines.append(f"Section: {r['section']}")
        if r.get("clause_number"):
            lines.append(f"Clause Number: {r['clause_number']}")
        if r.get("clause"):
            lines.append(f"Clause Name: {r['clause']}")
        lines.append("---")
        lines.append(r["text"])

    return "\n".join(lines)


async def _search_policies_async(query: str, top_k: int = 6) -> str:
    """Async companion to `search_policies`, run via `asyncio.to_thread`.

    Why this exists — do not delete it as redundant indirection: without an
    explicit `async_fn`, `FunctionTool.from_defaults` builds its own async
    wrapper around the sync `search_policies` using `loop.run_in_executor`
    directly, which does NOT propagate `contextvars`. Every OTel span opened
    inside the tool (embed_query, search_vectors, rerank) would then start
    with an empty context and come out as its own disconnected root trace
    instead of nesting under the agent's tool-call span — exactly the bug
    this file was patched to fix. `asyncio.to_thread` copies the current
    context (`contextvars.copy_context()`) before handing the call to the
    same default executor, so the spans nest correctly, with no change in
    concurrency (still one pooled thread, not parallel execution).
    """
    return await asyncio.to_thread(search_policies, query, top_k)


# `fn` (not `async_fn`) is what the tool schema/description is derived from, so the
# LLM-facing signature `search_policies(query, top_k=6)` referenced in rag/agent.py's
# system prompt is unchanged. `async_fn` only swaps out which coroutine actually runs.
search_policies_tool = FunctionTool.from_defaults(
    fn=search_policies,
    async_fn=_search_policies_async,
)
