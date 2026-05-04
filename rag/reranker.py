"""
Reranker module — calls a rerank-compatible endpoint.

Supports three backends:
- llama-server: POST /v1/rerank, simple {model, query, documents, top_n} payload
- vllm:        POST /v1/rerank, query/docs wrapped in Qwen3 chat template
- vllm-score:  POST /v1/score with {model, text_1, text_2}; same wrapping as `vllm`
                (newer vLLM cross-encoder models expose /v1/score instead of /v1/rerank)

The Qwen3 chat-template wrapping (system prompt + <Instruct>/<Query> + <Document>: + think
suffix) is required for both vLLM modes — without it, score discrimination collapses
(e.g. 0.91 vs 0.32 instead of 0.997 vs 0.0003 on a clearly-relevant vs irrelevant doc pair).

Falls back to original ranking if server is unavailable — never blocks the pipeline.
"""

import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


_VLLM_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
)

_VLLM_DOC_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_VLLM_BACKENDS = {"vllm", "vllm-score"}


def _build_query(question: str) -> str:
    """Build the reranker query. Format depends on backend."""
    instruction = settings.reranker_instruction

    if settings.reranker_backend in _VLLM_BACKENDS:
        # vLLM Qwen3-Reranker: full chat template in query string
        return (
            f"<|im_start|>system\n{_VLLM_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {question}\n"
        )

    # llama-server: simple instruction+query template
    template = settings.reranker_query_template
    if not template or template == "{query}":
        return question
    template = template.replace("\\n", "\n")
    return template.format(instruction=instruction, query=question)


def _build_documents(texts: list[str]) -> list[str]:
    """Wrap document texts if backend requires it."""
    if settings.reranker_backend in _VLLM_BACKENDS:
        return [f"<Document>: {t}{_VLLM_DOC_SUFFIX}" for t in texts]
    return texts


def rerank(
    query: str,
    results: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """
    Rerank search results.

    Args:
        query: The user's search query.
        results: List of dicts, each must have "text" key.
        top_n: How many to keep after reranking (default: settings.reranker_top_n).

    Returns:
        Reranked list of dicts, trimmed to top_n. Each dict gets:
        - "rerank_score": float (0.0–1.0 relevance probability)
        - "original_rank": int (position before reranking, 1-indexed)
    """
    if not results:
        return results

    n = top_n or settings.reranker_top_n
    formatted_query = _build_query(query)
    documents = _build_documents([r.get("text", "") for r in results])

    # Tag original ranks before reranking
    for i, r in enumerate(results):
        r["original_rank"] = i + 1

    try:
        if settings.reranker_backend == "vllm-score":
            scored = _call_score(formatted_query, documents)
        else:
            scored = _call_rerank(formatted_query, documents, n)

        # Sort by score desc and trim to top_n (score endpoint preserves input order)
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:n]

        reranked = []
        for idx, score in scored:
            result = results[idx].copy()
            result["rerank_score"] = score
            reranked.append(result)

        if reranked:
            top = reranked[0]
            logger.info(
                f"Reranked {len(results)} → {len(reranked)} | "
                f"Best: score={top['rerank_score']:.4f} (was rank {top['original_rank']}) | "
                f"{top.get('doc_title', '')[:40]}"
            )

        return reranked

    except httpx.ConnectError:
        logger.warning(f"Reranker unavailable at {settings.reranker_url} — using original ranking")
        return results[:n]
    except httpx.TimeoutException:
        logger.warning("Reranker timed out — using original ranking")
        return results[:n]
    except Exception as e:
        logger.warning(f"Reranker error: {e} — using original ranking")
        return results[:n]


def _call_rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """POST /v1/rerank — returns list of (index, relevance_score)."""
    response = httpx.post(
        f"{settings.reranker_url}/v1/rerank",
        json={
            "model": settings.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
        timeout=20.0,
    )
    response.raise_for_status()
    data = response.json()
    return [(item["index"], item["relevance_score"]) for item in data["results"]]


def _call_score(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """POST /v1/score — returns list of (index, score)."""
    response = httpx.post(
        f"{settings.reranker_url}/v1/score",
        json={
            "model": settings.reranker_model,
            "text_1": query,
            "text_2": documents,
        },
        timeout=20.0,
    )
    response.raise_for_status()
    data = response.json()
    return [(item["index"], item["score"]) for item in data["data"]]
