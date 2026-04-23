"""
Reranker module — calls any /v1/rerank-compatible endpoint (llama-server, vLLM, etc.)

Query template is configurable via RERANKER_QUERY_TEMPLATE.
API: POST /v1/rerank with {model, query, documents, top_n}
Response: {results: [{index, relevance_score}, ...]} sorted by score desc

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


def _build_query(question: str) -> str:
    """Build the reranker query. Format depends on backend."""
    instruction = settings.reranker_instruction

    if settings.reranker_backend == "vllm":
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
    if settings.reranker_backend == "vllm":
        return [f"<Document>: {t}{_VLLM_DOC_SUFFIX}" for t in texts]
    return texts


def rerank(
    query: str,
    results: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """
    Rerank search results via llama-server /v1/rerank.

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
        response = httpx.post(
            f"{settings.reranker_url}/v1/rerank",
            json={
                "model": settings.reranker_model,
                "query": formatted_query,
                "documents": documents,
                "top_n": n,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        data = response.json()

        # Map scores back to results
        reranked = []
        for item in data["results"]:
            idx = item["index"]
            result = results[idx].copy()
            result["rerank_score"] = item["relevance_score"]
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
