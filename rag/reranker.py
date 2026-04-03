"""
Cross-encoder reranker for post-retrieval reranking.

Default model: nvidia/llama-nemotron-rerank-1b-v2
- Uses raw transformers API (NOT sentence-transformers CrossEncoder)
- Prompt template: "question:{query} \n \n passage:{passage}"
- trust_remote_code=True (custom LlamaBidirectional architecture)
- Returns raw logit scores (higher = more relevant)
"""

import logging
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import settings

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def _load_reranker():
    """Lazy-load the reranker model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        if settings.hf_token:
            os.environ["HF_TOKEN"] = settings.hf_token

        logger.info(f"Loading reranker: {settings.reranker_model}")
        _tokenizer = AutoTokenizer.from_pretrained(
            settings.reranker_model,
            trust_remote_code=True,
            padding_side="left",
        )
        _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForSequenceClassification.from_pretrained(
            settings.reranker_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()

        logger.info("Reranker loaded")


def _prompt_template(query: str, passage: str) -> str:
    """Format query and passage with the Nemotron prompt template."""
    return f"question:{query} \n \n passage:{passage}"


def rerank(
    query: str,
    results: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Rerank search results using the cross-encoder.

    Args:
        query: The user's search query.
        results: List of dicts, each must have "text" key.
        top_k: How many to keep after reranking (default: settings.reranker_top_k).

    Returns:
        Reranked list of dicts, trimmed to top_k. Each dict gets:
        - "rerank_score": float (raw logit from cross-encoder)
        - "original_rank": int (position before reranking, 1-indexed)
    """
    if not results:
        return results

    k = top_k or settings.reranker_top_k
    _load_reranker()

    # Build formatted texts using the prompt template
    texts = [_prompt_template(query, r.get("text", "")) for r in results]

    # Tokenize as single sequences (NOT pairs)
    batch_dict = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=8192,
    )

    # Move to same device as model
    device = next(_model.parameters()).device
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    # Score
    with torch.inference_mode():
        logits = _model(**batch_dict).logits
        scores = logits.view(-1).float().tolist()

    # Attach scores and original rank
    for i, (result, score) in enumerate(zip(results, scores)):
        result["rerank_score"] = score
        result["original_rank"] = i + 1

    # Sort by rerank score descending
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    # Log reranking effect
    if reranked:
        top = reranked[0]
        logger.info(
            f"Reranked {len(results)} → top_k={k} | "
            f"Best: score={top['rerank_score']:.3f} (was rank {top['original_rank']}) | "
            f"{top.get('doc_title', '')[:40]}"
        )

    return reranked[:k]
