"""Pre-retrieval input classification: greeting / in_scope / out_of_scope / unintelligible.

A single temperature-0 LLM call decides whether a message reaches policy search. The
classifier can only ADD short-circuits for high-confidence non-questions; it can never
refuse a real question — uncertainty or failure resolves to IN_SCOPE (see resolve()).
"""

import json
import logging
from enum import Enum

from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, ConfigDict, ValidationError

from config import settings
from rag.agent import get_llm
from rag.resilience import retry_transient
from rag.response import _extract_json  # reuse the tolerant JSON extractor (DRY)

logger = logging.getLogger(__name__)


class Category(str, Enum):
    IN_SCOPE = "in_scope"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"
    UNINTELLIGIBLE = "unintelligible"


class RouterDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: Category
    confidence: float
    fallback: bool = False  # True when set by the failure path, not the model


# Editable tuning surface for classifier behavior (see spec "Tuning surface").
ROUTER_SYSTEM_PROMPT = """\
You are an input classifier for an internal Compliance Policy assistant.
Classify the user's message into EXACTLY ONE category and output ONLY a JSON object.

Categories:
- "in_scope": a question an internal company compliance or policy document could plausibly
  answer (security, HR, data handling, device/access, conduct, travel, expenses, etc.).
  When unsure whether a message is a real policy question, choose "in_scope".
- "greeting": a greeting, pleasantry, or small talk with no question
  (e.g. "hi", "hello", "good morning", "thanks", "how are you").
- "out_of_scope": an intelligible request or question that company policy would NOT cover
  (e.g. "order me a pizza", "what's the weather", "who is Sarah Connor", general knowledge).
- "unintelligible": text that cannot be read as a meaningful message — random characters,
  gibberish, or text typed with the wrong keyboard layout (e.g. Cyrillic characters that are
  English words typed on a Ukrainian/Russian layout like "црфе ші").

Output EXACTLY one JSON object and nothing else:
{"category": "in_scope|greeting|out_of_scope|unintelligible", "confidence": <number 0..1>}

confidence is your certainty in the chosen category; use a low value when the message is ambiguous."""

_FALLBACK = RouterDecision(category=Category.IN_SCOPE, confidence=0.0, fallback=True)


def _parse_decision(raw: str) -> RouterDecision | None:
    """Parse the classifier's JSON output; return None if unparseable or invalid."""
    json_str = _extract_json(raw)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
        return RouterDecision(category=data["category"], confidence=float(data["confidence"]))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, ValidationError):
        return None


def classify_message(text: str) -> RouterDecision:
    """Classify a message. Never raises; any failure/unparseable output -> IN_SCOPE fallback."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=ROUTER_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=text),
    ]
    try:
        llm = get_llm(settings.router_llm_model or None)
        response = retry_transient(lambda: llm.chat(messages))
        decision = _parse_decision(str(response.message.content))
        if decision is None:
            logger.warning("classifier output unparseable, defaulting to in_scope")
            return _FALLBACK
        return decision
    except Exception as exc:
        # Fail safe to search: a classifier problem must never block a real question.
        logger.warning("classification failed, defaulting to in_scope: %s", exc)
        return _FALLBACK


def resolve(decision: RouterDecision, floor: float) -> Category:
    """Apply the safe-default bias: failure or low confidence -> IN_SCOPE."""
    if decision.fallback:
        return Category.IN_SCOPE
    if decision.confidence < floor:
        return Category.IN_SCOPE
    return decision.category
