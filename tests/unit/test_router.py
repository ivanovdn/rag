import httpx
import pytest

from rag.router import (
    Category,
    RouterDecision,
    classify_message,
    resolve,
    _parse_decision,
)


# --- resolve (pure) ---

@pytest.mark.parametrize("cat", list(Category))
def test_resolve_below_floor_is_in_scope(cat):
    d = RouterDecision(category=cat, confidence=0.4)
    assert resolve(d, 0.6) == Category.IN_SCOPE


def test_resolve_fallback_is_in_scope_regardless_of_confidence():
    d = RouterDecision(category=Category.OUT_OF_SCOPE, confidence=0.99, fallback=True)
    assert resolve(d, 0.6) == Category.IN_SCOPE


def test_resolve_at_or_above_floor_passes_category_through():
    d = RouterDecision(category=Category.GREETING, confidence=0.6)
    assert resolve(d, 0.6) == Category.GREETING
    d2 = RouterDecision(category=Category.OUT_OF_SCOPE, confidence=0.9)
    assert resolve(d2, 0.6) == Category.OUT_OF_SCOPE


# --- _parse_decision ---

def test_parse_plain_json():
    d = _parse_decision('{"category": "greeting", "confidence": 0.9}')
    assert d.category == Category.GREETING and d.confidence == 0.9


def test_parse_fenced_json():
    d = _parse_decision('```json\n{"category": "out_of_scope", "confidence": 0.8}\n```')
    assert d.category == Category.OUT_OF_SCOPE


def test_parse_unknown_category_is_none():
    assert _parse_decision('{"category": "banana", "confidence": 0.9}') is None


def test_parse_missing_confidence_is_none():
    assert _parse_decision('{"category": "greeting"}') is None


def test_parse_non_json_is_none():
    assert _parse_decision("I think this is a greeting") is None


# --- classify_message (LLM mocked) ---

class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeResp:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeLLM:
    def __init__(self, content=None, exc=None):
        self._content, self._exc = content, exc

    def chat(self, messages):
        if self._exc is not None:
            raise self._exc
        return _FakeResp(self._content)


def test_classify_success_returns_parsed_decision(monkeypatch):
    monkeypatch.setattr(
        "rag.router.get_llm",
        lambda model=None: _FakeLLM(content='{"category": "greeting", "confidence": 0.95}'),
    )
    d = classify_message("hi there")
    assert d.category == Category.GREETING and d.confidence == 0.95 and d.fallback is False


def test_classify_unparseable_falls_back_to_in_scope(monkeypatch):
    monkeypatch.setattr(
        "rag.router.get_llm", lambda model=None: _FakeLLM(content="not json at all")
    )
    d = classify_message("hi there")
    assert d.category == Category.IN_SCOPE and d.fallback is True


def test_classify_llm_failure_falls_back_to_in_scope(monkeypatch):
    monkeypatch.setattr("rag.resilience.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "rag.router.get_llm",
        lambda model=None: _FakeLLM(exc=httpx.ConnectError("llm down")),
    )
    d = classify_message("Can I install software?")
    assert d.category == Category.IN_SCOPE and d.fallback is True
