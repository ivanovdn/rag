"""The rewrite step: parse defensively, never block, never touch the network here."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rag.query_rewrite as qr

Q = "Can I disable the 5-minute auto-lock on my computer?"


# --- parsing --------------------------------------------------------------------


def test_parse_strips_markers_quotes_and_the_original():
    raw = f'1. "{Q}"\n2) Are Team Members permitted to disable automatic screen lock?\n- may a user override the inactivity lockout setting'
    assert qr.parse_rephrasings(raw, Q) == (
        "Are Team Members permitted to disable automatic screen lock?",
        "may a user override the inactivity lockout setting",
    )


def test_parse_drops_preamble_and_caps_at_two():
    raw = "Here are two versions:\nfirst version\nsecond version\nthird version"
    assert qr.parse_rephrasings(raw, Q) == ("first version", "second version")


def test_parse_deduplicates_case_insensitively():
    assert qr.parse_rephrasings("Same thing\nsame thing\n", Q) == ("Same thing",)


def test_parse_of_nothing_usable_is_empty():
    assert qr.parse_rephrasings("\n  \n", Q) == ()


def test_passage_is_one_whitespace_normalised_query():
    assert qr.parse_passage("  Team Members must\n not disable\tthe lock.  ") == (
        "Team Members must not disable the lock.",
    )
    assert qr.parse_passage("   ") == ()


# --- rewrite_query --------------------------------------------------------------


class _LLM:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.messages = text, exc, None

    def chat(self, messages):
        self.messages = messages
        if self.exc:
            raise self.exc
        return type("R", (), {"message": type("M", (), {"content": self.text})()})()


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(qr, "get_tracer", lambda: provider.get_tracer("t"))
    return exporter


def _use(monkeypatch, llm, mode):
    monkeypatch.setattr(qr.settings, "query_rewrite", mode)
    seen = {}

    def _get_llm(model=None, timeout=None):
        seen["timeout"] = timeout
        return llm

    monkeypatch.setattr(qr, "get_llm", _get_llm)
    return seen


def test_off_makes_no_llm_call_and_no_span(monkeypatch, spans):
    def _boom(*a, **kw):
        raise AssertionError("no LLM call when off")

    monkeypatch.setattr(qr.settings, "query_rewrite", "off")
    monkeypatch.setattr(qr, "get_llm", _boom)

    assert qr.rewrite_query(Q) == qr.RewriteResult("off")
    assert spans.get_finished_spans() == ()


def test_multi_returns_two_rephrasings_with_the_question_as_user_message(monkeypatch, spans):
    llm = _LLM("alt one\nalt two")
    seen = _use(monkeypatch, llm, "multi")
    monkeypatch.setattr(qr.settings, "query_rewrite_timeout", 20)

    r = qr.rewrite_query(Q)

    assert r.mode == "multi" and r.queries == ("alt one", "alt two") and not r.fallback
    assert seen["timeout"] == 20
    assert llm.messages[0].content == qr.REWRITE_PROMPTS["multi"]
    assert llm.messages[1].content == Q
    (span,) = spans.get_finished_spans()
    assert span.name == "query_rewrite"
    assert span.attributes["query_rewrite.mode"] == "multi"
    assert list(span.attributes["query_rewrite.queries"]) == ["alt one", "alt two"]
    assert span.attributes["query_rewrite.fallback"] is False


def test_multi_titles_puts_the_corpus_titles_in_the_prompt(monkeypatch, spans):
    llm = _LLM("alt one\nalt two")
    _use(monkeypatch, llm, "multi_titles")
    monkeypatch.setattr(
        qr,
        "policy_titles",
        lambda: ["Access Management Policy [Internal]", "Clear Desk And Clear Screen Policy [Internal]"],
    )

    qr.rewrite_query(Q)

    system = llm.messages[0].content
    assert "- Access Management Policy [Internal]\n- Clear Desk And Clear Screen Policy [Internal]" in system
    assert "{titles}" not in system


def test_a_titles_lookup_failure_falls_back_without_calling_the_llm(monkeypatch, spans):
    def _down():
        raise ConnectionError("qdrant down")

    llm = _LLM("unused")
    _use(monkeypatch, llm, "multi_titles")
    monkeypatch.setattr(qr, "policy_titles", _down)

    r = qr.rewrite_query(Q)

    assert r.fallback and r.queries == () and "ConnectionError" in r.error
    assert llm.messages is None


def test_hyde_returns_one_passage(monkeypatch, spans):
    _use(monkeypatch, _LLM("Team Members must not\ndisable the screen lock."), "hyde")
    assert qr.rewrite_query(Q).queries == ("Team Members must not disable the screen lock.",)


@pytest.mark.parametrize(
    "exc", [TimeoutError("slow"), ConnectionError("refused"), RuntimeError("Event loop is closed")]
)
def test_any_llm_exception_falls_back_to_the_original(monkeypatch, spans, exc):
    _use(monkeypatch, _LLM(exc=exc), "multi")

    r = qr.rewrite_query(Q)

    assert r.fallback and r.queries == () and type(exc).__name__ in r.error
    (span,) = spans.get_finished_spans()
    assert span.attributes["query_rewrite.fallback"] is True


def test_unusable_output_falls_back(monkeypatch, spans):
    _use(monkeypatch, _LLM(f"{Q}\n"), "multi")
    r = qr.rewrite_query(Q)
    assert r.fallback and r.error == "no usable rephrasing"


def test_the_prompts_forbid_answering_and_inventing():
    for mode in ("multi", "multi_titles"):
        assert "Do not answer" in qr.REWRITE_PROMPTS[mode]
        assert "Do not add" in qr.REWRITE_PROMPTS[mode]
