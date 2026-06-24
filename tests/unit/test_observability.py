import rag.observability as obs


def test_record_classification_no_raise_when_phoenix_disabled(monkeypatch):
    # phoenix disabled -> get_tracer() returns a no-op tracer; the call must not raise.
    monkeypatch.setattr(obs.settings, "phoenix_enabled", False)
    assert obs.record_classification("greeting", 0.95, False, "hi") is None
    assert obs.record_classification("in_scope", 0.0, True, "anything") is None


def test_record_classification_records_all_attributes_including_message(monkeypatch):
    # Capture the emitted span with an in-memory exporter and assert every attribute
    # lands — including the audited message text.
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(obs, "get_tracer", lambda: provider.get_tracer("test"))

    obs.record_classification("out_of_scope", 0.93, False, "order me a pizza")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "classification"
    assert span.attributes["router_category"] == "out_of_scope"
    assert span.attributes["router_confidence"] == 0.93
    assert span.attributes["router_fallback"] is False
    assert span.attributes["router_message"] == "order me a pizza"  # audit: full message recorded
