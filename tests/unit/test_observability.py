import rag.observability as obs


def test_record_classification_no_raise_when_phoenix_disabled(monkeypatch):
    # phoenix disabled -> get_tracer() returns a no-op tracer; the call must not raise.
    monkeypatch.setattr(obs.settings, "phoenix_enabled", False)
    assert obs.record_classification("greeting", 0.95, False) is None
    assert obs.record_classification("in_scope", 0.0, True) is None
