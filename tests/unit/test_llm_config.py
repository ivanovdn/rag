from config import settings
from rag.agent import get_llm


def test_ollama_llm_gets_keep_alive_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    monkeypatch.setattr(settings, "ollama_keep_alive", "7m")
    llm = get_llm()
    assert llm.keep_alive == "7m"


def test_ollama_keep_alive_default_is_longer_than_ollama_default():
    # Ollama's own default is '5m'; ours must outlast a working-day gap.
    value = settings.ollama_keep_alive
    number = float(value[:-1])
    unit = value[-1]
    minutes = number * 60 if unit == "h" else number
    assert minutes >= 30


def test_ollama_llm_gets_num_ctx_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    # Distinctive value: neither the old hardcoded 8192 nor the 4096
    # default, so this can only pass if num_ctx truly flows from settings.
    monkeypatch.setattr(settings, "ollama_num_ctx", 12345)
    llm = get_llm()
    assert llm.additional_kwargs["num_ctx"] == 12345


def test_ollama_num_ctx_default_is_below_crash_threshold():
    # num_ctx >= 8192 is one of five conditions for the reproducible
    # MoE+CUDA crash documented in
    # docs/superpowers/specs/2026-09-17-ollama-moe-cuda-crash.md. The
    # upstream crash matrix proved only 4096 safe; this guard fails the
    # suite if the default is ever raised back past that boundary.
    # Bind to a local first (not `assert settings.ollama_num_ctx < 8192`
    # directly) so a failure's pytest introspection never prints the
    # Settings repr, which carries live secrets (hf_token, smtp_password, …).
    value = settings.ollama_num_ctx
    assert value < 8192


def test_openai_like_disables_thinking(monkeypatch):
    # Qwen3 models think by default on vLLM. Measured cost: 294 reasoning
    # tokens and 34.5s for one 16-token router classification.
    monkeypatch.setattr(settings, "llm_backend", "openai-compatible")
    llm = get_llm()
    body = llm.additional_kwargs["extra_body"]
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    # Without is_function_calling_model=True the agent emits ReAct text
    # instead of tool calls — a silent, expensive failure (CLAUDE.md
    # gotchas table).
    assert llm.is_chat_model is True
    assert llm.is_function_calling_model is True
    # Proves the timeout fix landed: OpenAILike has no `request_timeout`
    # field and silently drops it, leaving the SDK's 60s default.
    assert llm.timeout == float(settings.active_request_timeout)


def test_get_llm_builds_a_fresh_client_per_call(monkeypatch):
    # _run_rag runs each request in a new asyncio.run() loop. llama-index's Ollama
    # caches its httpx.AsyncClient on first use, and a pooled connection from a
    # closed loop raises "RuntimeError: Event loop is closed" on the next one
    # (reproduced 2026-09-16). A fresh client per call is what keeps that safe.
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    first = get_llm()
    # `_async_client` is a PrivateAttr populated lazily on first use, not at
    # construction -- two never-used clients both read `_async_client is
    # None` regardless of caching, which would make the check below a
    # no-op. Warming `first` here reproduces the real request sequence
    # (request 1 uses its client, then request 2 asks for one), which is
    # what makes that check meaningful. Do not delete this line as dead code.
    _ = first.async_client
    second = get_llm()
    assert second is not first
    # `is not` alone would still pass for `return _CACHED.model_copy()`: the
    # copy is a distinct object but carries `first`'s now-warm
    # `_async_client` over by reference, reintroducing the exact
    # dead-client-from-a-closed-loop bug. A genuinely fresh client has never
    # made a call, so this must be None.
    assert second._async_client is None


def test_the_request_budget_fits_in_the_context_window():
    """num_ctx is pinned at 4096 by the MoE+CUDA crash matrix, so the budget is
    fixed and has to be checked, not hoped for.

    Before this work the fixed overhead was 1883 tokens (1066 prompt + 817 of tool
    schemas) — 46% of the window — and num_predict 1024 exceeded the 956 tokens
    actually left at the largest observed prompt. The cap was nominal.

    FIXED_OVERHEAD_TOKENS is measured, not derived, so the char assertion below
    pins it to the prompt it was measured against: editing the prompt without
    re-measuring fails here instead of silently rotting the constant.
    """
    from rag.agent import FIXED_OVERHEAD_TOKENS, MAX_SOURCE_TOKENS, SYSTEM_PROMPT

    num_ctx = settings.ollama_num_ctx
    num_predict = 1024

    assert len(SYSTEM_PROMPT) == 2169, (
        "SYSTEM_PROMPT changed; re-measure FIXED_OVERHEAD_TOKENS with "
        "/api/chat num_predict=1 and update both numbers together"
    )
    assert FIXED_OVERHEAD_TOKENS + MAX_SOURCE_TOKENS + num_predict <= num_ctx


def test_the_system_prompt_names_no_tools():
    """The agent is tool-free (spec D2). A prompt that still orders a tool call
    would make the model attempt one that does not exist."""
    from rag.agent import SYSTEM_PROMPT

    for tool in ("search_policies", "get_section", "escalate_to_compliance"):
        assert tool not in SYSTEM_PROMPT
