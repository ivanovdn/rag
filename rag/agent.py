from llama_index.core.agent.workflow import AgentWorkflow
from pydantic import BaseModel, Field

from config import settings

# ============================================================
# Response schema
# ============================================================


# AUTHORITATIVE CONTRACT: the JSON block inside SYSTEM_PROMPT.
# These models are never instantiated and never sent to the LLM — no
# response_format is set (it kills tool-call emission), and rag/response.py parses
# with plain dict.get. They are documentation of the same contract, kept here for
# readers. Change the prompt and these together, or they drift apart silently.
class Citation(BaseModel):
    source_number: int = Field(
        default=0, description="Matches [Source N] from search results"
    )
    doc_title: str = Field(
        description="Full document title exactly as shown in search results"
    )
    section: str = Field(description="Section name exactly as shown in search results")
    clause: str = Field(
        description="Clause name exactly as shown in search results (empty string if no clause)"
    )
    clause_number: str = Field(
        description="Clause number exactly as shown in search results, e.g. '4.7' (empty string if no clause)"
    )
    quote: str = Field(
        description="Exact quote from the policy text that answers the question. Copy verbatim, do not paraphrase."
    )


class Escalation(BaseModel):
    needed: bool = Field(
        description="True ONLY if search_policies returned NO_RELEVANT_POLICY_FOUND or the question requires legal interpretation beyond policy text"
    )
    reason: str = Field(
        default="", description="Why escalation is needed. Empty string if not needed."
    )


class ComplianceAnswer(BaseModel):
    answer: str = Field(
        description="Direct answer pointing the user to the relevant policy. Do NOT interpret or paraphrase policy — state what the policy says and where to find it."
    )
    citations: list[Citation] = Field(
        description="One or more policy sources. Copy doc_title, section, clause, clause_number exactly from search results."
    )
    escalation: Escalation = Field(
        description="Set needed=true only if no relevant policy was found."
    )


# ============================================================
# Prompt builder
# ============================================================

# Measured against a live qwen3.6 36B at num_ctx 4096 via /api/chat with
# num_predict=1, reading prompt_eval_count:
#   this prompt            571 tokens   (the previous one: 1066)
#   tool schemas             0 tokens   (the previous three: 817)
#   fixed overhead         571 tokens   (previously 1883 — 46% of the window)
# Re-measure both constants below if you edit the prompt; the guard in
# tests/unit/test_llm_config.py pins them to its character count.
FIXED_OVERHEAD_TOKENS = 571
# Largest source payload observed in Phoenix (~1257 tokens), rounded up.
MAX_SOURCE_TOKENS = 1300

SYSTEM_PROMPT = """\
You are an internal Compliance Policy Locator. You find the company policy that answers the user's question and show exactly where it is.

You are a POINTER, not an ADVISOR. The policy text IS the answer. Never interpret, explain, summarize, advise, or add reasoning of your own.

== SOURCES ==

The relevant policy sources are given in the user message under [Source N] headers. Read ALL of them before answering.
If none of them answers the question, set escalation.needed = true with a short reason, and leave citations empty. Never guess and never answer from your own knowledge.

== RULES ==

- Quote policy text VERBATIM from every source you cite. Never paraphrase.
- Cite EVERY source that addresses the question, not just the first. Each gets its own entry in citations with its own quote.
- Never cite a source you did not use in the answer.
- Never invent or assume a rule that is not written in the sources.
- Copy source_number, doc_title, section, clause and clause_number exactly as they appear in the source header. Use an empty string when there is no clause.
- If you are uncertain whether a source applies, escalate instead of guessing.

== ANSWER ==

Name the document and location, then quote it:
"According to [Policy Name], Section: [Section Name], Clause [Number] ([Clause Name]): '[verbatim quote].'"

WRONG — advice, and not grounded in a source:
"You should not install software because it could pose a security risk. The IT team needs to approve all installations first."

WRONG — answered from general knowledge instead of the sources:
"Based on industry best practices, software installation should be controlled to prevent security vulnerabilities."

== OUTPUT ==

Reply with valid JSON only. No text before or after it.

{
  "answer": "[Document A], Section: [X], Clause [N]: '[quote]'. [Document B], Section: [Y], Clause [M]: '[quote]'.",
  "citations": [
    {"source_number": 1, "doc_title": "exact title from the source header", "section": "exact section name", "clause": "exact clause name", "clause_number": "4.7", "quote": "verbatim text from the source"},
    {"source_number": 2, "doc_title": "second document title", "section": "exact section name", "clause": "exact clause name", "clause_number": "8.7", "quote": "verbatim text from the second source"}
  ],
  "escalation": {"needed": false, "reason": ""}
}"""


# ============================================================
# Agent builder
# ============================================================

# Empty by design (spec D2), not by oversight.
# Measured: get_section and escalate_to_compliance were never invoked in
# production. escalate lost to the JSON escalation.needed field — the model must
# emit it anyway, so the tool call is a wasted round-trip — and get_section lost
# to satisfaction: once search returns usable sources the model stops, and three
# controlled probes (explicit order, order moved to the prompt tail, gating
# condition stripped from its docstring) all failed to force a second retrieval
# step. search_policies is gone because retrieval now runs in rag/search_first.py
# before the agent is built.
# Re-adding a tool costs ~270-410 tokens of schema on every request and puts
# tool-calling back — one of the five conditions in the MoE+CUDA crash matrix.
ALL_TOOLS = []

# Qwen3-family models emit reasoning traces by default on vLLM/llama-server.
# Measured: 294 reasoning tokens and 34.5s to produce a 16-token router
# classification, versus 2.2s with thinking off. The tolerant JSON extractor
# still parses it, so this regresses silently — hence the explicit switch.
# The Ollama branch has its own native `thinking=False` argument.
_NO_THINKING_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def get_llm(model: str | None = None):
    """Build a fresh LLM client. Deliberately NOT cached.

    _run_rag runs each request under its own asyncio.run() loop. llama-index's
    Ollama creates its httpx.AsyncClient once and reuses it, and a pooled
    connection from a closed loop fails on the next loop with
    "RuntimeError: Event loop is closed" (reproduced 2026-09-16) — which is not
    a transient error, so it would surface as a false content escalation.
    Construction is cheap (see scripts/bench_llm_construction.py). If a shared
    client is ever wanted, the worker thread must own one persistent event loop
    first.
    """
    if settings.llm_backend == "openai-compatible":
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=model or settings.openai_model,
            api_base=settings.openai_api_base,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            timeout=float(settings.active_request_timeout),
            is_chat_model=True,
            is_function_calling_model=True,
            additional_kwargs={"extra_body": _NO_THINKING_BODY},
        )
    else:
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=model or settings.llm_model,
            base_url=settings.active_ollama_url,
            request_timeout=float(settings.active_request_timeout),
            temperature=settings.llm_temperature,
            thinking=False,
            keep_alive=settings.ollama_keep_alive,
            # context_window: pass the same pinned value so llama-index's
            # Ollama.get_context_window() has a value != -1 and never calls
            # self.client.show(model) — that call would otherwise hit
            # settings.active_ollama_url (172.20.0.22 in production) on
            # every build_agent(), including from tests. Does NOT change
            # num_ctx itself; still the one value the crash matrix proved
            # safe (see below).
            context_window=settings.ollama_num_ctx,
            # num_ctx: settings.ollama_num_ctx (4096) is the only value the
            # upstream crash matrix proved safe against the MoE+CUDA fault —
            # see docs/superpowers/specs/2026-09-17-ollama-moe-cuda-crash.md.
            # num_predict=1024 is ~66% headroom over the observed max
            # completion (617 tokens, Phoenix spans) and, unlike the old
            # 4096, actually fits alongside the prompt in that same window.
            additional_kwargs={"num_predict": 1024, "num_ctx": settings.ollama_num_ctx},
        )


def build_agent() -> AgentWorkflow:
    """Build the tool-free compliance agent.

    NOT a ReAct agent, despite what this docstring said until 2026-09-24:
    AgentWorkflow.from_tools_or_functions picks FunctionAgent when
    llm.metadata.is_function_calling_model is True, and llama-index's Ollama
    reports True. Production has always used native tool calls, never ReAct text.
    """
    llm = get_llm()
    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=ALL_TOOLS,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        timeout=float(settings.agent_timeout),
        # Off: verbose=True prints ~20 [tick]/[run_agent_step] lines per question,
        # which buries the bot's own log — the WARNING lines an operator actually
        # needs (watermark holds, failed acks, force-advances) become unfindable
        # once 30 people are asking. The same detail is in Phoenix, structured and
        # searchable, which is where it belongs. Eval keeps its own verbose switch
        # (eval/agent_wrapper.py build_instrumented_agent) for debugging runs.
        verbose=False,
    )
    return agent
