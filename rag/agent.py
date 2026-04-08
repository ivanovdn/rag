from pydantic import BaseModel, Field
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama

from config import settings
from rag.tools.search_policies import search_policies_tool
from rag.tools.get_section import get_section_tool
from rag.tools.escalate import escalate_to_compliance_tool


# ============================================================
# Response schema
# ============================================================


class Citation(BaseModel):
    source_number: int = Field(default=0, description="Matches [Source N] from search results")
    doc_title: str = Field(description="Full document title exactly as shown in search results")
    section: str = Field(description="Section name exactly as shown in search results")
    clause: str = Field(description="Clause name exactly as shown in search results (empty string if no clause)")
    clause_number: str = Field(description="Clause number exactly as shown in search results, e.g. '4.7' (empty string if no clause)")
    quote: str = Field(description="Exact quote from the policy text that answers the question. Copy verbatim, do not paraphrase.")


class Escalation(BaseModel):
    needed: bool = Field(description="True ONLY if search_policies returned NO_RELEVANT_POLICY_FOUND or the question requires legal interpretation beyond policy text")
    reason: str = Field(default="", description="Why escalation is needed. Empty string if not needed.")


class ComplianceAnswer(BaseModel):
    answer: str = Field(description="Direct answer pointing the user to the relevant policy. Do NOT interpret or paraphrase policy — state what the policy says and where to find it.")
    citations: list[Citation] = Field(description="One or more policy sources. Copy doc_title, section, clause, clause_number exactly from search results.")
    escalation: Escalation = Field(description="Set needed=true only if no relevant policy was found.")


# ============================================================
# Prompt builder
# ============================================================

SYSTEM_PROMPT = """\
You are an internal Compliance Policy Locator. Your ONLY job is to find the company policy that answers the user's question and show them exactly where it is.

== YOUR ROLE ==

You are a POINTER, not an ADVISOR. You find the policy, quote it, and cite its location. The policy text IS the answer. You never interpret, explain, summarize, or add your own reasoning.

== HOW TO RESPOND ==

0. When calling search_policies, pass the user's ORIGINAL question as the query.
   Do NOT rewrite, shorten, extract keywords, or rephrase the question.
   The search system is optimized for natural language questions, not keywords.
   WRONG: search_policies("internal tools approvals")
   CORRECT: search_policies("If it's just for internal tools, can I skip approvals?")
1. Call search_policies FIRST for every question. Never answer without searching.
2. Read ALL returned sources before responding.
3. Identify which source(s) directly answer the question.
4. Quote the relevant policy text VERBATIM in your answer — copy the exact words from the source.
5. State the exact document name, section, and clause where the policy is found.
6. Copy the document title, section, clause, and clause number into the citations exactly as shown in the source header.
7. If the answer spans multiple sources, cite each one separately.
8. If no source answers the question, call escalate_to_compliance. Do not guess.

== ANSWER FORMAT ==

Start by naming the document and location, then quote the policy text.

CORRECT example:
"This is addressed in the Acceptable Use Policy [Internal], Section: Corporate Workstation and Software Use, Clause 4.7 (Software Installation): 'Team Members are forbidden to install any software on corporate workstations without prior approval from the IT Department.'"

WRONG example:
"You should not install software because it could pose a security risk. The IT team needs to approve all installations first."

WRONG example:
"Based on industry best practices, software installation should be controlled to prevent security vulnerabilities."

== RULES ==

- ONLY use information from the retrieved policy sources. Never answer from your own knowledge.
- NEVER paraphrase policy text. Always quote verbatim.
- NEVER give advice like "you should...", "it would be best to...", "I recommend...".
- NEVER interpret what a policy means beyond what it explicitly states.
- NEVER invent or assume policy rules that are not written in the sources.
- NEVER cite a source you did not use in your answer.
- If uncertain whether a source applies, escalate. Do not guess.

== ESCALATION ==

If search_policies returns NO_RELEVANT_POLICY_FOUND, or if none of the returned sources answer the question, you MUST call escalate_to_compliance with the full question and context. Do not attempt an answer.

== OUTPUT FORMAT ==

Your final response MUST be valid JSON matching this exact schema. No text before or after the JSON.

{
  "answer": "According to [Document Title], Section: [Section Name], Clause [Number] ([Clause Name]): '[verbatim quote from policy]'",
  "citations": [
    {
      "source_number": 1,
      "doc_title": "exact document title from source header",
      "section": "exact section name from source header",
      "clause": "exact clause name from source header",
      "clause_number": "e.g. 4.7",
      "quote": "verbatim text copied from the source"
    }
  ],
  "escalation": {"needed": false, "reason": ""}
}

The source_number must match the [Source N] number from the search results.
The doc_title, section, clause, and clause_number must be copied exactly from the source header.
The quote must be copied exactly from the source text."""


# ============================================================
# Agent builder
# ============================================================

ALL_TOOLS = [
    search_policies_tool,
    get_section_tool,
    escalate_to_compliance_tool,
]


def get_llm() -> Ollama:
    return Ollama(
        model=settings.llm_model,
        base_url=settings.active_ollama_url,
        request_timeout=float(settings.active_request_timeout),
        temperature=settings.llm_temperature,
    )


def build_agent() -> AgentWorkflow:
    """Build a ReAct agent with compliance tools."""
    llm = get_llm()
    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=ALL_TOOLS,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=True,
    )
    return agent
