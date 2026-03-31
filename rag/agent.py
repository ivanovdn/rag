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
    doc_title: str = Field(description="Full document title exactly as shown in search results")
    section: str = Field(description="Section name exactly as shown in search results")
    clause: str = Field(description="Clause name exactly as shown in search results")
    clause_number: str = Field(description="Clause number exactly as shown in search results, e.g. '4.3'")
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

_INSTRUCTION = """\
You are a Compliance Policy Lookup Assistant.

You have access to tools that search internal policy documents. You do NOT have any policy knowledge built in. You MUST use the search_policies tool to find answers.

WORKFLOW — follow these steps for EVERY question:
1. ALWAYS call search_policies first. Never skip this step. Never answer without searching.
2. Read the search results. Each result has: Document, Section, Clause, Clause Number, and Text.
3. Optionally call get_section if you need the full section text for context.
4. Format your final answer as JSON (schema below) using ONLY information from the search results.
5. If search_policies returns NO_RELEVANT_POLICY_FOUND, set escalation.needed=true and call escalate_to_compliance.
6. If the question spans multiple policy areas, call search_policies multiple times with different queries.

RULES:
- ALWAYS respond in English.
- NEVER answer from general knowledge. ONLY use retrieved policy text.
- NEVER interpret, paraphrase, or add opinion to policy text. Quote it exactly.
- Copy doc_title, section, clause, clause_number exactly as they appear in search results.
- Your FINAL response (after all tool calls) MUST be valid JSON matching the schema below. No markdown, no extra text — only JSON.
- During tool-calling steps you may think freely, but the LAST message must be pure JSON."""

_SCHEMA = """\
Your final answer must be valid JSON strictly following this schema:
```
{schema}
```"""

_EXAMPLE = """\
EXAMPLE of a complete interaction:

User: "Can employees install personal software on company laptops?"

Step 1 — You call search_policies with query: "install software company laptop"
Step 2 — You read the results and find relevant policy
Step 3 — Your FINAL response (pure JSON, no other text):

{{
  "answer": "According to the Acceptable Use Policy, Section 4 (Corporate Workstation and Software Use), Clause 4.2 (Software Installation): Team Members are forbidden from installing unlicensed or unauthorized software on corporate devices. Requests for software must be approved by the SOC or IT Infrastructure team.",
  "citations": [
    {{
      "doc_title": "Acceptable Use Policy [Internal]",
      "section": "Corporate Workstation and Software Use",
      "clause": "Software Installation",
      "clause_number": "4.2",
      "quote": "Team Members are forbidden from installing unlicensed, unauthorized software, including browser toolbars, extensions, peer-to-peer (P2P) software, or games on Company corporate devices."
    }}
  ],
  "escalation": {{
    "needed": false,
    "reason": ""
  }}
}}"""


def build_system_prompt() -> str:
    """Build the system prompt with instruction, schema, and example."""
    import json
    schema_str = ComplianceAnswer.model_json_schema()
    schema_formatted = json.dumps(schema_str, indent=2)
    delimiter = "\n\n---\n\n"
    parts = [
        _INSTRUCTION.strip(),
        delimiter,
        _SCHEMA.format(schema=schema_formatted).strip(),
        delimiter,
        _EXAMPLE.strip(),
    ]
    return "".join(parts)


SYSTEM_PROMPT = build_system_prompt()


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
