from rag.tools.search_policies import search_policies_tool
from rag.tools.get_section import get_section_tool
from rag.tools.clarify import ask_clarification_tool
from rag.tools.escalate import escalate_to_compliance_tool

ALL_TOOLS = [
    search_policies_tool,
    get_section_tool,
    ask_clarification_tool,
    escalate_to_compliance_tool,
]
