from llama_index.core.tools import FunctionTool


def ask_clarification(question_to_user: str) -> str:
    """
    Ask the user a clarifying question BEFORE searching policies, when the
    original question is ambiguous and could refer to multiple different
    policy areas. For example: "reporting something" could mean reporting
    an incident, reporting to regulators, or reporting a colleague.
    Use sparingly - only when ambiguity would lead to wrong policy retrieval.
    Input: the clarifying question to ask the user.
    Output: a formatted clarification request (the agent should pause and
    present this to the user before proceeding).
    """
    return f"CLARIFICATION_NEEDED: {question_to_user}"


ask_clarification_tool = FunctionTool.from_defaults(fn=ask_clarification)
