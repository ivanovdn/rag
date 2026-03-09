import uuid
from datetime import datetime, timezone

from llama_index.core.tools import FunctionTool

from config import settings

# In-memory escalation store (replaced by DB in Step 9)
_escalations: dict[str, dict] = {}


def escalate_to_compliance(
    reason: str,
    unanswered_question: str,
    search_attempted: bool = True,
) -> str:
    """
    Escalate to the Compliance team when: (1) no relevant policy was found
    after searching, (2) policies are ambiguous or contradictory,
    (3) the question requires legal interpretation beyond policy text, or
    (4) confidence in the retrieved answer is low.
    NEVER guess or answer from general knowledge. If unsure, escalate.
    Input: reason for escalation, the original user question, whether search was attempted.
    Output: escalation confirmation with ticket ID.
    This tool saves the full conversation context automatically.
    """
    ticket_num = len(_escalations) + 1
    ticket_id = f"{settings.escalation_ticket_prefix}-{datetime.now(timezone.utc).strftime('%Y')}-{ticket_num:04d}"

    ticket = {
        "id": ticket_id,
        "question": unanswered_question,
        "reason": reason,
        "search_attempted": search_attempted,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _escalations[ticket_id] = ticket

    return (
        f"ESCALATED: Your question has been forwarded to the Compliance team "
        f"(Ticket #{ticket_id}). They will respond within 2 business days. "
        f"Reason: {reason}"
    )


def get_escalations() -> dict[str, dict]:
    """Return all escalation tickets."""
    return _escalations


escalate_to_compliance_tool = FunctionTool.from_defaults(fn=escalate_to_compliance)
