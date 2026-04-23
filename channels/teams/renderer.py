"""HTML rendering for Teams bot responses.

Takes ComplianceAnswer dicts from the RAG pipeline and renders
Teams-compatible HTML.
"""


WELCOME_HTML = """<p><b>Compliance Policy Assistant</b></p>

<p>Ask any compliance question — the assistant will find the relevant
<b>policy</b>, <b>section</b>, and <b>clause</b>, and quote the exact
policy text.</p>

<hr>

<p><b>How to use</b></p>
<ul>
<li>Type your question in plain language.</li>
<li>Type <code>start</code> any time to see this message again.</li>
</ul>

<p><i>Lookups typically take 30-60 seconds.</i></p>"""

LOADING_HTML = (
    "<p><b>Searching compliance policies...</b><br>"
    "<i>Finding the relevant policy, section, and clause. This may take up to a minute.</i></p>"
)


def render_answer(result: dict) -> str:
    """Render a successful ComplianceAnswer as Teams HTML."""
    answer = result.get("answer", "")
    citations = result.get("citations", [])

    parts = [f"<p>{answer}</p>"]

    if citations:
        parts.append("<hr>")
        parts.append("<p><b>Sources:</b></p>")
        parts.append("<ul>")
        for c in citations:
            doc = c.get("doc_title", "")
            section = c.get("section", "")
            clause = c.get("clause", "")
            clause_num = c.get("clause_number", "")

            location = doc
            if section:
                location += f" &gt; {section}"
            if clause and clause_num:
                location += f" &gt; Clause {clause_num}: {clause}"
            elif clause:
                location += f" &gt; {clause}"

            parts.append(f"<li>{location}</li>")
        parts.append("</ul>")

    return "\n".join(parts)


def render_escalation(question: str, result: dict) -> str:
    """Render an escalation response."""
    reason = result.get("escalation", {}).get("reason", "No relevant policy found.")
    return (
        "<p><b>Escalated to Compliance Team</b></p>"
        f"<p><b>Question:</b> {question}</p>"
        f"<p><b>Reason:</b> {reason}</p>"
        "<p>Your question has been forwarded to the Compliance team. "
        "They will follow up with you directly.</p>"
    )


def render_error(question: str, error: str) -> str:
    """Render an error response."""
    return (
        "<p><b>Compliance lookup failed</b></p>"
        f"<p><b>Question:</b> {question}</p>"
        f"<p>{error}</p>"
        "<p>Please try again or contact the Compliance team directly.</p>"
    )
