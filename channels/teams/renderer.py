"""HTML rendering for Teams bot responses.

Takes ComplianceAnswer dicts from the RAG pipeline and renders
Teams-compatible HTML.
"""


WELCOME_HTML = """<p><b>Trinetix Compliance Q&amp;A bot</b></p>

<p>Hi! I quote policy rather than interpret it, so I won't give you a
personal ruling — I point you to the exact <b>policy</b>, <b>section</b>,
and <b>clause</b>.</p>

<hr>

<p><b>How to use</b></p>
<ul>
<li>Ask your question in plain English.</li>
<li>I don't remember earlier messages yet — put your whole question in one message.</li>
<li>Lookups take ~30–60 seconds.</li>
<li>Type <code>start</code> anytime to see this again.</li>
</ul>"""

LOADING_HTML = (
    "<p><b>Searching compliance policies...</b><br>"
    "<i>Finding the relevant policy, section, and clause. This may take up to a minute.</i></p>"
)

# Shown when a backend (policy DB / models) is transiently unreachable.
# Editable: reword freely; it must stay valid Teams-limited HTML.
UNAVAILABLE_HTML = (
    "<p><b>⚠️ Policy service temporarily unavailable</b></p>"
    "<p>I can't reach the policy database right now. "
    "Please try again in a moment.</p>"
)


def render_unavailable() -> str:
    """Render the transient-infra-unavailable message."""
    return UNAVAILABLE_HTML


# Shown when the router classifies a message as off-topic (ROUTER-3). Editable: reword
# freely; must stay valid Teams-limited HTML.
OUT_OF_SCOPE_HTML = (
    "<p><b>I can only answer questions about company policies.</b></p>"
    "<p>Ask me about a policy and I'll find the relevant section and clause.</p>"
)

# Shown when the router can't read the message (gibberish / wrong keyboard layout). Editable.
UNINTELLIGIBLE_HTML = (
    "<p><b>I couldn't read that.</b></p>"
    "<p>It may have been typed with a different keyboard layout. "
    "Please retype your question.</p>"
)


def render_out_of_scope() -> str:
    """Render the out-of-scope redirect message."""
    return OUT_OF_SCOPE_HTML


def render_unintelligible() -> str:
    """Render the unintelligible-input retype prompt."""
    return UNINTELLIGIBLE_HTML


def render_answer(result: dict) -> str:
    """Render a successful ComplianceAnswer as Teams HTML.
    Each citation is rendered as a separate block: bold location + verbatim quote.
    Multiple citations are separated with <hr>.
    """
    citations = result.get("citations", [])
    answer = result.get("answer", "")

    if not citations:
        # Fallback to prose answer when there are no structured citations
        return f"<p>{answer}</p>"

    parts = []
    if len(citations) > 1:
        parts.append(f"<p>This is addressed in {len(citations)} policies:</p>")

    for i, c in enumerate(citations):
        if i > 0:
            parts.append("<hr>")

        doc = c.get("doc_title", "")
        section = c.get("section", "")
        clause = c.get("clause", "")
        clause_num = c.get("clause_number", "")
        quote = c.get("quote", "")

        # Location line: bold labels for Document / Section / Clause
        location_lines = []
        if doc:
            location_lines.append(f"<b>📄 {doc}</b>")
        if section:
            location_lines.append(f"<b>Section:</b> {section}")
        # If LLM mistakenly copied the number into the clause name, just show the number
        clause_is_just_number = clause and clause_num and clause.strip() == clause_num.strip()
        if clause_is_just_number:
            location_lines.append(f"<b>Clause {clause_num}</b>")
        elif clause and clause_num:
            location_lines.append(f"<b>Clause {clause_num}:</b> {clause}")
        elif clause:
            location_lines.append(f"<b>Clause:</b> {clause}")
        elif clause_num:
            location_lines.append(f"<b>Clause {clause_num}</b>")

        parts.append(f"<p>{'<br>'.join(location_lines)}</p>")

        if quote:
            parts.append(f"<p><i>\"{quote}\"</i></p>")

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


RATING_PROMPT_HTML = (
    "<p><i>Was this helpful? Reply "
    "<b>-1</b> (should have been escalated), "
    "<b>0</b> (wrong), "
    "<b>1</b> (partially), or "
    "<b>2</b> (correct)</i></p>"
)

RATING_THANKS_HTML = "<p><i>Thanks for the feedback!</i></p>"
