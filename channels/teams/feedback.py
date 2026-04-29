"""Feedback storage — append-only JSONL file."""

import json
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_FILE = Path("channels/teams/data/feedback.jsonl")


def save_feedback(
    question: str,
    answer: str,
    citations: list[dict],
    rating: int,
    user: str,
    chat_id: str,
) -> None:
    """Append one feedback record to the JSONL file."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "question": question,
        "answer": answer,
        "citations": citations,
        "rating": rating,
        "user": user,
        "chat_id": chat_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_feedback() -> list[dict]:
    """Load all feedback records. Returns empty list if file doesn't exist."""
    if not FEEDBACK_FILE.exists():
        return []
    records = []
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
