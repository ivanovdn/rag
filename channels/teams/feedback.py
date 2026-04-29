"""Feedback storage — appends to both JSONL and SQLite."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_FILE = Path("channels/teams/data/feedback.jsonl")
FEEDBACK_DB = Path("channels/teams/data/feedback.db")


def _ensure_db():
    """Create the feedback table if it doesn't exist."""
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations TEXT NOT NULL,
                rating INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)"
        )


def save_feedback(
    question: str,
    answer: str,
    citations: list[dict],
    rating: int,
    user: str,
    chat_id: str,
) -> None:
    """Append one feedback record to JSONL + SQLite."""
    timestamp = datetime.now(timezone.utc).isoformat()
    citations_json = json.dumps(citations, ensure_ascii=False)

    # Append to JSONL
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "question": question,
        "answer": answer,
        "citations": citations,
        "rating": rating,
        "user": user,
        "chat_id": chat_id,
        "timestamp": timestamp,
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Insert into SQLite
    _ensure_db()
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.execute(
            """
            INSERT INTO feedback (timestamp, user, chat_id, question, answer, citations, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, user, chat_id, question, answer, citations_json, rating),
        )


def load_feedback() -> list[dict]:
    """Load all feedback records from JSONL. Returns empty list if file doesn't exist."""
    if not FEEDBACK_FILE.exists():
        return []
    records = []
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_feedback_db() -> list[dict]:
    """Load all feedback records from SQLite, newest first."""
    if not FEEDBACK_DB.exists():
        return []
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY timestamp DESC"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "user": r["user"],
            "chat_id": r["chat_id"],
            "question": r["question"],
            "answer": r["answer"],
            "citations": json.loads(r["citations"]),
            "rating": r["rating"],
        }
        for r in rows
    ]
