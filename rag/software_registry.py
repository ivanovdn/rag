"""Pure logic for the software allowed/forbidden registry: row model, name
normalization, in-memory fuzzy name index, and embedding-text construction.

No Qdrant or LLM here — the check_software tool wires this to the backends.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from config import settings


class SoftwareRow(BaseModel):
    name: str
    status: str  # "allowed" | "forbidden"
    category: str = ""
    note: str = ""
    alternative: str = ""
    aliases: list[str] = Field(default_factory=list)
    source_list: str = ""


@dataclass
class LookupResult:
    row: SoftwareRow | None
    score: float
    exact: bool = False
    suggestion: SoftwareRow | None = None


def normalize_name(s: str) -> str:
    """Lowercase and collapse internal whitespace."""
    return " ".join(s.lower().split())


def load_registry(path: str | None = None) -> list[SoftwareRow]:
    """Load and validate the committed JSON registry."""
    p = Path(path or settings.software_registry_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return [SoftwareRow(**row) for row in data]


def build_name_index(rows: list[SoftwareRow]) -> dict[str, SoftwareRow]:
    """Map each normalized name AND alias -> its row."""
    index: dict[str, SoftwareRow] = {}
    for row in rows:
        index[normalize_name(row.name)] = row
        for alias in row.aliases:
            index[normalize_name(alias)] = row
    return index


def lookup_name(query: str, index: dict[str, SoftwareRow], threshold: float) -> LookupResult:
    """Exact match, else fuzzy match >= threshold. Below threshold -> row=None with
    the closest candidate as `suggestion` (for a 'did you mean' hint)."""
    q = normalize_name(query)
    if not index:
        return LookupResult(row=None, score=0.0)
    if q in index:
        return LookupResult(row=index[q], score=100.0, exact=True)

    # fuzz.ratio (length-sensitive whole-string) NOT WRatio: WRatio's partial/token
    # scorers give a long garbage/compound query a confident score against a short
    # registry name (e.g. "dockerr composer xyz" -> "docker" = 90), a wrong
    # authoritative verdict. ratio rejects those (46) while keeping real typos
    # (winrarr -> winrar = 92). Bare partial names miss here and fall through to the
    # semantic fallback + aliases — a safe miss, not a false verdict.
    match = process.extractOne(q, list(index.keys()), scorer=fuzz.ratio)
    if match is None:
        return LookupResult(row=None, score=0.0)
    matched_key, score = match[0], match[1]
    if score >= threshold:
        return LookupResult(row=index[matched_key], score=score, exact=False)
    return LookupResult(row=None, score=score, suggestion=index[matched_key])


def software_embed_text(row: SoftwareRow) -> str:
    """Text embedded for the semantic (category) fallback. Forbidden rows fold in
    category + description + alternative; allowed rows are name (+ note)."""
    if row.status == "forbidden":
        parts = [row.name]
        if row.category:
            parts.append(row.category)
        if row.note:
            parts.append(row.note)
        if row.alternative:
            parts.append(f"Alternative: {row.alternative}")
        return " | ".join(parts)
    return f"{row.name} | {row.note}" if row.note else row.name
