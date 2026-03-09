import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from slugify import slugify

from config import settings
from ingest.chunk_models import PolicyChunk


def extract_heading_level(paragraph) -> int | None:
    """Returns 1, 2, 3 etc. for Heading styles, None otherwise."""
    style_name = paragraph.style.name or ""
    if style_name.startswith("Heading"):
        try:
            return int(style_name.split()[-1])
        except (ValueError, IndexError):
            return None
    return None


def extract_clause_number(text: str) -> str | None:
    """Regex: matches 1.2, 4.2.1, 3. etc at start of paragraph."""
    pattern = r"^(\d+(?:\.\d+)*\.?)\s"
    match = re.match(pattern, text.strip())
    return match.group(1).rstrip(".") if match else None


def extract_label(text: str) -> str | None:
    """Extract 'Label:' prefix from List Paragraph items (e.g. 'Acceptable Use: ...')."""
    match = re.match(r"^([A-Z][A-Za-z\s&\-/()]+):\s", text.strip())
    return match.group(1).strip() if match else None


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Split text at sentence boundaries if it exceeds max_tokens."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts = []
    current = []
    current_len = 0

    for sentence in sentences:
        s_tokens = _estimate_tokens(sentence)
        if current and current_len + s_tokens > max_tokens:
            parts.append(" ".join(current))
            current = [sentence]
            current_len = s_tokens
        else:
            current.append(sentence)
            current_len += s_tokens

    if current:
        parts.append(" ".join(current))

    return parts


def _table_to_text(table) -> str:
    """Convert a DOCX table to text rows: 'Header1: Val1 | Header2: Val2'."""
    rows = table.rows
    if not rows:
        return ""

    headers = [cell.text.strip() for cell in rows[0].cells]
    lines = []

    for row in rows[1:]:
        cells = [cell.text.strip() for cell in row.cells]
        if headers and any(h for h in headers):
            pairs = [f"{h}: {c}" for h, c in zip(headers, cells)]
            lines.append(" | ".join(pairs))
        else:
            lines.append(" | ".join(cells))

    if not lines and headers:
        lines.append(" | ".join(headers))

    return "\n".join(lines)


def _build_section_display(section_path: list[str]) -> str:
    """Build display string like 'Section A > Subsection B > Clause C'."""
    return " > ".join(p for p in section_path if p)


def _make_chunk(
    text: str,
    doc_id: str,
    doc_title: str,
    doc_filename: str,
    doc_link: str,
    section_path: list[str],
    clause_number: str,
    chunk_index: int,
    last_updated: str,
) -> PolicyChunk:
    clean_path = [p for p in section_path if p]
    return PolicyChunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        doc_title=doc_title,
        doc_filename=doc_filename,
        doc_link=doc_link,
        section_path=clean_path,
        section_display=_build_section_display(clean_path),
        clause_number=clause_number,
        text=text.strip(),
        char_count=len(text.strip()),
        chunk_index=chunk_index,
        last_updated=last_updated,
    )


def parse_docx(filepath: Path, doc_link: str) -> list[PolicyChunk]:
    """Main parser. Returns flat list of PolicyChunk objects."""
    doc = Document(filepath)

    doc_filename = filepath.name
    doc_title = filepath.stem.replace("_", " ").title()
    doc_id = slugify(filepath.stem)
    last_updated = datetime.now(timezone.utc).isoformat()

    chunks: list[PolicyChunk] = []
    current_headings = ["", "", ""]  # h1, h2, h3
    current_text_buffer: list[str] = []
    current_clause_number = ""
    # Accumulator for undersized chunks within the same section
    pending_small: list[str] = []
    pending_clause = ""
    chunk_index = 0

    def _emit_chunk(text: str, clause: str):
        """Create chunk(s) from text, splitting if oversized."""
        nonlocal chunk_index
        parts = _split_oversized(text, settings.chunk_max_tokens)
        for part in parts:
            c = _make_chunk(
                text=part,
                doc_id=doc_id,
                doc_title=doc_title,
                doc_filename=doc_filename,
                doc_link=doc_link,
                section_path=list(current_headings),
                clause_number=clause,
                chunk_index=chunk_index,
                last_updated=last_updated,
            )
            chunks.append(c)
            chunk_index += 1

    def flush_buffer():
        """Flush current buffer: emit if large enough, otherwise accumulate."""
        nonlocal current_text_buffer, current_clause_number, pending_small, pending_clause
        text = "\n".join(current_text_buffer).strip()
        current_text_buffer = []
        if not text:
            return

        if _estimate_tokens(text) >= settings.chunk_min_tokens:
            # First emit any pending small text
            flush_pending()
            _emit_chunk(text, current_clause_number)
        else:
            # Accumulate into pending
            pending_small.append(text)
            if not pending_clause and current_clause_number:
                pending_clause = current_clause_number

    def flush_pending():
        """Emit accumulated small chunks as one combined chunk."""
        nonlocal pending_small, pending_clause
        if not pending_small:
            return
        combined = "\n".join(pending_small).strip()
        pending_small = []
        if combined:
            _emit_chunk(combined, pending_clause)
        pending_clause = ""

    def flush_all():
        """Flush both buffer and pending at section/document boundaries."""
        flush_buffer()
        flush_pending()

    # Build a lookup from element to paragraph object for fast access
    para_map = {p._element: p for p in doc.paragraphs}
    table_map = {t._element: t for t in doc.tables}

    # Iterate doc.element.body to handle paragraphs and tables in order
    for element in doc.element.body:
        if element.tag == qn("w:p"):
            para = para_map.get(element)
            if para is None:
                continue

            text = para.text.strip()
            if not text:
                continue

            level = extract_heading_level(para)
            if level:
                flush_all()
                idx = min(level, 3) - 1
                current_headings[idx] = text
                for i in range(idx + 1, 3):
                    current_headings[i] = ""
                current_clause_number = ""
            else:
                clause_num = extract_clause_number(text)
                if not clause_num:
                    # Check for "Label:" pattern in List Paragraphs
                    label = extract_label(text)
                    if label and para.style.name == "List Paragraph":
                        clause_num = label
                if clause_num and current_text_buffer:
                    flush_buffer()
                    current_clause_number = clause_num
                elif clause_num:
                    current_clause_number = clause_num
                current_text_buffer.append(text)

        elif element.tag == qn("w:tbl"):
            table = table_map.get(element)
            if table:
                table_text = _table_to_text(table)
                if table_text:
                    current_text_buffer.append(table_text)

    flush_all()
    return chunks
