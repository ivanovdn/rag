# TASK: Add Section Name and Clause Name to Chunk Metadata

## Goal

Each chunk must have these separate, clean metadata fields:

```
doc_title:      "Acceptable Use Policy [Internal]"
section:        "Private Information"
section_number: "7"
clause:         "Blogging and Social Media"
clause_number:  "7.5"
```

So the LLM can cite like:
**Document:** Acceptable Use Policy [Internal] | **Section:** 7. Private Information | **Clause:** 7.5. Blogging and Social Media

## Where Each Piece Comes From

Based on diagnostic analysis of the actual DOCX files:

```
ilvl=0, Heading 1, decimal         → section_number="7", section="Private Information"
  ilvl=1, List Paragraph, decimal  → clause_number="7.5", clause="Blogging and Social Media"
    ilvl=2, bullet                 → content (bullet items under clause)
    Normal                         → content (continuation paragraphs)
  ilvl=1, List Paragraph, decimal  → clause_number="7.6", clause="Landline Telephones and Mobile Phones"
```

**Section** = ilvl=0 (Heading 1). Resolved number "7." → `section_number = "7"`. The `para.text` = "Private Information" → `section = "Private Information"`.

**Clause** = ilvl=1 (List Paragraph, decimal). Resolved number "7.5." → `clause_number = "7.5"`. The clause name is the **bold first run** text with trailing colon stripped. Example: `runs[0].text = "Blogging and Social Media:"` → `clause = "Blogging and Social Media"`.

**Important:** The clause name is NOT the full paragraph text. Only the bold label before the colon. The full paragraph text (including the body after the colon) goes into the chunk `text` field.

## Step 1: Update `ingest/chunk_models.py`

Replace the section/clause fields with:

```python
class PolicyChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str          # "Acceptable Use Policy [Internal]"
    doc_filename: str
    doc_link: str

    section: str = ""        # "Private Information"
    section_number: str = "" # "7"
    clause: str = ""         # "Blogging and Social Media"
    clause_number: str = ""  # "7.5"

    section_display: str = "" # "7. Private Information > 7.5. Blogging and Social Media"

    text: str
    char_count: int = 0
    chunk_index: int = 0
    last_updated: str = ""
```

Remove `section_path: list[str]` — replaced by the explicit fields above.

## Step 2: Update `ingest/docx_parser.py`

### 2a: Add helper to extract clause name from bold runs

```python
def extract_clause_name(para) -> str:
    """
    Extract clause name from the bold first run of a paragraph.
    
    ilvl=1 paragraphs look like:
        runs = [bold:"Blogging and Social Media:", normal:" Limited and occasional..."]
    
    Returns "Blogging and Social Media" (bold text, colon stripped).
    Returns empty string if first run is not bold.
    """
    runs = para.runs
    if not runs or not runs[0].bold:
        return ""
    
    # Collect all consecutive bold runs (sometimes name spans multiple runs)
    bold_parts = []
    for run in runs:
        if run.bold:
            bold_parts.append(run.text)
        else:
            break
    
    name = "".join(bold_parts).strip().rstrip(":").strip()
    return name
```

### 2b: Track section and clause as separate variables

In `parse_docx()`, replace the `current_headings` tracking with explicit variables:

```python
    # Replace:
    #   current_headings = ["", "", ""]
    # With:
    current_section = ""         # "Private Information"
    current_section_number = ""  # "7"
    current_clause = ""          # "Blogging and Social Media"
    current_clause_number = ""   # "7.5"
```

### 2c: Update the main loop paragraph processing

When processing paragraphs with resolved numbering:

```python
        if element.tag == qn("w:p"):
            para = para_map.get(element)
            if para is None:
                continue

            text = para.text.strip()
            if not text:
                continue

            num_info = resolver.resolve(element)
            level = extract_heading_level(para)

            if num_info and num_info["numFmt"] == "decimal":
                resolved = num_info["resolved"]  # "7." or "7.5."
                # Strip trailing dot for clean number
                clean_number = resolved.rstrip(".")  # "7" or "7.5"
                
                # Prepend number to text for the chunk content
                text = f"{resolved} {text}"

                if num_info["ilvl"] == 0:
                    # Section level (Heading 1)
                    flush_all()
                    current_section = para.text.strip()  # original text WITHOUT number
                    current_section_number = clean_number
                    current_clause = ""
                    current_clause_number = ""
                    level = None  # already handled, skip the level block below
                    
                elif num_info["ilvl"] == 1:
                    # Clause level
                    flush_all()
                    current_clause = extract_clause_name(para) or para.text.strip()
                    current_clause_number = clean_number
                    # Don't set level — this is content start, not a heading
                    # The text goes into the buffer as chunk content
                    current_text_buffer.append(text)
                    level = None  # skip heading block

            if level:
                # Standard heading processing (for docs without numbering)
                flush_all()
                idx = min(level, 3) - 1
                if idx == 0:
                    current_section = text
                    current_section_number = ""
                    current_clause = ""
                    current_clause_number = ""
                elif idx == 1:
                    current_clause = text
                    current_clause_number = ""
                # Clear deeper levels handled by old logic
                
            elif num_info is None or num_info["numFmt"] != "decimal":
                # Regular paragraph or bullet — existing clause detection logic
                clause_num = extract_clause_number(text)
                if not clause_num:
                    label = extract_label(text)
                    if label and para.style.name == "List Paragraph":
                        clause_num = label
                if clause_num and current_text_buffer:
                    flush_buffer()
                    current_clause_number = clause_num
                elif clause_num:
                    current_clause_number = clause_num
                current_text_buffer.append(text)
```

### 2d: Update `_make_chunk()` to use new fields

```python
def _make_chunk(
    text: str,
    doc_id: str,
    doc_title: str,
    doc_filename: str,
    doc_link: str,
    section: str,
    section_number: str,
    clause: str,
    clause_number: str,
    chunk_index: int,
    last_updated: str,
) -> PolicyChunk:
    # Build display string
    parts = []
    if section_number and section:
        parts.append(f"{section_number}. {section}")
    elif section:
        parts.append(section)
    if clause_number and clause:
        parts.append(f"{clause_number}. {clause}")
    elif clause:
        parts.append(clause)
    section_display = " > ".join(parts)

    return PolicyChunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        doc_title=doc_title,
        doc_filename=doc_filename,
        doc_link=doc_link,
        section=section,
        section_number=section_number,
        clause=clause,
        clause_number=clause_number,
        section_display=section_display,
        text=text.strip(),
        char_count=len(text.strip()),
        chunk_index=chunk_index,
        last_updated=last_updated,
    )
```

### 2e: Update all `_emit_chunk` calls

In `_emit_chunk()`, pass the current section/clause variables:

```python
    def _emit_chunk(text: str, clause_num: str):
        nonlocal chunk_index
        parts = _split_oversized(text, settings.chunk_max_tokens)
        for part in parts:
            c = _make_chunk(
                text=part,
                doc_id=doc_id,
                doc_title=doc_title,
                doc_filename=doc_filename,
                doc_link=doc_link,
                section=current_section,
                section_number=current_section_number,
                clause=current_clause,
                clause_number=current_clause_number or clause_num,
                chunk_index=chunk_index,
                last_updated=last_updated,
            )
            chunks.append(c)
            chunk_index += 1
```

## Step 3: Update search result formatting

In `rag/tools/search_policies.py` (and `rag/hybrid_search.py`), update the output format to use the new fields:

```python
# Old format:
f"Document: {doc_title} | Section: {section_display} | Clause: {clause_number}"

# New format:
section_str = f"{section_number}. {section}" if section_number else section
clause_str = f"{clause_number}. {clause}" if clause_number else clause

f"Document: {doc_title} | Section: {section_str} | Clause: {clause_str}"

# Example output:
# Document: Acceptable Use Policy [Internal] | Section: 7. Private Information | Clause: 7.5. Blogging and Social Media
```

## Step 4: Update Qdrant payload indexes

In `rag/vector_store.py`, make sure payload indexes exist for the new fields:

```python
client.create_payload_index(collection, "section", PayloadSchemaType.KEYWORD)
client.create_payload_index(collection, "section_number", PayloadSchemaType.KEYWORD)
client.create_payload_index(collection, "clause", PayloadSchemaType.KEYWORD)
client.create_payload_index(collection, "clause_number", PayloadSchemaType.KEYWORD)
```

## Expected Result

After re-ingestion, a chunk for "Blogging and Social Media" will have:

```json
{
  "doc_title": "Acceptable Use Policy [Internal]",
  "section": "Private Information",
  "section_number": "7",
  "clause": "Blogging and Social Media",
  "clause_number": "7.5",
  "section_display": "7. Private Information > 7.5. Blogging and Social Media",
  "text": "7.5. Blogging and Social Media: Limited and occasional use of Company's facilities..."
}
```

## Verification

After implementing, re-ingest and check:

```python
from rag.vector_store import get_client
from config import settings

client = get_client()
results, _ = client.scroll(
    collection_name=settings.qdrant_collection,
    limit=200, with_payload=True, with_vectors=False,
)

for r in results:
    p = r.payload
    if "blogging" in p.get("text", "").lower():
        print(f"doc_title:      {p['doc_title']}")
        print(f"section:        {p['section']}")
        print(f"section_number: {p['section_number']}")
        print(f"clause:         {p['clause']}")
        print(f"clause_number:  {p['clause_number']}")
        print(f"display:        {p['section_display']}")
        break
```

Expected:
```
doc_title:      Acceptable Use Policy [Internal]
section:        Private Information
section_number: 7
clause:         Blogging and Social Media
clause_number:  7.5
display:        7. Private Information > 7.5. Blogging and Social Media
```

## Files Changed

| File | Action |
|------|--------|
| `ingest/numbering.py` | CREATE (from FIX_DOCX_PARSER_NUMBERING.md) |
| `ingest/chunk_models.py` | MODIFY — replace section_path with section, section_number, clause, clause_number |
| `ingest/docx_parser.py` | MODIFY — add resolver, track section/clause separately, extract clause name from bold runs |
| `rag/vector_store.py` | MODIFY — add payload indexes for new fields |
| `rag/tools/search_policies.py` | MODIFY — update output format |
| `rag/hybrid_search.py` | MODIFY — update HybridResult and output format |

## Re-Ingest Required

After deploying: `python scripts/ingest_all.py --folder ./policies`
