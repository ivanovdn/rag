# DOCX Parser Fixes — Step by Step

Apply these fixes to `ingest/docx_parser.py` in order.
After all fixes, re-ingest all documents and re-run Tier 1 evaluation.

---

## Fix 1: Add bold-as-heading detection function

**Problem:** Documents like "Privacy Notice for Team Members" use `style='Normal'` with bold formatting for headings instead of Heading 1/2/3 styles. The parser sees zero structure, all chunks get empty section/clause.

**Add this function** after `extract_clause_name()` (around line 65):

```python
def _is_bold_heading(para, num_info) -> bool:
    """
    Detect bold-as-heading pattern: short, all-bold Normal paragraphs
    that serve as section headings in documents without Heading styles.

    Heuristics:
    - Style is "Normal" (not already a Heading)
    - All text runs are bold
    - Short text (≤ 10 words)
    - Not a numbered paragraph (those are handled separately)
    - Not a metadata line (version, date, etc.)
    """
    if para.style.name != "Normal":
        return False

    # Skip if has Word auto-numbering (handled by NumberingResolver)
    if num_info and num_info["numFmt"] == "decimal":
        return False

    text = para.text.strip()
    if not text:
        return False

    # Must have runs, all bold
    text_runs = [r for r in para.runs if r.text.strip()]
    if not text_runs:
        return False
    if not all(r.bold for r in text_runs):
        return False

    # Short text only (headings are brief)
    if len(text.split()) > 10:
        return False

    # Skip metadata lines
    skip_prefixes = {"version", "date", "created by", "approved by", "managed by", "sensitivity"}
    if any(text.lower().startswith(p) for p in skip_prefixes):
        return False

    # Skip if it looks like a clause number (e.g., "4.2 Something")
    if re.match(r"^\d+(\.\d+)*\.?\s", text):
        return False

    return True
```

---

## Fix 2: Integrate bold-as-heading into the main parsing loop

**In the main `for element in doc.element.body:` loop, AFTER the `if level:` block and BEFORE the `elif num_info is None...` block (around line 295), add a new elif:**

Replace this section (lines 283-308):

```python
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

With this:

```python
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

            elif _is_bold_heading(para, num_info):
                # Bold-as-heading: treat as section heading
                flush_all()
                current_section = text
                current_section_number = ""
                current_clause = ""
                current_clause_number = ""

            elif num_info is None or num_info["numFmt"] != "decimal":
                # Regular paragraph or bullet — clause detection logic
                clause_num = extract_clause_number(text)
                if not clause_num:
                    label = extract_label(text)
                    if label and para.style.name == "List Paragraph":
                        # FIX 3: Label goes to clause NAME, not clause_number
                        flush_buffer()
                        current_clause = label
                        current_clause_number = ""
                        current_text_buffer.append(text)
                        continue
                if clause_num and current_text_buffer:
                    flush_buffer()
                    current_clause_number = clause_num
                elif clause_num:
                    current_clause_number = clause_num
                current_text_buffer.append(text)
```

Note **Fix 3** is embedded here: when `extract_label` finds a label like "Ongoing Awareness Activities", it now goes to `current_clause` (the name field) instead of `current_clause_number`.

---

## Fix 4: Fix section name pollution for ilvl=0 paragraphs

**Problem:** Line 268 uses the full paragraph text as section name. For paragraphs like "There are the following types of awareness and training within Company which Team Members must complete:", the entire sentence becomes the section name.

**Replace lines 265-272:**

```python
                if num_info["ilvl"] == 0:
                    # Section level (Heading 1)
                    flush_all()
                    current_section = para.text.strip()  # original text WITHOUT number
                    current_section_number = clean_number
                    current_clause = ""
                    current_clause_number = ""
                    level = None  # already handled, skip the level block below
```

With:

```python
                if num_info["ilvl"] == 0:
                    # Section level
                    flush_all()
                    # Use bold text as section name if available (cleaner),
                    # otherwise fall back to full paragraph text
                    section_name = extract_clause_name(para)  # gets bold prefix
                    if not section_name:
                        # Truncate long section names to first phrase
                        raw = para.text.strip()
                        if len(raw.split()) > 8:
                            # Take up to first colon, comma, or period
                            for sep in [":", ",", "."]:
                                if sep in raw:
                                    raw = raw[:raw.index(sep)].strip()
                                    break
                        section_name = raw
                    current_section = section_name
                    current_section_number = clean_number
                    current_clause = ""
                    current_clause_number = ""
                    level = None
```

This handles long section text like "There are the following types of awareness and training within Company which Team Members must complete:" → truncated to "There are the following types of awareness and training within Company which Team Members must complete" or uses the bold prefix if available.

---

## Fix 5: Filter noise chunks at the end

**Add this BEFORE `return chunks` at the end of `parse_docx()` (line 318):**

```python
    flush_all()

    # ── Filter noise chunks ──
    NOISE_SECTIONS = {
        "revision history",
        "change history",
        "table of contents",
        "document control",
        "version control",
        "approval history",
    }

    filtered = []
    for c in chunks:
        section_lower = (c.section or "").lower().strip()

        # Skip noise sections
        if section_lower in NOISE_SECTIONS:
            continue

        # Skip title-only chunks (document name with no real content)
        if not c.section and not c.clause and _estimate_tokens(c.text) < 15:
            continue

        # Skip near-empty chunks
        if len(c.text.strip()) < 20:
            continue

        filtered.append(c)

    # Re-index
    for i, c in enumerate(filtered):
        c.chunk_index = i

    return filtered
```

---

## Testing after fixes

### Test 1: Privacy Notice (was all empty sections)

```python
from ingest.docx_parser import parse_docx
from pathlib import Path

chunks = parse_docx(
    Path("policies/Privacy Notice for Team Members [Internal].docx"),
    "http://test"
)
print(f"Total chunks: {len(chunks)}")
for c in chunks:
    print(f"  [{c.chunk_index:2d}] section='{c.section}' | clause='{c.clause}' | clause_num='{c.clause_number}' | text={c.text[:80]}...")
```

**Expected:** Sections like "Sharing your personal data", "Data storage", "Your privacy rights" — not empty.

### Test 2: Awareness and Training (was polluted section names)

```python
chunks = parse_docx(
    Path("policies/Awareness and Training Policy [Internal].docx"),
    "http://test"
)
for c in chunks:
    if c.section_number == "4.2":
        print(f"  section='{c.section}' | clause='{c.clause}' | clause_num='{c.clause_number}'")
```

**Expected:** `clause='Ongoing Awareness Activities'`, `clause_number=''` — not `clause_number='Ongoing Awareness Activities'`.

### Test 3: Noise filtering

```python
# Check that Revision History chunks are gone
noise = [c for c in chunks if "revision" in c.section.lower() or "change history" in c.section.lower()]
print(f"Noise chunks remaining: {len(noise)}")
```

**Expected:** 0

### Test 4: Re-run Tier 1 after re-ingesting

```bash
# Re-ingest all documents
python scripts/ingest_all.py --folder ./policies

# Re-run Tier 1
python eval/run_experiment.py --tier tier1 --name post-parser-fix-v1
```

Compare `hit_evaluator` and `mrr_evaluator` against `baseline-hybrid-v1` in Phoenix. The 6 misses should decrease — especially the Privacy Notice, Organization Roles, and Customer Grading questions.

---

## Summary of changes

| Fix | What | Lines affected | Impact |
|-----|------|---------------|--------|
| 1 | Bold-as-heading detection function | New function after line 65 | Fixes documents with no Heading styles |
| 2 | Integrate into main loop | Replace lines 283-308 | Routes bold headings to section detection |
| 3 | Label → clause name, not number | Inside fix 2 (label handling) | Fixes `clause_number='Ongoing Activities'` |
| 4 | Clean ilvl=0 section names | Replace lines 265-272 | Fixes long sentence section names |
| 5 | Noise chunk filter | Before `return chunks` | Removes Revision/Change History chunks |
