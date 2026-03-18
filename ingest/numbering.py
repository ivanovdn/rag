"""
Resolve auto-generated numbering from Word's numbering.xml.

Word stores numbering definitions separately from paragraph text.
When you see "7.5. Blogging and Social Media:" in the rendered document
but para.text only contains "Blogging and Social Media:", the "7.5."
comes from the numbering XML.

This module reconstructs those numbers by simulating Word's counters.

Usage:
    from ingest.numbering import NumberingResolver

    resolver = NumberingResolver(doc)

    for element in doc.element.body:
        num_info = resolver.resolve(element)
        # num_info = {"resolved": "7.5.", "ilvl": 1, "numFmt": "decimal"}
        # or None if no numbering
"""

from docx import Document
from docx.oxml.ns import qn


class NumberingResolver:
    """
    Resolves Word auto-numbering for each paragraph.

    Initialize once per document, then call resolve(element) for each
    paragraph element to get its resolved number string.
    """

    def __init__(self, doc: Document):
        self.num_to_abstract: dict[str, str] = {}
        self.abstract_levels: dict[str, dict] = {}
        self.counters: dict[str, dict[str, int]] = {}

        self._parse_numbering(doc)

    def _parse_numbering(self, doc: Document) -> None:
        """Parse numbering.xml to extract definitions."""
        try:
            numbering_part = doc.part.numbering_part
            if numbering_part is None:
                return
            numbering_xml = numbering_part._element
        except Exception:
            return

        # Map numId -> abstractNumId
        for num_el in numbering_xml.findall(qn("w:num")):
            num_id = num_el.get(qn("w:numId"))
            abstract_ref = num_el.find(qn("w:abstractNumId"))
            if abstract_ref is not None:
                self.num_to_abstract[num_id] = abstract_ref.get(qn("w:val"))

        # Map abstractNumId -> level definitions
        for abstract_el in numbering_xml.findall(qn("w:abstractNum")):
            abstract_id = abstract_el.get(qn("w:abstractNumId"))
            levels = {}
            for lvl_el in abstract_el.findall(qn("w:lvl")):
                ilvl = lvl_el.get(qn("w:ilvl"))
                num_fmt_el = lvl_el.find(qn("w:numFmt"))
                lvl_text_el = lvl_el.find(qn("w:lvlText"))
                start_el = lvl_el.find(qn("w:start"))
                levels[ilvl] = {
                    "numFmt": num_fmt_el.get(qn("w:val")) if num_fmt_el is not None else "decimal",
                    "lvlText": lvl_text_el.get(qn("w:val")) if lvl_text_el is not None else "",
                    "start": int(start_el.get(qn("w:val"))) if start_el is not None else 1,
                }
            self.abstract_levels[abstract_id] = levels

    def resolve(self, element) -> dict | None:
        """
        Resolve numbering for a paragraph element.

        Returns dict with:
            resolved: str  - the number string, e.g. "7.5."
            ilvl: int      - indent level (0=section, 1=clause, 2+=bullets)
            numFmt: str    - "decimal" or "bullet"

        Returns None if the paragraph has no numbering.
        """
        pPr = element.find(qn("w:pPr"))
        if pPr is None:
            return None

        numPr = pPr.find(qn("w:numPr"))
        if numPr is None:
            return None

        ilvl_el = numPr.find(qn("w:ilvl"))
        numId_el = numPr.find(qn("w:numId"))

        if ilvl_el is None or numId_el is None:
            return None

        ilvl = ilvl_el.get(qn("w:val"))
        num_id = numId_el.get(qn("w:val"))
        ilvl_int = int(ilvl)

        # Get level definitions
        abstract_id = self.num_to_abstract.get(num_id, "0")
        levels = self.abstract_levels.get(abstract_id, {})
        level_def = levels.get(ilvl, {})
        num_fmt = level_def.get("numFmt", "decimal")

        # Initialize counters for this numId
        if num_id not in self.counters:
            self.counters[num_id] = {}

        # Initialize level counter if needed
        if ilvl not in self.counters[num_id]:
            start = level_def.get("start", 1)
            self.counters[num_id][ilvl] = start
        else:
            self.counters[num_id][ilvl] += 1

        # Reset deeper levels
        for deeper_ilvl in list(self.counters[num_id].keys()):
            if int(deeper_ilvl) > ilvl_int:
                deeper_start = levels.get(deeper_ilvl, {}).get("start", 1)
                self.counters[num_id][deeper_ilvl] = deeper_start - 1

        # Build number string from lvlText template
        lvl_text = level_def.get("lvlText", "")
        resolved = lvl_text
        for lvl_idx in range(ilvl_int + 1):
            lvl_str = str(lvl_idx)
            placeholder = f"%{lvl_idx + 1}"
            if lvl_str in self.counters[num_id]:
                resolved = resolved.replace(placeholder, str(self.counters[num_id][lvl_str]))

        return {
            "resolved": resolved,
            "ilvl": ilvl_int,
            "numFmt": num_fmt,
        }
