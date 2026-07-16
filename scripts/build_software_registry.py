"""One-time bootstrap: parse the Allowed/Forbidden Software PDFs into
software/software_registry.json. Best-effort — the Forbidden table has merged
cells; review and hand-correct the JSON (especially `alternative` and `aliases`)
before committing. The JSON, not the PDF, is the canonical source of truth.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber  # noqa: E402

from config import settings  # noqa: E402


def _clean(cell: str | None) -> str:
    return " ".join((cell or "").split())


def _extract_aliases(name: str) -> list[str]:
    """Pull example tool names out of a descriptive forbidden name, e.g.
    'Any torrent client (uTorrent, BitTorrent, qBittorrent)' -> [uTorrent, ...].
    Also splits multi-line bundles (JetBrains) on newlines."""
    aliases: list[str] = []
    paren = re.search(r"\(([^)]*)\)", name)
    if paren:
        inner = re.sub(r"(?i)\b(for example|e\.g\.|etc\.?)\b", "", paren.group(1))
        aliases += [a.strip() for a in re.split(r"[,;]", inner) if a.strip()]
    return aliases


def parse_allowed(pdf_path: Path) -> list[dict]:
    rows: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for r in table:
                    name = _clean(r[0]) if r else ""
                    if not name or name.lower() == "name":
                        continue
                    note = _clean(r[1]) if len(r) > 1 else ""
                    rows.append({
                        "name": name, "status": "allowed", "category": "",
                        "note": note, "alternative": "", "aliases": [],
                        "source_list": "Allowed Software",
                    })
    return rows


def parse_forbidden(pdf_path: Path) -> list[dict]:
    rows: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for r in table:
                    name = _clean(r[0]) if r else ""
                    if not name or name.lower() == "name":
                        continue
                    category = _clean(r[1]) if len(r) > 1 else ""
                    description = _clean(r[2]) if len(r) > 2 else ""
                    alternative = _clean(r[3]) if len(r) > 3 else ""
                    remarks = _clean(r[4]) if len(r) > 4 else ""
                    note = " ".join(p for p in (description, remarks) if p)
                    rows.append({
                        "name": name, "status": "forbidden", "category": category,
                        "note": note, "alternative": alternative,
                        "aliases": _extract_aliases(name),
                        "source_list": "Forbidden Software",
                    })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Bootstrap software_registry.json from PDFs")
    parser.add_argument("--folder", default=settings.software_docs_folder)
    parser.add_argument("--out", default=settings.software_registry_path)
    args = parser.parse_args()

    folder = Path(args.folder)
    rows = parse_allowed(folder / "Allowed Software.pdf") + parse_forbidden(folder / "Forbidden Software.pdf")
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.out}. REVIEW merged cells + aliases before committing.")


if __name__ == "__main__":
    main()
