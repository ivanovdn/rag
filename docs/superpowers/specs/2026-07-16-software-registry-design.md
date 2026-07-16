# Software Registry — Design Spec

**Date:** 2026-07-16
**Status:** Approved (design); implementation plan pending
**Feature:** Answer employee questions about which software is **allowed** or **forbidden**, sourced from two curated lists ("Allowed Software", "Forbidden Software"), integrated into the existing Agentic-RAG compliance bot.

---

## 1. Motivation & problem framing

Two new source documents (`data/Allowed Software.pdf`, `data/Forbidden Software.pdf`) are flat lookup **tables**, fundamentally different from the heading/clause-structured policy DOCX corpus:

- **Allowed Software** — long alphabetical table: `Name | Comment | Status(=Allowed)`. Hundreds of rows; comment often empty or a caveat ("With request to IT", "A license is required for commercial usage").
- **Forbidden Software** — shorter table: `Name | Category | Description | Alternatives | Remarks | Status(=Forbidden)`. ~15 rows, some with merged cells in the Alternatives column.

A software question — "Can I use Docker?" — is a **name lookup against a known list**, not semantic retrieval over paragraphs. The existing `PolicyChunk` pipeline (heading-aware chunking, `NumberingResolver`, `[Source N]` citations with clause numbers) does not fit this data. Two query shapes exist:

| Query shape | Example | Strategy |
|---|---|---|
| **Specific software** | "Is Docker allowed?" | Exact/fuzzy **name lookup** across both tables → return that row verbatim |
| **Category / "what can I use for X"** | "What VPN can I use?" | **Semantic search over the Forbidden table's Category+Description+Alternatives** → the ban + the curated alternative |

Key insight: the company already curated the answer to every "what can I use instead?" question — it is the **Alternatives column of the Forbidden table**. The Allowed table is a flat whitelist for "is X approved?" verification.

Worked examples (from the real data):
- **"What VPN can I use?"** → Forbidden row "Any personal/non-corporate VPN (ClearVPN, ExpressVPN, NordVPN…)" → Alternatives: **Corporate Cisco AnyConnect VPN**; Allowed row "Cisco AnyConnect (VPN)". Answer: use Cisco AnyConnect.
- **"What IDE can I use?"** → Forbidden row "JetBrains Software (IntelliJ IDEA, PyCharm, WebStorm, Rider…)" → Alternatives: **MS Visual Studio, VS Code**; Remarks: "Allowed to use until expiration of existing licenses". Answer: VS Code / Visual Studio (also Eclipse, Android Studio); JetBrains phased out.

---

## 2. Decisions (locked)

1. **Mechanism: Hybrid** — fuzzy name lookup + a semantic index over the registry rows (forbidden rows carry the curated alternatives). Not pure vector search (weak at exact-name matching), not pure lookup (misses category questions).
2. **Routing: Agent tool** — add `check_software` as a 3rd retrieval tool alongside `search_policies`. The **router stays a pure gatekeeper** (unchanged). "Is Docker allowed?" is already `in_scope` and passes through. This is native to the existing `AgentWorkflow` multi-tool design and handles blended questions in one shot; it avoids scope-creeping the router's safe-default invariant.
3. **Storage: separate Qdrant collection `software_registry`** — holds every row (allowed + forbidden) with full payload + embedding of the row's descriptive text. Same embedding model/dim as policies (no config change). Isolated from `compliance_policies` (different schema; avoids polluting policy semantic search with one-word rows).
4. **Source of truth: committed CSV/JSON canonical** — convert the PDFs once into a reviewable `software/software_registry.json`; ingest reads that. **Future work (documented):** later the source may be Excel / Confluence / SharePoint — the ingest is structured to accept a pluggable loader producing the same JSON shape. This future-source note also applies to the policy corpus.
5. **Not-found behavior: canned "ask IT" reply with fuzzy suggestion** — deterministic response pointing to IT (`support@trinetix.com`), with a "did you mean X?" if a close fuzzy match exists. No heavyweight compliance escalation (software approval is an IT function). No rating prompt, no feedback row.

---

## 3. Data model & source of truth

Canonical file: `software/software_registry.json` — an array of row objects:

```json
{
  "name": "Docker",
  "status": "allowed",
  "category": "",
  "note": "A license is required for commercial usage",
  "alternative": "",
  "aliases": [],
  "source_list": "Allowed Software"
}
```

```json
{
  "name": "TeamViewer",
  "status": "forbidden",
  "category": "Prohibited software - remote desktop control",
  "note": "Remote desktop software pose significant risk of unauthorized access and data leakage",
  "alternative": "Microsoft Teams remote desktop sharing and control",
  "aliases": [],
  "source_list": "Forbidden Software"
}
```

Field mapping:
- **Allowed** rows → `note` = Comment column; `category`/`alternative` empty.
- **Forbidden** rows → `category` = Category, `note` = Description merged with Remarks, `alternative` = Alternatives.
- Multi-name forbidden rows (JetBrains bundle; "Any personal VPN (X, Y, Z…)") keep the full descriptive `name` **and** populate `aliases: [...]` with the individual tool names (IntelliJ, PyCharm, ExpressVPN, …) so name lookup resolves each one.

Bootstrap: `scripts/build_software_registry.py` uses `pdfplumber` to parse the two PDFs into the JSON (one-time / on-demand). Output is human-reviewed and committed. Thereafter the JSON is the source of truth — editable and git-diffable. `pdfplumber` is a bootstrap-only dependency (not on the runtime path).

**Correctness note:** because a mis-parsed row flips an allowed/forbidden answer, the committed JSON is reviewed by a human, and `tests/docs/` asserts known rows.

---

## 4. Storage & ingest

- **Collection `software_registry`** (config `software_collection`). Vectors: same model/dim as policies (Cosine). Payload = the full row. Payload indexes on `name` (keyword), `status` (keyword), `category` (keyword).
- `scripts/ingest_software.py` reads `software/software_registry.json`, builds per-row embedding text:
  - forbidden: `"{name} | {category} | {note} | Alternative: {alternative}"`
  - allowed: `"{name}{ ' | ' + note if note}"`
  - embeds via the existing `rag.embeddings.embed_texts`, upserts to `software_registry`.
- **`rag/vector_store.py` change (small, safe):** `init_collection`, `search_vectors`, and `scroll_by_filter` take an optional `collection_name` argument defaulting to `settings.qdrant_collection`, so the software path reuses the same client/plumbing. A `init_software_collection()` helper (or parametrized `init_collection`) creates the collection + the software payload indexes.

---

## 5. `check_software` tool (hybrid)

New module `rag/tools/check_software.py`. Signature: `check_software(name: str) -> str`.

The agent extracts and passes the specific software name ("Docker", "TeamViewer") or a short category phrase ("personal VPN", "IDE"). Tool description guides this (contrast with `search_policies`, which receives the original question verbatim).

Algorithm:
1. **Name index** — lazily built module-level cache, populated by scrolling the `software_registry` collection (single runtime source of truth = Qdrant, consistent with policies). Normalize (lowercase, trim, collapse whitespace) and check `name` + `aliases`. Exact match wins; else `rapidfuzz` fuzzy match against all names/aliases, accepted if score ≥ `SOFTWARE_FUZZY_THRESHOLD`.
2. Confident name hit → return that row. Forbidden rows include the `alternative`.
3. No confident name → **semantic search** over the collection (`embed_query` → `search_vectors(collection_name=software_registry, top_k≈5)`). This is the "what VPN/IDE can I use?" path.
4. Nothing confident either way → `SOFTWARE_NOT_LISTED`, plus the closest fuzzy candidate (name + status) as a "did you mean" suggestion.

Output format returned to the agent (verbatim-friendly, mirrors `format_sources`):

```
=== SOFTWARE REGISTRY RESULTS ===

[Software 1] Docker
Status: allowed
List: Allowed Software
Note: A license is required for commercial usage
```

For forbidden rows, include `Category:` and `Alternative:` lines. When not listed, return `SOFTWARE_NOT_LISTED` (optionally with `DID_YOU_MEAN: WinRAR (forbidden)`).

**Resilience (mirrors `search_policies`):** `embed_query` + `search_vectors` wrapped in `retry_transient`; on transient failure, set module flag `_software_unavailable = True` and return a sentinel string; record an `infra_unavailable` Phoenix span (`failed_component="software_embeddings"|"software_qdrant"`). Non-transient errors propagate (AgentWorkflow swallows tool exceptions → agent escalates, as today).

**Not-found flag:** on `SOFTWARE_NOT_LISTED`, set module flag `_software_not_found = True` and store the suggestion, for the deterministic canned reply (see §7).

---

## 6. Agent integration

- `check_software_tool = FunctionTool.from_defaults(fn=check_software)` added to `ALL_TOOLS` in `rag/agent.py` (after `search_policies_tool`, before `escalate`). `check_software` is guarded by `software_lookup_enabled`.
- **`SYSTEM_PROMPT`** gains a tool-selection branch:
  - Specific-software or "what can I use for X" → call `check_software` (pass the software name or category keyword, not the whole sentence).
  - Policy/rule question → call `search_policies` (unchanged: pass the original question verbatim).
  - Blended questions may call both.
  - Software rules: state the status **verbatim**, cite the source list, use the forbidden row's `alternative`, never infer allowed/forbidden, and if `SOFTWARE_NOT_LISTED` relay the not-listed guidance and do not speculate ("not forbidden" ≠ "allowed").
  - `OUTPUT FORMAT` documents the two new optional citation fields.
- **`ComplianceAnswer.Citation`** gains two **optional** fields: `status` (`"allowed" | "forbidden" | ""`) and `alternative` (default `""`). Policy citations leave them empty. Single schema, single parser — `rag/response.py` `parse_agent_response` needs no change (it passes citations through).

---

## 7. Rendering & outcomes (Teams)

- **`render_answer`** branches per citation: a citation carrying a non-empty `status` renders the **software layout**:
  - `✅ Allowed — Allowed Software` or `⛔ Forbidden — <category>`
  - the verbatim `note` as the quote (`<i>"…"</i>`)
  - an `Alternative: …` line for forbidden rows
  - Uses only Teams-limited HTML (`<p> <b> <i> <ul>/<li> <hr>`).
  - A citation without `status` renders today's policy layout, untouched.
- **Not-found outcome** — distinct, treated like `unavailable`:
  - `check_software` sets `_software_not_found` (+ suggestion).
  - `_run_rag` checks the flag **after** parsing the agent response and returns `{"status": "software_not_found", "suggestion": ...}` **only when the parsed answer has no citations** — so a blended answer that produced a real policy citation still renders normally with a rating prompt.
  - `_send_reply` handles `status == "software_not_found"` → renders a new canned reply (in `renderer.py`, e.g. `render_software_not_listed(name, suggestion)`): *"'Foo' isn't on the approved or forbidden software list. Did you mean **WinRAR** (Forbidden)? Otherwise request approval from IT before installing (support@trinetix.com)."* — **no rating prompt, no feedback row** (matches the `unavailable` treatment).
- **Transient failure** during `check_software` → existing `render_unavailable()` path (`{"status": "unavailable"}`).

`_run_rag` must reset `_software_unavailable` / `_software_not_found` at the start of each run (as it already does for `sp._retrieval_unavailable`) and check `_software_unavailable` alongside the retrieval flag.

---

## 8. Config, data location, dependencies

- **Config (`config.py` + `.env.example`):**
  - `software_lookup_enabled: bool = True` — kill switch (drops the tool from `ALL_TOOLS` when off).
  - `software_collection: str = "software_registry"`.
  - `software_fuzzy_threshold: float = 85.0` — rapidfuzz acceptance score on the 0–100 scale (`fuzz`/`process.extractOne`). A candidate name/alias is a confident match only at or above this.
- **Dependencies:** `rapidfuzz` (runtime, fuzzy matching); `pdfplumber` (bootstrap script only).
- **Data location:** move the two PDFs into a new `software/` folder (mirrors `policies/`), alongside the generated `software_registry.json`.
- **Commands (new):**
  ```bash
  PYTHONPATH=. python scripts/build_software_registry.py   # PDF -> software/software_registry.json (one-time)
  PYTHONPATH=. python scripts/ingest_software.py            # JSON -> software_registry collection
  ```

---

## 9. Error handling summary

| Situation | Behavior |
|---|---|
| Specific software found | Verbatim status + list + note (+ alternative if forbidden); rating prompt |
| Category question | Semantic hit(s), forbidden row supplies the alternative; rating prompt |
| Software not on either list | Canned "ask IT" reply + fuzzy "did you mean"; **no** rating prompt |
| Transient backend failure in `check_software` | `unavailable` reply (retried first); no rating prompt |
| Non-transient tool error | AgentWorkflow swallows → agent escalates (existing behavior) |
| Blended (software not-found + policy found) | Renders the policy answer normally (not-found short-circuit suppressed when citations exist) |

---

## 10. Testing

- **`tests/unit/`** (pure logic, no LLM): normalize / exact / fuzzy / did-you-mean; tool decision logic (found-allowed, found-forbidden+alternative, not-listed+suggestion) against a fake in-memory registry; `render_answer` software-layout branch; `render_software_not_listed`.
- **`tests/docs/`** (corpus guard): parse the committed `software_registry.json`; assert row counts, required fields present, and known rows — Docker=allowed, TeamViewer=forbidden, JetBrains alternative contains "VS Code", personal-VPN alternative contains "Cisco AnyConnect".
- **`tests/live/`** (accuracy, auto-skips without an LLM): "Is Docker allowed?" → allowed; "What VPN can I use?" → Cisco AnyConnect / personal VPN forbidden; unknown tool → not-listed. Keep any `rag.*` imports function-local (same pytest-collection gotcha as the router live test).

---

## 11. Known limitation

Categorical "what can I use for X?" is reliable only where the Forbidden table has a curated alternative (VPN, IDE, archiver, remote-desktop, antivirus). Categories with only allowed-side entries and no forbidden counterpart (e.g. "what screen recorder can I use?") get weak semantic matches, because the Allowed table has **no category column**. Deferred; the fix is later allowed-side tagging (LLM-assisted or manual `category`/`tags` on allowed rows at ingest).

---

## 12. Out of scope (v1)

- Allowed-side category tagging (see §11).
- Pluggable Excel / Confluence / SharePoint source loaders (structure for it; do not build).
- Router-level software category (explicitly rejected — see decision 2).
