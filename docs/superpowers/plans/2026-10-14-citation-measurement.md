# Citation measurement (#10, steps 1-3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the eval measure what the answering model *cites*, fix the labels that mark correct answers wrong, and add a small set of role-mismatch questions, so that a later answer-prompt change can be judged on numbers. This plan ends with a new baseline. It changes no bot behaviour.

**Architecture:** Three `citation_*` evaluators already exist in `eval/evaluators.py` and were taken out of the run groups on 2026-09-25. Putting them back is a list edit. `eval/compare_runs.py` gains a per-question citation status (cited / selection miss / retrieval miss / escalated) built on `citation_clause_accuracy`, so the script and the Phoenix column cannot disagree. Four chatbot labels get alternatives plus `match_mode: any`, uploaded as `chatbot-test-v2` so the v1 history stays comparable. A new 6-question `role-mismatch-v1` dataset covers the one real defect #10 found. Every label is checked against the parsed corpus by a corpus test.

**Tech Stack:** Python 3.12, pytest, Phoenix datasets/experiments (`scripts/make_dataset.py`, `eval/run_experiment.py`), `ingest/docx_parser.py` for the corpus check.

**Spec:** `docs/superpowers/specs/2026-10-06-one-candidate-knob-results.md`, item **10** ("The model cites the wrong retrieved chunk — mostly a labelling problem; one real defect, a role mismatch"), its "When picked up, in this order" list, steps 1-3. Re-measured 2026-10-14 on the deployed configuration (`RXhwZXJpbWVudDoyNQ==`, design `2026-10-09-query-rephrasing-design.md`): 6 of 61 answers miss the expected clause. In 4 of them the clause was in the sources at rank 1-2: auto-lock (CB-032), Teams chats (CB-014), colleague sharing files (CB-016), and "data wasn't very sensitive" (CB-035, the role mismatch). Step 4, the prompt rule, is a separate plan written after this baseline exists.

## Global Constraints

- No change to `rag/`, `channels/` or `config.py`. The bot's behaviour is untouched.
- Tests never touch the network or `172.20.0.22`. Corpus tests live in `tests/docs/` and auto-skip without `policies/`.
- `scripts/make_dataset.py` runs without `--overwrite`. `chatbot-test-v1` stays as it is, and the relabelled set is a new dataset `chatbot-test-v2`.
- Labels are the owner's ground truth. Tasks 3 and 4 stop for the user's sign-off before their commit.
- VM commands are given one at a time, and the user runs each one (no loops).
- Imports at the top of the module.
- `CLAUDE.md` stays ≤150 lines.
- No new dependencies.

## Review Focus

1. **An escalated answer with no citations.** Expected: status `escalated`, never a selection or retrieval miss, so a refusal can't be read as the model picking the wrong clause.
2. **A case whose expectations carry no clause** (`citation_clause_accuracy` returns `skip`, score 1.0). Expected: status `unlabelled`, not `cited`, so the count of correct citations never includes questions that weren't checked.
3. **A relabel alternative whose clause name is a substring of an unrelated clause in the same document** (e.g. "Exceptions"). Expected: the corpus test pins every alternative's (doc, section, clause) to a real chunk. Matching stays substring-based, as everywhere else in the evaluators.
4. **Comparing runs on `chatbot-test-v1` with runs on `-v2`.** Expected: `compare_runs` uses each run's own `reference_output`, so a label change shows up as a status change on that question. The doc records that v1 and v2 numbers aren't comparable.
5. **`match_mode` lost on upload.** Expected: `map_chatbot` copies it already (fixed 2026-09-25). Task 3's test checks the mapped output of the relabelled cases, not just the JSON file.

---

### Task 1: Put the citation evaluators back in the chatbot and tier2 runs

**Files:**
- Modify: `eval/evaluators.py` (the `GENERATION_EVALUATORS` group and the comment above it)
- Test: `tests/unit/test_evaluators.py`

**Interfaces:**
- Produces: `CHATBOT_EVALUATORS` and `TIER2_EVALUATORS` now include `citation_doc_accuracy`, `citation_section_accuracy` and `citation_clause_accuracy`. `TIER1_EVALUATORS` is unchanged, since tier1 produces no citations.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_evaluators.py`; add the three names to its existing `from eval.evaluators import (...)` block if missing: `CHATBOT_EVALUATORS, TIER1_EVALUATORS, TIER2_EVALUATORS, citation_doc_accuracy, citation_section_accuracy`)

```python
def test_answer_tiers_score_what_the_model_cited():
    """#10: with these off, a prompt change could degrade citations while every
    reported number stayed the same (they were dropped 2026-09-25)."""
    cited = {citation_doc_accuracy, citation_section_accuracy, citation_clause_accuracy}
    assert cited <= set(CHATBOT_EVALUATORS)
    assert cited <= set(TIER2_EVALUATORS)
    assert not cited & set(TIER1_EVALUATORS)  # tier1 produces no citations
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_evaluators.py -o addopts="" -q -p no:warnings`
Expected: FAIL on `cited <= set(CHATBOT_EVALUATORS)`

- [ ] **Step 3: Implement**. In `eval/evaluators.py`, replace the comment block above `GENERATION_EVALUATORS` and the group itself:

```python
# answer_coverage stays OUT (2026-09-25): it is a word-overlap heuristic that scores
# an answer by how many significant words of a human-written expected_answer summary
# appear in it. This system quotes policy text VERBATIM rather than paraphrasing, so
# the two strings legitimately diverge -- it measured 0.4754 with 32 zeros on a run
# whose citations were 0.81-0.88. It penalises the intended behaviour.
#
# The citation_* three are back IN (2026-10-14, #10). They measure what the MODEL
# cited, which is not what RETRIEVAL returned (hit/mrr): with them off, a prompt
# change could degrade citations with every reported number unchanged. They ignore
# extra citations, so an answer citing more than expected is not punished. Last
# values before they were dropped, chatbot-test-v1 2026-09-25 on the search-first
# branch: doc 0.8770, section 0.8443, clause 0.8115.
GENERATION_EVALUATORS = [
    json_parse_success,
    citation_doc_accuracy,
    citation_section_accuracy,
    citation_clause_accuracy,
]
```

- [ ] **Step 4: Run the suite**

Run: `PYTHONPATH=. pytest tests -o addopts="" -q -p no:warnings`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add eval/evaluators.py tests/unit/test_evaluators.py
git commit -m "feat(eval): score what the model cited again (#10)"
```

### Task 2: Citation status per question in `compare_runs`

**Files:**
- Modify: `eval/compare_runs.py`
- Test: `tests/unit/test_compare_runs.py`

**Interfaces:**
- Consumes: `citation_clause_accuracy(output, expected) -> {"score", "label", ...}` and `first_match_rank` (same module).
- Produces: `citation_status(output: dict | None, reference: dict) -> str`, one of `"cited" | "selection" | "retrieval" | "escalated" | "unlabelled"`. `compare()` adds `out["citations"] = {"gained": [(q, base_status, cand_status)], "lost": [...], "base": Counter, "cand": Counter}`. "gained" means the status became `cited` and "lost" means it stopped being `cited`. Errored pairs are excluded, as for ranks.

Definitions, in order of precedence:
- `escalated`: `output.escalation.needed` is true and there are no citations.
- `unlabelled`: `citation_clause_accuracy` returns label `skip` (no expected clause).
- `cited`: `citation_clause_accuracy` score == 1.0. This follows the case's own `match_mode`: with `any`, one valid clause is enough; with `all`, every expected clause is needed.
- `selection`: not cited, but an expected clause is in `search_results` (the model had it and cited something else).
- `retrieval`: not cited, and no expected clause was retrieved.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_compare_runs.py`; extend its import to `from eval.compare_runs import citation_status, compare, first_match_rank`)

```python
HIT = {"doc_title": "Doc A", "section": "Sec", "clause": "Cl"}
OTHER = {"doc_title": "Doc B", "section": "Sec", "clause": "Other"}


def _out(citations, results, escalated=False):
    return {"citations": citations, "search_results": results,
            "escalation": {"needed": escalated, "reason": ""}}


def test_citation_status_separates_selection_from_retrieval():
    assert citation_status(_out([HIT], [HIT]), REF) == "cited"
    assert citation_status(_out([OTHER], [OTHER, HIT]), REF) == "selection"
    assert citation_status(_out([OTHER], [OTHER]), REF) == "retrieval"


def test_an_escalation_without_citations_is_not_a_selection_miss():
    """Review focus 1: a refusal is not the model picking the wrong clause."""
    assert citation_status(_out([], [HIT], escalated=True), REF) == "escalated"


def test_a_case_with_no_expected_clause_is_never_counted_as_cited():
    """Review focus 2: citation_clause_accuracy scores these 1.0 ('skip')."""
    ref = {"expected_citations": [{"doc_id": "Doc A", "section": "Sec", "clause": ""}]}
    assert citation_status(_out([HIT], [HIT]), ref) == "unlabelled"


def test_any_mode_accepts_one_of_several_valid_clauses():
    ref = {"match_mode": "any", "expected_citations": [
        {"doc_id": "Doc A", "section": "Sec", "clause": "Cl"},
        {"doc_id": "Doc B", "section": "Sec", "clause": "Other"},
    ]}
    assert citation_status(_out([OTHER], [OTHER]), ref) == "cited"


def test_compare_lists_citations_gained_and_lost():
    def run(q, cites, results):
        return {"input": {"question": q}, "reference_output": REF,
                "output": _out(cites, results)}
    base = [run("gain", [OTHER], [HIT]), run("lose", [HIT], [HIT]), run("keep", [HIT], [HIT])]
    cand = [run("gain", [HIT], [HIT]), run("lose", [OTHER], [HIT]), run("keep", [HIT], [HIT])]

    c = compare(base, cand)["citations"]

    assert c["gained"] == [("gain", "selection", "cited")]
    assert c["lost"] == [("lose", "cited", "selection")]
    assert c["base"]["cited"] == 2 and c["cand"]["cited"] == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_compare_runs.py -o addopts="" -q -p no:warnings`
Expected: FAIL with `ImportError: cannot import name 'citation_status'`

- [ ] **Step 3: Implement** in `eval/compare_runs.py`

Imports (top of module): `from collections import Counter`, and extend the evaluators import to `from eval.evaluators import _extract_expected, _match_result, citation_clause_accuracy  # noqa: E402`.

After `first_match_rank`:

```python
def citation_status(output: dict | None, reference: dict) -> str:
    """What happened to the expected clause on its way into the answer.

    Built on citation_clause_accuracy, so this listing and the Phoenix column
    cannot disagree. 'selection' is #10's case: the clause was in the sources
    and the model cited another one.
    """
    if not output:
        return "retrieval"
    if (output.get("escalation") or {}).get("needed") and not output.get("citations"):
        return "escalated"
    scored = citation_clause_accuracy(output, reference)
    if scored["label"] == "skip":
        return "unlabelled"
    if scored["score"] == 1.0:
        return "cited"
    return "selection" if first_match_rank(output, reference) else "retrieval"
```

In `compare()`, initialise `cites = {"gained": [], "lost": [], "base": Counter(), "cand": Counter()}` beside `latency`. After the `errored` check (inside the loop, before computing `b`/`c`), add:

```python
        bs = citation_status(base[key]["output"], base[key]["reference_output"])
        cs = citation_status(run["output"], run["reference_output"])
        cites["base"][bs] += 1
        cites["cand"][cs] += 1
        if cs == "cited" and bs != "cited":
            cites["gained"].append((q, bs, cs))
        elif bs == "cited" and cs != "cited":
            cites["lost"].append((q, bs, cs))
```

and before `return out`: `out["citations"] = cites`.

In `main()`, after the latency block:

```python
    ct = c["citations"]
    print("\nCITATIONS  (status: cited / selection / retrieval / escalated / unlabelled)")
    for side in ("base", "cand"):
        counts = ", ".join(f"{k} {ct[side][k]}" for k in
                           ("cited", "selection", "retrieval", "escalated", "unlabelled"))
        print(f"  {'baseline ' if side == 'base' else 'candidate'}: {counts}")
    for key in ("gained", "lost"):
        print(f"\n  CITATION {key.upper()} ({len(ct[key])})")
        for q, b, n in ct[key]:
            print(f"    {b:>10} -> {n:<10} {q}")
```

Update the module docstring's second paragraph to say it also lists citations gained and lost, and per-status counts.

- [ ] **Step 4: Run the suite**

Run: `PYTHONPATH=. pytest tests -o addopts="" -q -p no:warnings`
Expected: all pass, including the existing rank tests. Their `_run` outputs have no `citations` or `escalation`, so they get status `retrieval` or `selection`, which those tests don't check.

- [ ] **Step 5: Smoke check against Phoenix** (read-only; it needs the network, so it is not a test)

Run: `PYTHONPATH=. python eval/compare_runs.py RXhwZXJpbWVudDoxOA== RXhwZXJpbWVudDoyNQ== --phoenix-url http://172.20.1.10:6006 2>&1 | grep -v "^\.env sets\|is set but does nothing" | tail -25`
Expected: baseline `cited 52`, candidate `cited 55`, candidate `selection 4`, `retrieval 2`, the same figures the design doc records for production-before-today vs the deployed config (counted then by an ad-hoc script with the same matching). If the counts differ, find out why before committing; a difference would mean `match_mode: all` on a multi-citation case.

- [ ] **Step 6: Commit**

```bash
git add eval/compare_runs.py tests/unit/test_compare_runs.py
git commit -m "feat(eval): compare_runs lists citation gains and losses, and selection vs retrieval misses"
```

### Task 3: Relabel four narrow cases as `chatbot-test-v2` (owner sign-off)

**Files:**
- Modify: `eval/datasets/chatbot_test_cases.json` (CB-014, CB-017, CB-032, CB-047; `metadata.version` → `"2.0"`, `metadata.last_updated` → `"2026-10-14"`)
- Create: `tests/unit/test_chatbot_labels.py`
- Create: `tests/docs/test_dataset_labels.py`

**Interfaces:**
- Produces: four cases with an added alternative and `"match_mode": "any"`. Task 4 reuses the corpus test's helper `_label_exists`.

The relabels: each answer was read in #10 and is correct, and the label lists only one of the clauses that answer it.

| id | question | keep | add |
|---|---|---|---|
| CB-014 | Teams chats private between colleagues? | AUP / Private Information / Email, Network, Internet, and Communication Monitoring Activities | AUP / Private Information / **Chat Messages** (7.3: chats solely between Team Members are personal) |
| CB-017 | free productivity app on work laptop? | AUP / Corporate Workstation and Software Use / Software Installation | AUP / Corporate Workstation and Software Use / **Unacceptable Use** (4.3: unapproved software); 4.7 is about *unlicensed* software, which "free" is not |
| CB-032 | disable the 5-minute auto-lock? | Clear Desk and Clear Screen Policy / Workplace Protection: Clear Screen / Computers | Access Management Policy / Access Management / **Inactivity Logoff/Lockout** (4.14: users are prohibited from disabling lock settings), the same pair CB-033 already accepts |
| CB-047 | personal laptop to access company files? | Mobile Device and Teleworking Policy / Mobile Device / Device Usage and Encryption | AUP / Corporate Workstation and Software Use / **Exceptions** (4.2), the label of the neighbouring CB-001 |

(AUP = `Acceptable Use Policy [Internal]`; other `doc_id`s are the full titles with `[Internal]`.)

- [ ] **Step 1: Write the failing unit test** `tests/unit/test_chatbot_labels.py`

```python
"""The relabelled cases accept every clause that answers them (#10, step 2).

Checked on the MAPPED output, not only the JSON file: match_mode silently
dropped on upload made "any" a no-op until 2026-09-25 (review focus 5).
"""

import json
from pathlib import Path

from scripts.make_dataset import map_chatbot

CASES = json.loads(Path("eval/datasets/chatbot_test_cases.json").read_text(encoding="utf-8"))
BY_ID = {c["id"]: c for c in CASES["test_cases"]}
RELABELLED = {
    "CB-014": "Chat Messages",
    "CB-017": "Unacceptable Use",
    "CB-032": "Inactivity Logoff/Lockout",
    "CB-047": "Exceptions",
}


def test_each_relabelled_case_accepts_its_alternative_under_any():
    _, outputs, meta = map_chatbot([BY_ID[i] for i in RELABELLED])
    for (case_id, added), out in zip(RELABELLED.items(), outputs):
        assert out["match_mode"] == "any", case_id
        assert added in [c["clause"] for c in out["expected_citations"]], case_id
        assert len(out["expected_citations"]) == 2, case_id


def test_the_set_is_still_61_unique_questions():
    ids = [c["id"] for c in CASES["test_cases"]]
    assert len(ids) == 61 and len(set(ids)) == 61
    assert CASES["metadata"]["version"] == "2.0"
```

- [ ] **Step 2: Write the corpus test** `tests/docs/test_dataset_labels.py`. It auto-skips without `policies/`.

```python
"""Every expected citation in the chatbot datasets names a clause that exists.

A label with a typo can never be cited, so it reads as a model failure forever.
Matching mirrors eval/evaluators._match_result: doc exact (case-insensitive),
section and clause as substrings, an empty field matching anything.
"""

import json
from pathlib import Path

import pytest

from ingest.docx_parser import parse_docx
from tests._corpus import policy_docx_files

_DOCX = policy_docx_files()
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not _DOCX, reason="policy corpus not present (gitignored/local-only)"),
]
DATASETS = ["eval/datasets/chatbot_test_cases.json"]


@pytest.fixture(scope="module")
def chunks():
    return [c for p in _DOCX for c in parse_docx(p, doc_link="")]


def _label_exists(label: dict, chunks) -> bool:
    doc, sec, cl = (label.get(k, "").strip().lower() for k in ("doc_id", "section", "clause"))
    return any(
        (not doc or doc == c.doc_title.strip().lower())
        and (not sec or sec in (c.section or "").lower())
        and (not cl or cl in (c.clause or "").lower())
        for c in chunks
    )


@pytest.mark.parametrize("path", DATASETS)
def test_every_expected_citation_names_a_real_clause(path, chunks):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["test_cases"]
    missing = [(c["id"], e) for c in cases for e in c["expected_citations"]
               if not _label_exists(e, chunks)]
    assert not missing, missing
```

- [ ] **Step 3: Run both to verify**

Run: `PYTHONPATH=. pytest tests/unit/test_chatbot_labels.py tests/docs/test_dataset_labels.py -o addopts="" -q -p no:warnings`
Expected: the unit tests FAIL (no `match_mode` on CB-014, version `1.0`). The corpus test PASSES on the current labels. If it fails, it has found an existing broken label: record it in the ledger and show it to the user alongside the relabel. Don't fix it silently.

- [ ] **Step 4: Edit the four cases** in `eval/datasets/chatbot_test_cases.json`. Append the "add" row's citation to `expected_citations`, add `"match_mode": "any"` at the top level of the case (CLAUDE.md gotcha "Multi-citation eval flags valid answers"), and bump `metadata.version` and `metadata.last_updated`. Example, CB-032:

```json
    {
      "id": "CB-032",
      "question": "Can I disable the 5-minute auto-lock on my computer?",
      "expected_answer": "<unchanged>",
      "expected_citations": [
        {
          "doc_id": "Clear Desk and Clear Screen Policy [Internal]",
          "section": "Workplace Protection: Clear Screen",
          "clause": "Computers"
        },
        {
          "doc_id": "Access Management Policy [Internal]",
          "section": "Access Management",
          "clause": "Inactivity Logoff/Lockout"
        }
      ],
      "match_mode": "any"
    },
```

(`<unchanged>` means: leave the existing `expected_answer` string as it is.)

- [ ] **Step 5: Run to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_chatbot_labels.py tests/docs/test_dataset_labels.py -o addopts="" -q -p no:warnings`
Expected: PASS

- [ ] **Step 6: STOP — owner sign-off.** Show the user `git diff eval/datasets/chatbot_test_cases.json` and the table above, and wait for an explicit yes. The dataset is ground truth: an alternative the owner disagrees with must not reach a commit. A "no" on any row removes that row, along with its entry in `RELABELLED`, before continuing.

- [ ] **Step 7: Run the suite and commit**

Run: `PYTHONPATH=. pytest tests -o addopts="" -q -p no:warnings` → all pass

```bash
git add eval/datasets/chatbot_test_cases.json tests/unit/test_chatbot_labels.py tests/docs/test_dataset_labels.py
git commit -m "data: chatbot-test-v2 — four cases accept every clause that answers them (#10)"
```

### Task 4: `role-mismatch-v1`, six questions (owner sign-off)

**Files:**
- Create: `eval/datasets/role_mismatch_cases.json`
- Modify: `tests/docs/test_dataset_labels.py` (add the file to `DATASETS`)
- Create: `tests/unit/test_role_mismatch_dataset.py`

**Interfaces:**
- Consumes: `_label_exists` and `DATASETS` (Task 3), and `detect_tier` and `map_chatbot` from `scripts/make_dataset.py`.
- Produces: a chatbot-tier dataset file with ids `RM-001`…`RM-006`, uploaded in Task 5 as `role-mismatch-v1`.

The defect (#10): the model cites the clause whose wording matches the question but whose addressee is another role (DPO, controller, Head of IT), when the asker is a Team Member and a clause telling *them* what to do is retrieved beside it. Four questions set that trap. Two are **controls**, where the right answer *is* a clause addressed to another role, so a later prompt rule that over-corrects ("always prefer the Team Member clause") shows up as a loss.

| id | question | expected (doc / section / clause) | trap |
|---|---|---|---|
| RM-001 | I accidentally exposed some customers' personal data. Do I need to notify the affected customers myself? | Data Breach Response and Notification Procedure [Internal] / Data Breach Response Management / Initial Reporting | 6.4 Controller to Data Subject (DPO notifies) |
| RM-002 | Personal data we process for a client was leaked from our system. Should I email the client to let them know? | same as RM-001 | 6.2 Processor to Controller (DPO, 48 h) |
| RM-003 | I lost a laptop with personal data on it. Do we have to tell the data protection authority within 72 hours? | same as RM-001 | 6.3 Controller to Supervisory Authority (DPO) |
| RM-004 | I think someone stole my account password. What should I do? | Password Policy [Internal] / Password Protection Standards / Reporting | 7.5 Safeguarding (Head of IT controls) |
| RM-005 *(control)* | Can my team lead approve an exception to the information security policy for our project? | Information Security And Privacy Policy [Internal] / *(any section)* / Handling Exemptions and Exceptions | none: 7.3 says the Head of IT approves, which is the answer |
| RM-006 *(control)* | Who should I alert about a security incident in the middle of the night? | Incident Response Policy [Internal] / Security Incident Management / Contact Person | none: 6.1 names CTO / Head of IT 24/7 and SOC email |

Expected answers (one sentence each, quoting the policy's duty):
- RM-001–003: "Report it immediately to the information security team at the SOC email; the Data Protection Officer decides on and makes any notification to data subjects, clients or the supervisory authority."
- RM-004: "Report it to the Head of IT (IT Director) and change all your passwords immediately."
- RM-005: "No — exemptions and exceptions to the policy must be approved by the Head of IT (IT Director)."
- RM-006: "Inform the IT Security Officer via the SOC email; the CTO or Head of IT is designated to be alerted on incidents 24/7."

- [ ] **Step 1: Write the failing test** `tests/unit/test_role_mismatch_dataset.py`

```python
"""role-mismatch-v1: four traps and two controls (#10, step 3).

One failing case cannot show a fix worked; the controls catch a prompt rule
that over-corrects into never citing a clause addressed to another role.
"""

import json
from pathlib import Path

from scripts.make_dataset import detect_tier, map_chatbot

DATA = json.loads(Path("eval/datasets/role_mismatch_cases.json").read_text(encoding="utf-8"))
CASES = DATA["test_cases"]


def test_it_uploads_as_a_chatbot_dataset():
    assert detect_tier(CASES) == "chatbot"
    inputs, outputs, _ = map_chatbot(CASES)
    assert len(inputs) == 6 and all(o["expected_citations"] for o in outputs)


def test_ids_are_unique_and_the_controls_are_marked():
    assert [c["id"] for c in CASES] == [f"RM-00{i}" for i in range(1, 7)]
    assert {c["id"] for c in CASES if c.get("control")} == {"RM-005", "RM-006"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_role_mismatch_dataset.py -o addopts="" -q -p no:warnings`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Create** `eval/datasets/role_mismatch_cases.json` from the table. Each case gets `id`, `question`, `expected_answer`, `expected_citations` (one entry; RM-005's `section` is `""`), and `"control": true` on RM-005 and RM-006. Use a `metadata` block like the chatbot set's: `"name": "Role-mismatch test set"`, `"version": "1.0"`, `"last_updated": "2026-10-14"`, `"total_cases": 6`, and a `"description"` naming the defect. `map_chatbot` ignores the extra `control` key. Add `"eval/datasets/role_mismatch_cases.json"` to `DATASETS` in `tests/docs/test_dataset_labels.py`.

- [ ] **Step 4: Run to verify**

Run: `PYTHONPATH=. pytest tests/unit/test_role_mismatch_dataset.py tests/docs/test_dataset_labels.py -o addopts="" -q -p no:warnings`
Expected: PASS. A corpus failure means a label names no real clause: fix the label from the parsed chunk text, never by loosening the test.

- [ ] **Step 5: STOP — owner sign-off.** Show the user the six questions, their expected clauses and the expected answers, and wait for an explicit yes. Rewording a question is fine; changing which clause is expected is the owner's call.

- [ ] **Step 6: Run the suite and commit**

Run: `PYTHONPATH=. pytest tests -o addopts="" -q -p no:warnings` → all pass

```bash
git add eval/datasets/role_mismatch_cases.json tests/unit/test_role_mismatch_dataset.py tests/docs/test_dataset_labels.py
git commit -m "data: role-mismatch-v1 — four traps, two controls (#10)"
```

### Task 5: Upload, run the new baseline, record it

**Files:**
- Modify: `docs/superpowers/specs/2026-10-06-one-candidate-knob-results.md` (item 10: steps 1-3 done, plus the baseline)
- Modify: `CLAUDE.md` (Eval tiers line: the citation evaluators are on; the two new datasets)
- Modify: `SETUP.md` (Step 11: the upload commands and `compare_runs`' CITATIONS block)

This task is run with the user. The user pushes the branch and runs each VM command; Claude reads Phoenix and writes the numbers. No env overrides are needed: the deployed `.env` already holds the configuration being baselined (`RERANKER_DOC_HEADER=true`, `QUERY_REWRITE=multi`, `use_header`).

- [ ] **Step 1: User pushes** `feat/citation-measurement`; on the VM, `cd ~/rag-eval && git fetch && git checkout feat/citation-measurement && git pull && git log --oneline -1`.

- [ ] **Step 2: Upload `chatbot-test-v2`** (VM, from `~/rag`, `$EVAL` as in SETUP.md Step 11):

```bash
docker compose -f docker-compose-remote.yml run --rm $EVAL --entrypoint python bot \
  scripts/make_dataset.py eval/datasets/chatbot_test_cases.json --name chatbot-test-v2 --phoenix-url http://phoenix:6006
```

Expected: `Dataset created: chatbot-test-v2` with 61 examples.

- [ ] **Step 3: Upload `role-mismatch-v1`:**

```bash
docker compose -f docker-compose-remote.yml run --rm $EVAL --entrypoint python bot \
  scripts/make_dataset.py eval/datasets/role_mismatch_cases.json --name role-mismatch-v1 --phoenix-url http://phoenix:6006
```

Expected: `Dataset created: role-mismatch-v1` with 6 examples.

- [ ] **Step 4: Baseline on v2** (chatbot tier):

```bash
docker compose -f docker-compose-remote.yml run --rm $EVAL \
  -e GIT_COMMIT=$(git -C /home/sa.ivanov/rag-eval describe --always --dirty) \
  --entrypoint python bot eval/run_experiment.py \
  --tier chatbot --dataset chatbot-test-v2 --name citations-baseline-v2 --phoenix-url http://phoenix:6006
```

Expected banner: `Rerank prompt: 69990c1bedf7`, `Rewrite: multi`, and `Evaluators:` listing the three `citation_*`.

- [ ] **Step 5: Baseline on role-mismatch:** the same command with `--dataset role-mismatch-v1 --name citations-baseline-rm`.

- [ ] **Step 6: Read the runs.**
  - Check that each experiment's metadata carries `reranker_prompt_sha12: 69990c1bedf7`, `query_rewrite: multi` and `reranker_doc_header: true`.
  - Pull the `citation_*` means from Phoenix.
  - Run `compare_runs.py RXhwZXJpbWVudDoyNQ== <v2 id>`. Its CITATIONS block should show CB-014 and CB-032 gained by relabel, with no answer change expected (the answers are the same configuration's). Anything listed as lost is answer drift, not a label effect: read it.
  - On role-mismatch, list each RM case's status, and read RM-001–004's citations against their traps.

- [ ] **Step 7: Record and commit.**
  - **`2026-10-06-one-candidate-knob-results.md` item 10:**
    - Steps 1-3 are marked done (2026-10-14).
    - Add a baseline table: experiment ids, the three `citation_*` means on v2, the status counts (cited / selection / retrieval / escalated), and the RM per-case statuses.
    - Add one line: v1 and v2 numbers are not comparable, because v2 accepts more answers.
    - Step 4 (the prompt rule) is planned next against these two baselines.
  - **`CLAUDE.md`:** in the **Eval tiers** paragraph, change `chatbot \`chatbot-test-v1\` (realistic)` to `chatbot \`chatbot-test-v2\` (realistic; v1 kept for history) + \`role-mismatch-v1\` (#10)`, and change the evaluators note to say the `citation_*` evaluators score the chatbot and tier2 runs. No new line; `wc -l CLAUDE.md` ≤ 150.
  - **`SETUP.md` Step 11:** add the two upload commands, and one sentence explaining `compare_runs`' CITATIONS block (selection = the clause was in the sources and the model cited another).

```bash
git add docs/superpowers/specs/2026-10-06-one-candidate-knob-results.md CLAUDE.md SETUP.md
git commit -m "docs: citation baseline on chatbot-test-v2 and role-mismatch-v1 (#10 steps 1-3)"
```
