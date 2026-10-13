# Reranker identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every experiment and every production `compliance_request` span records which reranker prompt (instruction + backend wrapper) and whether the doc header was used, so a run or trace can be matched by string comparison like the other `identity.*` values.

**Architecture:** One function `reranker_identity()` in `rag/run_identity.py` hashes the reranker query exactly as `rag/reranker.py` builds it (so a wording change, a backend change, or a `RERANKER_QUERY_TEMPLATE` change on llama-server all change the hash). Eval merges it into `infra_meta` for every tier and records the instruction text verbatim; the bot adds it plus the header flag to `_identity_attributes()`.

**Tech Stack:** Python 3, pytest, OpenTelemetry span attributes, Phoenix experiment metadata.

**Spec:** `docs/superpowers/specs/2026-10-09-query-rephrasing-design.md`, section "`RERANKER_INSTRUCTION` (one change against header + `multi`)" — the instruction changed results (MRR 0.782 → 0.802, cited 54 → 55/61) and was recorded nowhere but the run name. Pattern to follow: `rewrite_identity()` (same file) and its tests in `tests/unit/test_run_identity.py`.

## Global Constraints

- Identity values on the span are strings (`_identity_attributes() -> dict[str, str]`).
- Reranker off → `reranker_prompt_sha12 == ""` (as `rewrite_prompt_sha12` is with rewrite off), so "not used" never looks like a hash.
- No network calls added to the request path; hashing is microseconds.
- `eval/run_experiment.py` keeps importing `rag.*` only inside `main()` (`test_prompt_meta_does_not_drag_llamaindex_into_module_import`).
- `channels/teams/bot.py` keeps its deferred `rag.run_identity` import (observability first).
- No new dependencies; tests never touch the network.
- `CLAUDE.md` stays ≤150 lines.

## Review Focus

1. `RERANKER_INSTRUCTION` changed only in `.env` → hash changes (the case that motivated this).
2. Backend changes vllm-score → llama-server with the same instruction → hash changes (the wrapper is part of what the model reads).
3. Reranker disabled → empty hash on both eval metadata and span, not a hash of an unused prompt.
4. The `\n` handling in `_build_query` (llama-server converts `\\n`) is hashed after conversion, i.e. what is sent, not what is configured.
5. Eval metadata key collision: `reranker_doc_header` already comes from `_reranker_meta()` as a bool; the identity must not overwrite it with a string.

---

### Task 1: `reranker_identity()`

**Files:**
- Modify: `rag/run_identity.py` (import `rag.reranker as reranker`; new function after `rewrite_identity`)
- Test: `tests/unit/test_run_identity.py` (new section after "query rewrite")

**Interfaces:**
- Produces: `reranker_identity() -> dict[str, str]` returning `{"reranker_prompt_sha12": str}` — `sha12(reranker._build_query(""))` when `settings.reranker_enabled`, else `""`.

- [ ] **Step 1: Write the failing tests** (add `import rag.reranker as rr` to the module's imports)

```python
# --- reranker ----------------------------------------------------------------------


def test_reranker_off_has_no_prompt_hash(monkeypatch):
    monkeypatch.setattr(ri.settings, "reranker_enabled", False)
    assert ri.reranker_identity() == {"reranker_prompt_sha12": ""}


def test_the_reranker_hash_is_of_the_query_as_sent(monkeypatch):
    monkeypatch.setattr(ri.settings, "reranker_enabled", True)
    monkeypatch.setattr(ri.settings, "reranker_backend", "vllm-score")
    assert ri.reranker_identity() == {"reranker_prompt_sha12": ri.sha12(rr._build_query(""))}


def test_editing_the_instruction_changes_the_hash(monkeypatch):
    """The 2026-10-13 instruction change moved MRR 0.782 -> 0.802 and was
    recorded only in a run name."""
    monkeypatch.setattr(ri.settings, "reranker_enabled", True)
    monkeypatch.setattr(ri.settings, "reranker_backend", "vllm-score")
    before = ri.reranker_identity()["reranker_prompt_sha12"]
    monkeypatch.setattr(ri.settings, "reranker_instruction", ri.settings.reranker_instruction + " ")
    assert ri.reranker_identity()["reranker_prompt_sha12"] != before


def test_changing_the_backend_changes_the_hash(monkeypatch):
    """Same instruction, different wrapper: the model reads a different text."""
    monkeypatch.setattr(ri.settings, "reranker_enabled", True)
    monkeypatch.setattr(ri.settings, "reranker_backend", "vllm-score")
    vllm = ri.reranker_identity()["reranker_prompt_sha12"]
    monkeypatch.setattr(ri.settings, "reranker_backend", "llama-server")
    assert ri.reranker_identity()["reranker_prompt_sha12"] != vllm
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_run_identity.py -o addopts="" -q -p no:warnings -k reranker`
Expected: FAIL — `AttributeError: module 'rag.run_identity' has no attribute 'reranker_identity'`

- [ ] **Step 3: Implement** in `rag/run_identity.py`

```python
def reranker_identity() -> dict[str, str]:
    """The reranker prompt as sent: instruction plus the backend's wrapper.

    Hashed from rag/reranker.py's own builder, so an instruction edit in .env, a
    backend switch, or a llama-server template change all show up -- each
    changes what the model reads. Empty with the reranker off, as
    rewrite_prompt_sha12 is with rewrite off.
    """
    if not settings.reranker_enabled:
        return {"reranker_prompt_sha12": ""}
    return {"reranker_prompt_sha12": sha12(reranker._build_query(""))}
```

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_run_identity.py -o addopts="" -q -p no:warnings`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add rag/run_identity.py tests/unit/test_run_identity.py
git commit -m "feat(identity): hash the reranker prompt as sent"
```

### Task 2: Experiments record it

**Files:**
- Modify: `eval/run_experiment.py` (`_reranker_meta()`, `infra_meta` in `main()`, banner)
- Test: `tests/unit/test_run_identity.py`, `tests/unit/test_eval_metadata.py`

**Interfaces:**
- Consumes: `reranker_identity()` (Task 1)
- Produces: metadata keys `reranker_prompt_sha12` (every tier, via `infra_meta`) and `reranker_instruction` (verbatim text, `None` with the reranker off, via `_reranker_meta()`)

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_run_identity.py`, after `test_experiments_record_the_rewrite_identity_on_every_tier`:

```python
def test_experiments_record_the_reranker_identity_on_every_tier():
    src = inspect.getsource(rx.main)
    assert "**reranker_identity()" in src
    assert src.index("**reranker_identity()") < src.index('if args.tier == "tier1"')
```

In `tests/unit/test_eval_metadata.py`, in the "reranker parameters" section:

```python
def test_the_instruction_text_is_recorded_verbatim(monkeypatch):
    """A hash says THAT it changed; the text says to what, without git archaeology
    -- the instruction lives in .env, not in the repo."""
    from eval.run_experiment import _reranker_meta

    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "reranker_instruction", "Judge clauses; use the header")
    assert _reranker_meta()["reranker_instruction"] == "Judge clauses; use the header"
    monkeypatch.setattr(settings, "reranker_enabled", False)
    assert _reranker_meta()["reranker_instruction"] is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_run_identity.py tests/unit/test_eval_metadata.py -o addopts="" -q -p no:warnings`
Expected: 2 FAIL — `"**reranker_identity()" in src` is False; `KeyError: 'reranker_instruction'`

- [ ] **Step 3: Implement** in `eval/run_experiment.py`

`_reranker_meta()` — add beside `reranker_doc_header`:

```python
        "reranker_instruction": settings.reranker_instruction if on else None,
```

`main()` — change `from rag.run_identity import rewrite_identity` to `from rag.run_identity import reranker_identity, rewrite_identity`, and in `infra_meta` after `**rewrite_identity(),`:

```python
        # The reranker prompt as sent (instruction + backend wrapper). The
        # instruction lives in .env, so without this two runs that differ only
        # there were told apart by their names alone (2026-10-13).
        **reranker_identity(),
```

Banner — after the `Rewrite:` line:

```python
    print(f"  Rerank prompt: {reranker_identity()['reranker_prompt_sha12'] or 'n/a'}")
```

(The banner line comes after the deferred import; move the import above the banner prints if needed so the name is bound — `tests/unit/test_no_undefined_names.py` will catch it if not.)

- [ ] **Step 4: Run the suite**

Run: `PYTHONPATH=. pytest tests -o addopts="" -q -p no:warnings`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add eval/run_experiment.py tests/unit/test_run_identity.py tests/unit/test_eval_metadata.py
git commit -m "feat(eval): record the reranker prompt hash and instruction text"
```

### Task 3: Production spans carry it

**Files:**
- Modify: `channels/teams/bot.py` (`_identity_attributes`)
- Test: `tests/unit/test_run_identity.py`

**Interfaces:**
- Consumes: `reranker_identity()` (Task 1)
- Produces: span attributes `identity.reranker_prompt_sha12` (str) and `identity.reranker_doc_header` (`"true"` / `"false"`, `""` with the reranker off)

- [ ] **Step 1: Write the failing test**

```python
def test_production_spans_carry_the_reranker_identity(monkeypatch):
    monkeypatch.setattr(ri.settings, "reranker_enabled", True)
    monkeypatch.setattr(ri.settings, "reranker_doc_header", True)
    b = object.__new__(bot.TeamsBot)
    b._llm_digest_at_startup = ""
    attrs = b._identity_attributes()
    assert attrs["identity.reranker_prompt_sha12"] == ri.reranker_identity()["reranker_prompt_sha12"]
    assert attrs["identity.reranker_doc_header"] == "true"


def test_the_header_flag_is_empty_with_the_reranker_off(monkeypatch):
    monkeypatch.setattr(ri.settings, "reranker_enabled", False)
    b = object.__new__(bot.TeamsBot)
    b._llm_digest_at_startup = ""
    attrs = b._identity_attributes()
    assert attrs["identity.reranker_prompt_sha12"] == ""
    assert attrs["identity.reranker_doc_header"] == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_run_identity.py -o addopts="" -q -p no:warnings -k "reranker_identity or header_flag"`
Expected: FAIL with `KeyError: 'identity.reranker_prompt_sha12'`

- [ ] **Step 3: Implement** in `_identity_attributes`

Extend the deferred import with `reranker_identity`, and add to the returned dict:

```python
            **{f"identity.{k}": v for k, v in reranker_identity().items()},
            "identity.reranker_doc_header": (
                str(settings.reranker_doc_header).lower() if settings.reranker_enabled else ""
            ),
```

- [ ] **Step 4: Run the suite**

Run: `PYTHONPATH=. pytest tests -o addopts="" -q -p no:warnings`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add channels/teams/bot.py tests/unit/test_run_identity.py
git commit -m "feat(bot): stamp the reranker prompt hash and header flag on every request"
```

### Task 4: Docs

**Files:**
- Modify: `CLAUDE.md` (Eval tiers paragraph), `SETUP.md` (reranker-header comparison block)

- [ ] **Step 1: CLAUDE.md** — in the **Eval tiers** paragraph, extend the identity list `` `git_commit`, LLM digest, `system_prompt_sha12`, `agent_input_sha12` (the question+sources layout) `` with `` , `reranker_prompt_sha12` (instruction + backend wrapper as sent) ``. No new line; check `wc -l CLAUDE.md` ≤ 150.

- [ ] **Step 2: SETUP.md** — at the end of the "Reranker-header comparison" block, add one paragraph: `RERANKER_INSTRUCTION` is set in `.env`, so experiments record it verbatim (`reranker_instruction`) and by hash (`reranker_prompt_sha12`); production spans carry `identity.reranker_prompt_sha12` and `identity.reranker_doc_header`. Quote any instruction containing `;` in `-e "RERANKER_INSTRUCTION=…"`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md SETUP.md
git commit -m "docs: reranker identity in eval metadata and spans"
```
