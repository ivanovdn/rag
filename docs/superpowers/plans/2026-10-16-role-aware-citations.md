# Role-aware citations (#10, step 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix #10's one real defect: the answering model cites a clause addressed to another role (DPO, controller, Head of IT) instead of the Team Member's own duty. The fix is one rule in `SYSTEM_PROMPT`, measured against the step 1-3 baselines, and shipped only if the stop rule passes.

**Architecture:** One added line in `== RULES ==` of `rag/agent.py::SYSTEM_PROMPT`; nothing else in the bot changes. The prompt's token overhead is measured, not derived (`FIXED_OVERHEAD_TOKENS`, pinned by `tests/unit/test_llm_config.py`), so it is re-measured live and both constants move together. The candidate is measured on the VM by mounting this branch's `rag/` over the image (SETUP.md Step 11). It is compared with `compare_runs` against the baselines: `RXhwZXJpbWVudDoyOA==` (`chatbot-test-v2`) and `RXhwZXJpbWVudDoyOQ==` (`role-mismatch-v1`).

**Tech Stack:** Python 3.12, llama-index `Ollama` (qwen3.6 @ `07d35212591f`), Phoenix experiments, `eval/compare_runs.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-one-candidate-knob-results.md`, item **10**, step 4: *"Only then try one prompt rule ('the asker is a Team Member; prefer the clause that says what they must do; if you cite one addressed to another role, say so') and compare against [the baseline] by its identity metadata."* Plus the 2026-10-16 baseline block: the model took the trap in all three breach questions (RM-001..003). CB-034 flipped from *Initial Reporting* to *Removable Storage Devices* when only sources 5 and 6 swapped. 58/61 answers repeat word for word between same-config runs.

## Global Constraints

- One change: the rule line, plus the two constants that measure it. No other prompt edit, no retrieval setting, no `.env` change.
- `temperature=0.0`, `num_ctx` 4096, `num_predict` 1024. `FIXED_OVERHEAD_TOKENS + MAX_SOURCE_TOKENS + 1024 <= 4096` must still hold.
- Citations come only from retrieved chunks. The rule changes which retrieved clause is cited, never whether citing is required.
- The live token measurement is one inference call to the shared Ollama at `172.20.0.22:11434`: no server config change, `keep_alive` passed, `num_ctx` 4096.
- Tests never touch the network.
- VM commands are given one at a time; the user runs each one.
- Shipping (merge, deploy) is the user's decision after the stop rule. A failed stop rule leaves the branch unmerged, with the result recorded.
- `CLAUDE.md` stays ≤150 lines. Imports at the top of the module.

## Review Focus

1. **Over-correction on the controls.** RM-005 (*can my team lead approve an exception?*) is answered by a clause addressed to the Head of IT. Expected: it is still cited; the rule's "who decides or approves" clause exists for this.
2. **"Must be told" questions.** RM-002 (*should I email the client?*) and RM-006 (*who should I alert at night?*). Expected: RM-002 moves to *Initial Reporting*; RM-006 still cites *Contact Person*, whose text already holds the Team Member's duty ("the person who discovers an incident must inform…").
3. **Escalation creep.** A rule about choosing between clauses could tip uncertain answers into "escalate instead of guessing". Expected: escalated stays 0 on v2.
4. **Answers that cite the Team Member duty but drop the other role's clause the user needed** (e.g. *the DPO decides whether to notify*). Expected: the rule allows citing it "next to the Team Member's own duty", and every changed answer is read.
5. **Noise read as signal.** Same-config reruns differ on ~1/61. Expected: the decision rests on the role-mismatch set, run twice, and on reading every v2 citation lost, never on a net count alone.

---

### Task 1: RM-003's label (owner decision, before measuring)

**Files:**
- Modify (local, gitignored): `eval/datasets/role_mismatch_cases.json` (RM-003 only, if approved)

**Interfaces:**
- Produces: the role-mismatch dataset name and baseline id that Task 3 compares against. Either `role-mismatch-v1` / `RXhwZXJpbWVudDoyOQ==` (unchanged), or `role-mismatch-v2` / a new baseline id.

The 2026-10-16 baseline found RM-003 (*lost a laptop with personal data — tell the authority within 72 h?*) labelled only with *Initial Reporting*. *Initial Reporting* was not retrieved, but Mobile Device 5.5 *Handling and Reporting* was, at rank 3: "Prompt reporting of any lost, stolen, or damaged mobile device is required, within 12 hours". That is the Team Member's duty, and exactly what this rule should make the model cite. Under the v1 label a correct fix to RM-003 would score as a miss.

- [ ] **Step 1: STOP — ask the user** whether RM-003 should also accept `Mobile Device and Teleworking Policy [Internal]` / `Mobile Device` / `Handling and Reporting`, with `"match_mode": "any"`.
  - **No:** ledger it; Task 3 uses `role-mismatch-v1` and `RXhwZXJpbWVudDoyOQ==`; skip to Task 2.
  - **Yes:** do steps 2-4.
- [ ] **Step 2: Edit RM-003** (append the alternative, add `match_mode`, set `metadata.version` to `"2.0"`). Then run: `PYTHONPATH=. pytest tests/docs/test_dataset_labels.py tests/unit/test_role_mismatch_dataset.py -o addopts="" -q -p no:warnings`
  Expected: PASS. The guards require the clause to name one clause and both labels to be clause-level, which they are.
- [ ] **Step 3: Upload** from the laptop: `PYTHONPATH=. python scripts/make_dataset.py eval/datasets/role_mismatch_cases.json --name role-mismatch-v2 --phoenix-url http://172.20.1.10:6006`
  Expected: `Dataset created: role-mismatch-v2`, 6 examples. Verify RM-003's mapped output carries `match_mode: any` via `GET /v1/datasets/<id>/examples`.
- [ ] **Step 4: Re-baseline role-mismatch on v2** (VM, `main`'s code, so `~/rag-eval` checked out at `main` with `$EVAL` mounted):

```bash
docker compose -f docker-compose-remote.yml run --rm $EVAL \
  -e GIT_COMMIT=$(git -C /home/sa.ivanov/rag-eval describe --always --dirty) \
  --entrypoint python bot eval/run_experiment.py \
  --tier chatbot --dataset role-mismatch-v2 --name citations-baseline-rm-v2 --phoenix-url http://phoenix:6006
```

  Expected: 6/6 rows, `Rerank prompt: 69990c1bedf7`. Record its id in the ledger as the role-mismatch baseline. No commit: dataset files are local, and nothing tracked changed.

### Task 2: The rule, with its overhead re-measured

**Files:**
- Modify: `rag/agent.py` (`SYSTEM_PROMPT`, `FIXED_OVERHEAD_TOKENS` and the measurement comment above it)
- Modify: `tests/unit/test_llm_config.py` (the char-count guard)
- Test: `tests/unit/test_llm_config.py`

**Interfaces:**
- Produces: `SYSTEM_PROMPT` of exactly **2688** characters (currently 2350) and a new `system_prompt_sha12`, which identifies the candidate in Task 3's metadata and in production traces after shipping.

The rule (one line, inserted directly after `- If you are uncertain whether a source applies, escalate instead of guessing.`):

```
- The user is a Team Member. When the sources give different roles different duties, cite the clause that says what a Team Member must do. A duty of another role (the DPO, Head of IT, controller or processor) is not the user's: cite it only next to the Team Member's own duty, or when the question asks who decides or approves something.
```

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_llm_config.py`)

```python
def test_the_prompt_tells_the_model_whose_duty_to_cite():
    """#10 step 4: the model cited the DPO's notification duties (Breach 6.2-6.4)
    for an employee asking what to do, with the employee's own duty (5.1 Initial
    Reporting) retrieved beside them. The rule must keep the exception for
    who-decides questions, or the controls (RM-005) break."""
    from rag.agent import SYSTEM_PROMPT

    assert "The user is a Team Member." in SYSTEM_PROMPT
    assert "cite the clause that says what a Team Member must do" in SYSTEM_PROMPT
    assert "when the question asks who decides or approves something" in SYSTEM_PROMPT
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_llm_config.py -o addopts="" -q -p no:warnings -k whose_duty`
Expected: FAIL on the first assert

- [ ] **Step 3: Insert the rule** into `SYSTEM_PROMPT` at the position above. Run the same test: PASS. Then run `PYTHONPATH=. pytest tests/unit/test_llm_config.py -o addopts="" -q -p no:warnings`
Expected: `test_the_request_budget_fits_in_the_context_window` FAILS with "SYSTEM_PROMPT changed; re-measure…". That is the guard doing its job.

- [ ] **Step 4: Measure the overhead live**, the way the 571 was measured: `/api/chat` with the system prompt alone, `num_predict` 1, read `prompt_eval_count`. **Calibrate first on the current prompt from `main`** (it must read 571 ± 2, or the method differs from the original and the new number is not comparable). Run from the repo root:

```bash
git show main:rag/agent.py > "${TMPDIR:-/tmp}/agent_main.py"
for F in "${TMPDIR:-/tmp}/agent_main.py" rag/agent.py; do
  python3 - "$F" <<'EOF'
import json, sys, urllib.request
src = open(sys.argv[1]).read()
key = 'SYSTEM_PROMPT = """' + chr(92) + chr(10)
start = src.index(key) + len(key)
prompt = src[start:src.index('"""', start)]
body = {"model": "qwen3.6:latest", "stream": False, "think": False, "keep_alive": "30m",
        "messages": [{"role": "system", "content": prompt}],
        "options": {"num_ctx": 4096, "num_predict": 1, "temperature": 0.0}}
req = urllib.request.Request("http://172.20.0.22:11434/api/chat", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=120))
print(sys.argv[1], "chars", len(prompt), "prompt_eval_count", r["prompt_eval_count"])
EOF
done
```

The first file is `main`'s prompt (the calibration); the second is the working tree with Step 3's rule in it.
Expected: `main chars 2350 prompt_eval_count 571` (± 2), then the candidate at `chars 2688` and a count roughly 640-660. If calibration is off by more than 2, stop and ledger it: the measuring method differs from the original.

- [ ] **Step 5: Update both numbers together.**
  - In `rag/agent.py`, set `FIXED_OVERHEAD_TOKENS` to the measured value and update the comment's `this prompt` / `fixed overhead` lines: `<n> tokens (the previous one: 571)`.
  - In `tests/unit/test_llm_config.py`, change `assert len(SYSTEM_PROMPT) == 2350` to `== 2688`.
  - Run `PYTHONPATH=. pytest tests -o addopts="" -q -p no:warnings`. Expected: all pass. The budget is `<n> + 1300 + 1024 <= 4096` with ~1100 to spare.

- [ ] **Step 6: Commit**

```bash
git add rag/agent.py tests/unit/test_llm_config.py
git commit -m "feat(prompt): cite the Team Member's own duty over another role's (#10 step 4)"
```

### Task 3: Measure the candidate on the VM, apply the stop rule

**Files:**
- Modify: `docs/superpowers/specs/2026-10-06-one-candidate-knob-results.md` (item 10: step 4 result)

**Interfaces:**
- Consumes: the role-mismatch baseline from Task 1, `RXhwZXJpbWVudDoyOA==` (v2 baseline), and Task 2's commit.

**Stop rule:** ship only if ALL of these hold.
- **Role-mismatch, in BOTH candidate runs:** at least 2 of the 3 breach traps (RM-001..003) cited (baseline 0/3); RM-004 still cited; both controls (RM-005, RM-006) still cited.
- **`chatbot-test-v2`:** cited ≥ 57 (baseline 57). Every question under CITATION LOST is read, and none reads worse in substance. CB-034 is read either way: it is the fragile case.
- **Health:** `escalated` stays 0 and `json_parse_success` stays 1.0 on v2. Median latency within +0.5 s of 6.7 s.

- [ ] **Step 1: User pushes** `feat/role-aware-citations`; on the VM: `cd ~/rag-eval && git fetch && git checkout feat/role-aware-citations && git pull && git log --oneline -1`. Expected: Task 2's commit.

- [ ] **Step 2: Assert the candidate prompt is the one mounted**, before spending GPU (VM, from `~/rag`, `$EVAL` set):

```bash
docker compose -f docker-compose-remote.yml run --rm $EVAL --entrypoint python bot \
  -c "from rag.agent import SYSTEM_PROMPT; from rag.run_identity import prompt_identity; print(len(SYSTEM_PROMPT), prompt_identity())"
```

Expected: `2688` and a `system_prompt_sha12` that is not `42d2fec870f6`. Record it in the ledger.

- [ ] **Step 3: Role-mismatch, first run** (the dataset from Task 1):

```bash
docker compose -f docker-compose-remote.yml run --rm $EVAL \
  -e GIT_COMMIT=$(git -C /home/sa.ivanov/rag-eval describe --always --dirty) \
  --entrypoint python bot eval/run_experiment.py \
  --tier chatbot --dataset <role-mismatch dataset> --name role-rule-rm-1 --phoenix-url http://phoenix:6006
```

- [ ] **Step 4: Role-mismatch, second run**: the same command with `--name role-rule-rm-2`.

- [ ] **Step 5: `chatbot-test-v2`**: the same command with `--dataset chatbot-test-v2 --name role-rule-v2`.

- [ ] **Step 6: Read the runs** (laptop):
  - Check each experiment's metadata: the Step 2 `system_prompt_sha12`, plus `reranker_prompt_sha12: 69990c1bedf7` and `query_rewrite: multi`, so that only the prompt differs from the baselines.
  - Run `PYTHONPATH=. python eval/compare_runs.py <baseline> <candidate> --phoenix-url http://172.20.1.10:6006` for rm-1, rm-2 (against the role-mismatch baseline) and v2 (against `RXhwZXJpbWVudDoyOA==`).
  - For every RM case and every v2 CITATION GAINED or LOST, read the answer and its citations side by side with the baseline's: the `chat_summary`/`answers` pattern from `GET /v1/experiments/<id>/json`.
  - Apply the stop rule line by line.

- [ ] **Step 7: Record and commit.** In item 10, add a "Step 4" block:
  - the rule text;
  - the three experiment ids and the prompt hash;
  - the stop rule, each line marked met or not met;
  - the RM per-case citations for both runs;
  - every v2 gained or lost question, with a one-line read of each;
  - the verdict: **ship** or **not adopted**, and why.

```bash
git add docs/superpowers/specs/2026-10-06-one-candidate-knob-results.md
git commit -m "docs: role-aware citation rule measured (#10 step 4)"
```

- [ ] **Step 8: If the stop rule failed: STOP.** Report it to the user with the read answers. The branch stays unmerged; a different wording is a new decision, not a retry inside this plan.

### Task 4: Ship (only if Task 3 passed, on the user's go)

**Files:**
- Modify: `CLAUDE.md` (the production profile gains the new `system_prompt_sha12`)

- [ ] **Step 1: STOP — ask the user** to confirm shipping, and show the Task 3 verdict.
- [ ] **Step 2: Update `CLAUDE.md`.** In the production profile's **LLM** line, append `, answer prompt sha12 \`<hash>\` (role rule, 2026-10-16)`. Same line count; check `wc -l CLAUDE.md` ≤ 150. Commit: `docs(CLAUDE.md): production answers with the role-aware prompt`.
- [ ] **Step 3: Merge** `feat/role-aware-citations` into `main` (`--no-ff`, as the repo does). Run the suite on the merged result: all pass.
- [ ] **Step 4: User pushes `main`** to both remotes, and on the VM: `cd ~/rag && git pull && GIT_COMMIT=$(git describe --always --dirty) docker compose -f docker-compose-remote.yml build bot`. Then run the throwaway check `docker compose -f docker-compose-remote.yml run --rm --entrypoint python bot -c "from rag.run_identity import prompt_identity; print(prompt_identity())"`. Expected: the new hash.
- [ ] **Step 5: User replaces the bot:** `GIT_COMMIT=$(git describe --always --dirty) docker compose -f docker-compose-remote.yml up -d bot`, then one real question in Teams. Expected: the trace's `identity.system_prompt_sha12` equals the new hash, and `identity.git_commit` is the merge.
- [ ] **Step 6: Rollback, if needed:** check out the previous merge on the VM and `build` + `up -d` again. No `.env` change is involved.
