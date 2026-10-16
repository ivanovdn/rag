#!/usr/bin/env python3
"""Compare two Phoenix experiments question by question.

    python eval/compare_runs.py <baseline_experiment_id> <candidate_experiment_id> \
        [--phoenix-url http://phoenix:6006]

With 61 questions, one recovered miss and one new miss average to "no
change". The query-rewrite decision (design 2026-10-09, Measurement 2 and the
stop rule) is made on this listing: misses recovered, working questions lost,
rank moves, and what the rewrite cost in latency. It also lists, per question,
what happened to the expected clause on its way into the answer (cited, or
missed by selection / retrieval, or escalated), the citations gained and lost,
and the per-status counts (#10). Read-only against Phoenix.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluators import (  # noqa: E402
    _extract_expected,
    _match_result,
    citation_clause_accuracy,
    citation_section_accuracy,
)


def first_match_rank(output: dict | None, reference: dict) -> int | None:
    if not output:
        return None
    expectations = _extract_expected(reference)
    for rank, r in enumerate(output.get("search_results") or [], start=1):
        if any(_match_result(r, e) for e in expectations):
            return rank
    return None


def citation_status(output: dict | None, reference: dict) -> str:
    """What happened to the expected clause on its way into the answer.

    Graded at the deepest level the label has. Not every policy has named
    clauses (the AI policy, the Privacy Notice), and for those the section is
    the label: citation_clause_accuracy skips them, so they fall back to
    citation_section_accuracy. Built on those two Phoenix columns so the listing
    cannot disagree with them. 'selection' is #10's case: the labelled clause
    or section was in the sources and the model cited another one.
    """
    if not output:
        return "retrieval"
    if (output.get("escalation") or {}).get("needed") and not output.get("citations"):
        return "escalated"
    scored = citation_clause_accuracy(output, reference)
    if scored["label"] == "skip":
        scored = citation_section_accuracy(output, reference)
    if scored["label"] == "skip":
        return "unlabelled"
    if scored["score"] == 1.0:
        return "cited"
    return "selection" if first_match_rank(output, reference) else "retrieval"


def _key(run: dict) -> tuple[str, int]:
    return run["input"]["question"], run.get("repetition_number") or 1


def _failed(run: dict) -> bool:
    """Raised, or answered "service unavailable": the chatbot task does not raise
    on an outage, it returns status 'unavailable' with nothing retrieved."""
    if run.get("error") or run.get("output") is None:
        return True
    return run["output"].get("status") == "unavailable"


def compare(baseline: list[dict], candidate: list[dict]) -> dict:
    """Classify each candidate question against its baseline counterpart.

    A run that raised (tier1 raises on a retrieval outage) is ERRORED on either
    side, never "lost" or "recovered": an embedder blip must not decide the stop
    rule. Pairs are keyed by question AND repetition, so repetitions are
    compared pairwise instead of silently overwriting each other.
    """
    base = {_key(r): r for r in baseline}
    out = {k: [] for k in ("recovered", "lost", "better", "worse", "same", "errored")}
    latency, fallbacks, unmatched = [], 0, 0
    cites = {"gained": [], "lost": [], "base": Counter(), "cand": Counter()}
    for run in candidate:
        key = _key(run)
        if key not in base:
            unmatched += 1
            continue
        q = key[0]
        if _failed(base[key]) or _failed(run):
            out["errored"].append((q, base[key].get("error"), run.get("error")))
            continue
        bs = citation_status(base[key]["output"], base[key]["reference_output"])
        cs = citation_status(run["output"], run["reference_output"])
        cites["base"][bs] += 1
        cites["cand"][cs] += 1
        if cs == "cited" and bs != "cited":
            cites["gained"].append((q, bs, cs))
        elif bs == "cited" and cs != "cited":
            cites["lost"].append((q, bs, cs))
        b = first_match_rank(base[key]["output"], base[key]["reference_output"])
        c = first_match_rank(run["output"], run["reference_output"])
        rewrite = run["output"].get("rewrite") or {}
        if "latency_ms" in rewrite:
            latency.append(rewrite["latency_ms"])
        fallbacks += bool(rewrite.get("fallback"))
        if b is None and c is not None:
            kind = "recovered"
        elif b is not None and c is None:
            kind = "lost"
        elif b == c:
            kind = "same"
        else:
            kind = "better" if c < b else "worse"
        out[kind].append((q, b, c))
    out["latency_ms"] = latency
    out["fallbacks"] = fallbacks
    out["unmatched"] = unmatched
    out["citations"] = cites
    return out


def _fetch(base_url: str, experiment_id: str) -> list[dict]:
    resp = requests.get(f"{base_url.rstrip('/')}/v1/experiments/{experiment_id}/json", timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--phoenix-url", default="http://localhost:6006")
    a = p.parse_args()

    c = compare(_fetch(a.phoenix_url, a.baseline), _fetch(a.phoenix_url, a.candidate))
    for key in ("recovered", "lost", "better", "worse"):
        print(f"\n{key.upper()} ({len(c[key])})")
        for q, b, n in c[key]:
            print(f"  {str(b):>4} -> {str(n):<4} {q}")
    print(f"\nERRORED ({len(c['errored'])}) -- excluded from the decision; re-run them")
    for q, b_err, c_err in c["errored"]:
        print(f"  baseline: {b_err or 'ok'} | candidate: {c_err or 'ok'} | {q}")
    print(f"\nSAME: {len(c['same'])}   rewrite fallbacks: {c['fallbacks']}   "
          f"candidate questions not in baseline: {c['unmatched']}")
    if c["latency_ms"]:
        lat = sorted(c["latency_ms"])
        p95 = lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))]
        print(f"rewrite latency ms: p50 {statistics.median(lat):.0f}  p95 {p95}  max {lat[-1]}")
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


if __name__ == "__main__":
    main()
