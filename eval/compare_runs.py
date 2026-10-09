#!/usr/bin/env python3
"""Compare two Phoenix experiments question by question.

    python eval/compare_runs.py <baseline_experiment_id> <candidate_experiment_id> \
        [--phoenix-url http://phoenix:6006]

With 61 questions, one recovered miss and one new miss average to "no
change". The query-rewrite decision (design 2026-10-09, Measurement 2 and the
stop rule) is made on this listing: misses recovered, working questions lost,
rank moves, and what the rewrite cost in latency. Read-only against Phoenix.
"""

import argparse
import statistics
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluators import _extract_expected, _match_result  # noqa: E402


def first_match_rank(output: dict | None, reference: dict) -> int | None:
    if not output:
        return None
    expectations = _extract_expected(reference)
    for rank, r in enumerate(output.get("search_results") or [], start=1):
        if any(_match_result(r, e) for e in expectations):
            return rank
    return None


def compare(baseline: list[dict], candidate: list[dict]) -> dict:
    base = {r["input"]["question"]: r for r in baseline}
    out = {k: [] for k in ("recovered", "lost", "better", "worse", "same")}
    latency, fallbacks = [], 0
    for run in candidate:
        q = run["input"]["question"]
        if q not in base:
            continue
        b = first_match_rank(base[q]["output"], base[q]["reference_output"])
        c = first_match_rank(run["output"], run["reference_output"])
        rewrite = (run["output"] or {}).get("rewrite") or {}
        if "latency_ms" in rewrite:
            latency.append(rewrite["latency_ms"])
        fallbacks += bool(rewrite.get("fallback"))
        if b is None and c is not None:
            key = "recovered"
        elif b is not None and c is None:
            key = "lost"
        elif b == c:
            key = "same"
        else:
            key = "better" if c < b else "worse"
        out[key].append((q, b, c))
    out["latency_ms"] = latency
    out["fallbacks"] = fallbacks
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
    print(f"\nSAME: {len(c['same'])}   rewrite fallbacks: {c['fallbacks']}")
    if c["latency_ms"]:
        lat = sorted(c["latency_ms"])
        p95 = lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))]
        print(f"rewrite latency ms: p50 {statistics.median(lat):.0f}  p95 {p95}  max {lat[-1]}")


if __name__ == "__main__":
    main()
