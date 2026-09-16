"""How much of the router's in-pipeline latency is client construction?

Offline: constructs clients, makes no LLM calls.

    PYTHONPATH=. python scripts/bench_llm_construction.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.observability import init_observability

init_observability()

from rag.agent import get_llm

build = []
for _ in range(10):
    t0 = time.perf_counter()
    get_llm()
    build.append(time.perf_counter() - t0)

print(f"get_llm() construction: median {statistics.median(build) * 1000:7.1f} ms "
      f"| min {min(build) * 1000:.1f} | max {max(build) * 1000:.1f}")
print("Router call measured at 650 ms raw HTTP vs 1.8 s in-pipeline;")
print("this figure is how much of that ~1.2 s gap construction explains.")
print("Do NOT fix it with lru_cache(get_llm) — see the gotcha in CLAUDE.md.")
