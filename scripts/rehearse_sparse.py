#!/usr/bin/env python
"""Rehearse the native-sparse setup against a real Qdrant, on throwaway collections.

The 335 offline unit tests prove the SHAPE of every request this code sends.
None of them touches Qdrant, so none of them can prove the server on the other
end does what we expect. That is what this script is for: it exercises the real
code paths against a real server and reports pass/fail per check.

SAFETY. It creates two collections named `zz_sparse_rehearsal*`, uses them, and
deletes them in a finally block. It never writes to any other collection. It
reads `settings.qdrant_collection` exactly once, read-only, to confirm the live
pre-migration collection is correctly REJECTED by the write guard.

Deliberately does NOT call init_observability(): this script makes no LLM and no
embedding calls, and registering a tracer would push rehearsal spans into the
production Phoenix project. Nothing here imports LlamaIndex or Ollama, so the
usual ordering rule has nothing to order.

Usage:
    PYTHONPATH=. python scripts/rehearse_sparse.py            # prompts before writing
    PYTHONPATH=. python scripts/rehearse_sparse.py --yes      # no prompt
    PYTHONPATH=. python scripts/rehearse_sparse.py --keep     # leave collections for inspection
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import Distance, PointStruct, Prefetch, VectorParams

from config import settings
from rag.vector_store import (
    RRF_K,
    SPARSE_VECTOR_NAME,
    assert_sparse_vector,
    bm25_document,
    get_qdrant_client,
    init_collection,
    preflight_sparse_config,
    search_chunks,
)

SPARSE_COLLECTION = "zz_sparse_rehearsal"
NOSPARSE_COLLECTION = "zz_sparse_rehearsal_nosparse"

# Synthetic policy-shaped text. "install" appears only in the first one, so a
# query for "installation" can only match it through stemming.
FIXTURES = [
    (1, "Team Members are forbidden to install any unlicensed software on company laptops."),
    (2, "Backups are retained for ninety days and verified quarterly by the IT team."),
    (3, "Visitors must be escorted at all times while inside secure areas of the office."),
]

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Run a check function, record pass/fail, never let one failure stop the rest."""

    def decorator(fn):
        try:
            detail = fn()
            _results.append((name, True, detail or ""))
            print(f"  PASS  {name}" + (f"\n          {detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001 — a rehearsal reports, it does not raise
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
        return fn

    return decorator


def _random_dense() -> list[float]:
    return [random.random() for _ in range(settings.qdrant_vector_dim)]


class _Sparse:
    """indices/values, however the client hands the stored vector back."""

    def __init__(self, raw):
        if isinstance(raw, dict):
            self.indices, self.values = raw.get("indices", []), raw.get("values", [])
        else:
            self.indices, self.values = raw.indices, raw.values


def _sparse_of(point) -> _Sparse:
    vector = point.vector
    if not isinstance(vector, dict):
        raise AssertionError(f"expected a named-vector dict, got {type(vector).__name__}")
    if SPARSE_VECTOR_NAME not in vector:
        raise AssertionError(f"no '{SPARSE_VECTOR_NAME}' in stored vectors: {sorted(vector)}")
    return _Sparse(vector[SPARSE_VECTOR_NAME])


def run_checks(client) -> None:
    # --- 1. schema ----------------------------------------------------------
    @check("schema: init_collection builds unnamed dense + named sparse bm25/idf")
    def _():
        init_collection(SPARSE_COLLECTION)
        params = client.get_collection(SPARSE_COLLECTION).config.params
        sparse = params.sparse_vectors or {}
        assert SPARSE_VECTOR_NAME in sparse, f"sparse vectors: {sorted(sparse)}"
        modifier = str(sparse[SPARSE_VECTOR_NAME].modifier)
        assert "idf" in modifier.lower(), f"modifier is {modifier!r}, expected idf"
        dense = params.vectors
        # A bare VectorParams (not a dict) is what keeps the dense vector unnamed.
        assert hasattr(dense, "size"), f"dense vector is NAMED ({dense!r}) — search would break"
        assert dense.size == settings.qdrant_vector_dim, f"dim {dense.size}"
        return f"sparse={sorted(sparse)} modifier={modifier} dense=unnamed/{dense.size}/{dense.distance}"

    # --- 2. the server encodes, not us -------------------------------------
    @check("server-side encoding: Qdrant turns our text into a sparse vector")
    def _():
        client.upsert(
            collection_name=SPARSE_COLLECTION,
            points=[
                PointStruct(
                    id=pid,
                    vector={"": _random_dense(), SPARSE_VECTOR_NAME: bm25_document(text)},
                    payload={"text": text},
                )
                for pid, text in FIXTURES
            ],
            wait=True,
        )
        stored = client.retrieve(
            collection_name=SPARSE_COLLECTION, ids=[1], with_vectors=True
        )
        assert stored, "point 1 was not stored"
        sparse = _sparse_of(stored[0])
        n_indices, n_values = len(sparse.indices), len(sparse.values)
        assert n_indices > 0, "stored sparse vector is EMPTY — the server did not encode the text"
        assert n_indices == n_values, f"indices={n_indices} values={n_values}"
        return (
            f"{n_indices} terms encoded server-side from {len(FIXTURES[0][1].split())} words; "
            f"first values {[round(v, 4) for v in sparse.values[:3]]}"
        )

    # --- 3. stemming --------------------------------------------------------
    @check("stemming: 'software installation' matches a chunk containing 'install'")
    def _():
        hits = client.query_points(
            collection_name=SPARSE_COLLECTION,
            query=bm25_document("software installation"),
            using=SPARSE_VECTOR_NAME,
            limit=3,
            with_payload=True,
        ).points
        assert hits, "no sparse hits at all — server-side query encoding failed"
        assert hits[0].id == 1, f"top hit was id={hits[0].id}, expected the 'install' chunk (id=1)"
        return f"top hit id={hits[0].id} score={hits[0].score:.4f} ({len(hits)} hits)"

    # --- 4. avg_len is live -------------------------------------------------
    @check("avg_len: the per-document option actually changes stored weights")
    def _():
        text = FIXTURES[0][1]
        original = settings.bm25_avg_len
        try:
            weights = {}
            for pid, avg_len in ((901, 50.0), (902, 256.0)):
                settings.bm25_avg_len = avg_len
                client.upsert(
                    collection_name=SPARSE_COLLECTION,
                    points=[
                        PointStruct(
                            id=pid,
                            vector={"": _random_dense(), SPARSE_VECTOR_NAME: bm25_document(text)},
                            payload={"text": text},
                        )
                    ],
                    wait=True,
                )
                got = client.retrieve(
                    collection_name=SPARSE_COLLECTION, ids=[pid], with_vectors=True
                )
                weights[avg_len] = _sparse_of(got[0]).values[0]
        finally:
            settings.bm25_avg_len = original
        assert weights[50.0] != weights[256.0], (
            f"identical weights ({weights[50.0]}) at avg_len 50 and 256 — "
            "the option is being ignored, so BM25_AVG_LEN is dead"
        )
        return f"avg_len=50 -> {weights[50.0]:.6f}   avg_len=256 -> {weights[256.0]:.6f}"

    # --- 5. k=60 reaches the server ----------------------------------------
    @check(f"fusion: RRF_K={RRF_K} is on the wire, not Qdrant's default k=2")
    def _():
        orig_collection, orig_flag = settings.qdrant_collection, settings.bm25_enabled
        try:
            settings.qdrant_collection = SPARSE_COLLECTION
            # Forced on regardless of .env: with it off, search_chunks takes the
            # dense-only branch and returns a COSINE score, which this check
            # would then measure against an RRF threshold and misreport.
            settings.bm25_enabled = True
            points = search_chunks("software installation", _random_dense(), top_k=5)
        finally:
            settings.qdrant_collection, settings.bm25_enabled = orig_collection, orig_flag
        assert points, "fused query returned nothing"
        top = points[0].score
        # Two branches at k=60 cap at 1/60 + 1/60 = 0.0333. At Qdrant's default
        # k=2 the floor for a single branch is 1/(2 + limit) and the top is 0.5.
        assert top < 0.05, (
            f"top fused score {top:.6f} is far above the ~{1 / RRF_K:.5f} that k={RRF_K} "
            "produces — the server is fusing with a different k (default is 2 -> ~0.5)"
        )
        return f"top fused score {top:.6f} (1/{RRF_K} = {1 / RRF_K:.6f}); {len(points)} points"

    # Checks 6-8 need a collection shaped like the pre-migration one: dense, no
    # sparse. Built here rather than inside check 6 so that a failure there does
    # not cascade into two misleading failures below it.
    client.create_collection(
        collection_name=NOSPARSE_COLLECTION,
        vectors_config=VectorParams(
            size=settings.qdrant_vector_dim, distance=Distance.COSINE
        ),
    )

    # --- 6. a pre-migration collection rejects the sparse WRITE -------------
    @check("danger is real: upserting a sparse vector into a sparse-less collection fails")
    def _():
        try:
            client.upsert(
                collection_name=NOSPARSE_COLLECTION,
                points=[
                    PointStruct(
                        id=1,
                        vector={"": _random_dense(), SPARSE_VECTOR_NAME: bm25_document("x")},
                        payload={"text": "x"},
                    )
                ],
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001 — this failure is the expected result
            return f"rejected as expected: {type(exc).__name__}: {str(exc)[:120]}"
        raise AssertionError(
            "the upsert SUCCEEDED against a collection with no sparse vector — "
            "the premise of the ingest guard does not hold on this server"
        )

    # --- 7. the write guard catches it -------------------------------------
    @check("write guard: assert_sparse_vector rejects a real pre-migration collection")
    def _():
        try:
            assert_sparse_vector(NOSPARSE_COLLECTION, "Rehearsal.")
        except RuntimeError as exc:
            message = str(exc)
            assert NOSPARSE_COLLECTION in message, "message does not name the collection"
            assert SPARSE_VECTOR_NAME in message, "message does not name the missing vector"
            assert "migrate_collection.py" in message, "message does not say what to run"
            return f"raised, and names collection + vector + remedy: {message[:110]}..."
        raise AssertionError("assert_sparse_vector did NOT raise on a sparse-less collection")

    # --- 8. the query preflight catches it ---------------------------------
    @check("query preflight: refuses to start with BM25 on and no sparse vector")
    def _():
        orig_collection, orig_flag = settings.qdrant_collection, settings.bm25_enabled
        try:
            settings.qdrant_collection = NOSPARSE_COLLECTION
            settings.bm25_enabled = True
            try:
                preflight_sparse_config()
            except RuntimeError as exc:
                raised = str(exc)
            else:
                raise AssertionError("preflight_sparse_config did NOT raise")

            settings.bm25_enabled = False
            preflight_sparse_config()  # must be a no-op, and must not call Qdrant
        finally:
            settings.qdrant_collection, settings.bm25_enabled = orig_collection, orig_flag
        return f"raises with BM25 on, silent with it off: {raised[:100]}..."

    # --- 9. where the live collection currently stands (read-only) ----------
    # A report rather than a pass/fail: both answers are legitimate, and which
    # one you get tells you where in the rollout you are. It still fails loudly
    # if the collection is missing or unreachable.
    @check(f"report: migration state of the live collection '{settings.qdrant_collection}'")
    def _():
        live = settings.qdrant_collection
        try:
            assert_sparse_vector(live, "Rehearsal read-only check.")
        except RuntimeError:
            return (
                f"'{live}' has no sparse vector, so ingest would be blocked rather than "
                "deleting a policy and then failing the write"
            )
        return (
            f"'{live}' ALREADY has a sparse vector — it has been migrated, so the guard "
            "passes and ingest is safe against it"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rehearse native sparse vectors against a real Qdrant"
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument(
        "--keep", action="store_true", help="leave the throwaway collections in place"
    )
    args = parser.parse_args()

    url = settings.active_qdrant_url
    print("=" * 72)
    print(f"qdrant:            {url}")
    print(f"live collection:   {settings.qdrant_collection}  (read-only here)")
    print(f"will create:       {SPARSE_COLLECTION}, {NOSPARSE_COLLECTION}")
    print(f"will delete after: {'no (--keep)' if args.keep else 'yes'}")
    print("=" * 72)

    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("aborted")
            return 1

    client = get_qdrant_client()
    try:
        for name in (SPARSE_COLLECTION, NOSPARSE_COLLECTION):
            if client.collection_exists(name):
                print(f"removing leftover {name}")
                client.delete_collection(name)
        print()
        run_checks(client)
    except Exception:  # noqa: BLE001 — report, then always clean up
        traceback.print_exc()
    finally:
        if args.keep:
            print(f"\nkeeping {SPARSE_COLLECTION} and {NOSPARSE_COLLECTION} (--keep)")
        else:
            for name in (SPARSE_COLLECTION, NOSPARSE_COLLECTION):
                if client.collection_exists(name):
                    client.delete_collection(name)
            print(f"\ncleaned up {SPARSE_COLLECTION}, {NOSPARSE_COLLECTION}")

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n{passed}/{len(_results)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    print("All checks passed — the server behaves as this branch assumes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
