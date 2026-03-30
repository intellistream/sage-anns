# ADR 0001: Cross-Backend Consistency and ABI Regression Guard

- Status: Accepted
- Date: 2026-03-01
- Issue: intellistream/sage-anns#9

## Context

Issue #9 requires two safeguards:

- verify behavior consistency across available ANNS backends under the same contract;
- prevent binary ABI regressions caused by libstdc++ ABI mismatch.

The repository currently aggregates multiple native backends and bindings. A silent ABI shift
can break runtime loading/linking even when Python-level APIs look unchanged.

## Decision

1. Add regression test `tests/test_issue9_cross_backend_abi_regression.py`:
   - run the same deterministic dataset/query flow on available backends (`faiss_hnsw`, `vsag_hnsw`);
   - enforce shared output contract (shape, finite distances, valid id range);
   - enforce exact-vector top-1 identity consistency for stable cross-backend behavior.
2. Add ABI guard in `implementations/CMakeLists.txt` for `PyCANDYAlgo`:
   - `target_compile_definitions(PyCANDYAlgo PRIVATE _GLIBCXX_USE_CXX11_ABI=0)`.
3. Add source-level regression assertion that ABI pin exists.

## Consequences

- Cross-backend behavior drift is caught by CI tests.
- ABI mismatch risk is reduced for mixed native dependencies.
- No shim/re-export alias path is introduced.

## Verification

- `ruff check tests/test_issue9_cross_backend_abi_regression.py`
- `pytest -q tests/test_issue9_cross_backend_abi_regression.py`
