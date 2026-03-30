# ADR 0002: Unify factory registry and remove old algorithm alias

- Date: 2026-03-01
- Status: Accepted
- Issue: https://github.com/intellistream/sage-anns/issues/8

## Context

`sage_anns` previously had two maintenance problems:

1. Old alias usage (`candy_hnsw`) remained in tests/examples/docs even though it was not part of
   the active registry.
2. Default algorithm registration logic lived in package `__init__`, while historical wrapper code
   (`algorithms/pycandy_wrapper.py`) duplicated algorithm implementation paths.

This caused confusing behavior and made factory registration boundaries unclear.

## Decision

1. Keep one default registration path in `sage_anns.factory`.
2. Initialize default registry lazily and exactly once in factory internals.
3. Remove duplicate module `algorithms/pycandy_wrapper.py`.
4. Remove stale alias usage (`candy_hnsw`) from repository call sites; use canonical names from the
   unified factory registry.

## Consequences

- Registry behavior is deterministic and centralized.
- No shim/re-export alias path is introduced.
- Historical duplicated wrapper path is eliminated.
- Tests/examples/docs now match the real registry surface.

## Validation

- `ruff check python/sage_anns tests examples`
- `PYTHONPATH=python pytest -q tests/test_issue8_registry_cleanup.py tests/test_basic.py`
