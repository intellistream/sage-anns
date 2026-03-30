# ADR 0003: Enforce fail-fast factory registry and one-way boundary

- Date: 2026-03-01
- Status: Accepted
- Issue: https://github.com/intellistream/sage-anns/issues/7

## Context

`factory.py` used a silent registration path (`_register_if_available`) that swallowed
`ImportError` during default algorithm registration. This made backend absence look like algorithm
absence and blurred boundaries between Python factory/wrapper code and backend implementations.

## Decision

1. Replace silent registration with explicit default algorithm specs in `sage_anns.factory`.
2. Resolve algorithm classes on demand by spec (`module_path`, `class_name`) and cache resolved
   classes.
3. Fail fast when backend module import fails or algorithm class is missing; do not silently skip.
4. Keep a strict one-way dependency surface: factory resolves wrappers explicitly, wrappers own
   backend import details.

## Consequences

- Missing backend errors are explicit and attributable to concrete algorithm names.
- Registry behavior is deterministic and no longer depends on hidden import side effects.
- No shim/re-export/fallback path is introduced.

## Validation

- `ruff check python/sage_anns tests`
- `PYTHONPATH=python pytest -q tests/test_issue7_boundary_registry_failfast.py tests/test_issue8_registry_cleanup.py`
