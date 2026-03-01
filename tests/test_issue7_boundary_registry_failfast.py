"""Issue #7 regression tests: enforce one-way boundary and fail-fast registry behavior."""

from __future__ import annotations

import importlib

import pytest


def test_issue7_list_algorithms_is_spec_driven() -> None:
    """Default algorithm list should come from explicit factory specs."""
    from sage_anns.factory import _DEFAULT_ALGORITHM_SPECS, list_algorithms

    listed = list_algorithms()
    for name in _DEFAULT_ALGORITHM_SPECS:
        assert name in listed


def test_issue7_missing_backend_is_failfast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory should fail fast on backend import error instead of silently skipping."""
    from sage_anns import create_index
    from sage_anns import factory as factory_module

    factory_module._RESOLVED_DEFAULT_CLASSES.pop("vsag_hnsw", None)

    original_import_module = importlib.import_module

    def _patched_import(name: str, package: str | None = None):
        if name == "sage_anns.algorithms.vsag_wrapper":
            raise ImportError("simulated backend import failure")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _patched_import)

    with pytest.raises(ImportError, match="vsag_hnsw"):
        create_index("vsag_hnsw", dimension=16)
