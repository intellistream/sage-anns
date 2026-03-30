"""Regression tests for the migrated ANN interface layer."""

from __future__ import annotations

import numpy as np


def test_interface_symbols_are_exported() -> None:
    """Root package should expose the migrated interface symbols."""
    import sage_anns

    assert hasattr(sage_anns, "AnnIndex")
    assert hasattr(sage_anns, "AnnIndexMeta")
    assert hasattr(sage_anns, "AnnRegistryError")
    assert hasattr(sage_anns, "create")
    assert hasattr(sage_anns, "register")
    assert hasattr(sage_anns, "registered")
    assert hasattr(sage_anns, "as_mapping")


def test_migrated_interface_registry_and_dummy_impl() -> None:
    """Migrated interface registry should create the built-in dummy index."""
    from sage_anns.interface.factory import _registry, create, registered
    from sage_anns.interface.implementations import register_builtin

    _registry.clear()
    register_builtin()

    assert "dummy_bruteforce" in registered()

    index = create("dummy_bruteforce", metric="euclidean")
    index.setup(dtype="float32", max_points=16, dim=3)

    vectors = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=np.float32,
    )
    ids = np.array([10, 20, 30], dtype=np.uint32)
    queries = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    index.insert(vectors, ids)
    found_ids, distances = index.search(queries, k=2)

    assert found_ids.shape == (1, 2)
    assert distances.shape == (1, 2)
    assert found_ids[0, 0] == 10
    assert distances[0, 0] == 0.0
