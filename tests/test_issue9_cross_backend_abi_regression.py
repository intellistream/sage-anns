"""Issue #9 regression tests: cross-backend consistency and ABI guard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def _create_index(algorithm: str, **kwargs):
    """Import create_index lazily after source-path setup."""
    from sage_anns import create_index

    return create_index(algorithm, **kwargs)


def _can_create_backend(name: str) -> bool:
    """Return whether backend is usable (create, build, search)."""
    try:
        index = _create_index(name, dimension=16, metric="l2")
        probe_data = np.zeros((4, 16), dtype=np.float32)
        index.build(probe_data)
        index.search(probe_data[:1], k=1)
        return True
    except Exception:
        return False


def _build_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic dataset and exact-id queries."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((64, 16), dtype=np.float32)
    query_ids = np.array([0, 10, 20], dtype=np.int64)
    queries = data[query_ids].copy()
    return data, queries, query_ids


def test_cross_backend_top1_identity_consistency() -> None:
    """Available backends must keep top-1 identity on exact vector lookup."""
    candidates = ["faiss_hnsw", "vsag_hnsw"]
    available = [name for name in candidates if _can_create_backend(name)]

    if len(available) < 2:
        pytest.skip("Need at least two available backends for cross-backend consistency check")

    data, queries, query_ids = _build_dataset()
    top1_by_backend: dict[str, np.ndarray] = {}

    for backend in available:
        index = _create_index(backend, dimension=16, metric="l2")
        index.build(data)
        distances, indices = index.search(queries, k=5)

        assert distances.shape == (len(query_ids), 5)
        assert indices.shape == (len(query_ids), 5)
        assert np.all(np.isfinite(distances))
        assert np.all(indices >= 0)
        assert np.all(indices < len(data))

        top1 = indices[:, 0].astype(np.int64)
        assert np.array_equal(top1, query_ids)
        top1_by_backend[backend] = top1

    reference_backend = available[0]
    reference = top1_by_backend[reference_backend]
    for backend in available[1:]:
        assert np.array_equal(reference, top1_by_backend[backend])


def test_issue9_pycandy_target_pins_abi_flag() -> None:
    """PyCANDYAlgo target must pin old libstdc++ ABI for compatibility."""
    cmake_path = REPO_ROOT / "implementations" / "CMakeLists.txt"
    source = cmake_path.read_text(encoding="utf-8")

    assert "target_compile_definitions(PyCANDYAlgo PRIVATE" in source
    assert "_GLIBCXX_USE_CXX11_ABI=0" in source
