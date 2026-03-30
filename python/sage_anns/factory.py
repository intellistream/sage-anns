"""Factory for creating ANNS index instances."""

from __future__ import annotations

import importlib
from typing import Any

_ALGORITHM_REGISTRY: dict[str, type] = {}
_RESOLVED_DEFAULT_CLASSES: dict[str, type] = {}

_DEFAULT_ALGORITHM_SPECS: dict[str, tuple[str, str]] = {
    "candy_flat": ("sage_anns.algorithms.candy", "CANDYFlatIndex"),
    "candy_nndescent": ("sage_anns.algorithms.candy", "CANDYNNDescentIndex"),
    "candy_lshapg": ("sage_anns.algorithms.candy", "CANDYLSHAPGIndex"),
    "candy_onlinepq": ("sage_anns.algorithms.candy", "CANDYOnlinePQIndex"),
    "candy_dpg": ("sage_anns.algorithms.candy", "CANDYDPGIndex"),
    "faiss": ("sage_anns.algorithms.candy", "FAISSIndex"),
    "faiss_hnsw": ("sage_anns.algorithms.candy", "FAISSHNSWIndex"),
    "vsag_hnsw": ("sage_anns.algorithms.vsag_wrapper", "VSAGHNSWIndex"),
    "gti": ("sage_anns.algorithms.gti_wrapper", "GTIIndex"),
    "plsh": ("sage_anns.algorithms.plsh_wrapper", "PLSHIndex"),
}


def register_algorithm(name: str, cls: type) -> None:
    """Register an ANNS algorithm implementation.

    Args:
        name: Algorithm name (e.g., "faiss_hnsw")
        cls: Algorithm class implementing the ANNS interface

    Raises:
        ValueError: If the algorithm name is already registered
    """
    if name in _DEFAULT_ALGORITHM_SPECS or name in _ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' is already registered")
    _ALGORITHM_REGISTRY[name] = cls


def _resolve_default_algorithm_class(algorithm: str) -> type:
    """Resolve one default algorithm class by explicit spec.

    Raises:
        ValueError: If algorithm is not in default specs.
        ImportError: If the module cannot be imported.
        RuntimeError: If class is missing from the imported module.
    """
    if algorithm not in _DEFAULT_ALGORITHM_SPECS:
        raise ValueError(f"Unknown default algorithm '{algorithm}'")

    if algorithm in _RESOLVED_DEFAULT_CLASSES:
        return _RESOLVED_DEFAULT_CLASSES[algorithm]

    module_path, class_name = _DEFAULT_ALGORITHM_SPECS[algorithm]
    try:
        module = importlib.import_module(module_path)
    except ImportError as error:
        raise ImportError(
            f"Failed to import backend module '{module_path}' for algorithm '{algorithm}'."
        ) from error

    if not hasattr(module, class_name):
        raise RuntimeError(
            f"Algorithm class '{class_name}' not found in module '{module_path}' for '{algorithm}'."
        )

    cls = getattr(module, class_name)
    _RESOLVED_DEFAULT_CLASSES[algorithm] = cls
    return cls


def list_algorithms() -> list[str]:
    """List all registered ANNS algorithms.

    Returns:
        List of algorithm names
    """
    algorithms = list(_DEFAULT_ALGORITHM_SPECS.keys())
    algorithms.extend(_ALGORITHM_REGISTRY.keys())
    return algorithms


def create_index(algorithm: str, **kwargs: Any) -> Any:
    """Create an ANNS index instance.

    Args:
        algorithm: Algorithm name (e.g., "faiss_hnsw", "diskann")
        **kwargs: Algorithm-specific parameters

    Returns:
        ANNS index instance

    Raises:
        ValueError: If algorithm is not registered
    """
    if algorithm in _ALGORITHM_REGISTRY:
        cls = _ALGORITHM_REGISTRY[algorithm]
        return cls(**kwargs)

    if algorithm in _DEFAULT_ALGORITHM_SPECS:
        cls = _resolve_default_algorithm_class(algorithm)
        return cls(**kwargs)

    available = ", ".join(list_algorithms())
    raise ValueError(f"Unknown algorithm '{algorithm}'. Available algorithms: {available}")
