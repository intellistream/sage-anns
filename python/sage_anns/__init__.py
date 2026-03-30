"""SAGE ANNS: Approximate Nearest Neighbor Search algorithms.

This package provides high-performance C++ implementations of ANNS algorithms
with a unified Python interface.

It also owns the lightweight ANN registry/interface layer migrated from
``sage-libs`` so external implementations can share a stable contract without
pulling algorithm backends into the interface package.
"""

__version__ = "0.1.3"
__author__ = "IntelliStream Team"
__email__ = "shuhao_zhang@hust.edu.cn"

from .base import BaseANNSIndex
from .factory import create_index, list_algorithms, register_algorithm
from .interface import (
    AnnIndex,
    AnnIndexMeta,
    AnnRegistryError,
    as_mapping,
    create,
    register,
    registered,
)

__all__ = [
    "BaseANNSIndex",
    "AnnIndex",
    "AnnIndexMeta",
    "AnnRegistryError",
    "create_index",
    "register_algorithm",
    "list_algorithms",
    "create",
    "register",
    "registered",
    "as_mapping",
]
