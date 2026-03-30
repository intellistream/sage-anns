"""Approximate Nearest Neighbor (ANN) interfaces for SAGE.

This module defines shared abstractions and registry helpers so algorithms can
live in ``isage-anns`` and be reused by benchmark and service-layer packages.
"""

from __future__ import annotations

from .base import AnnIndex, AnnIndexMeta
from .factory import AnnRegistryError, as_mapping, create, register, registered

__all__ = [
    "AnnIndex",
    "AnnIndexMeta",
    "AnnRegistryError",
    "register",
    "create",
    "registered",
    "as_mapping",
]
