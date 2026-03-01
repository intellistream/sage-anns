"""SAGE ANNS: Approximate Nearest Neighbor Search algorithms.

This package provides high-performance C++ implementations of ANNS algorithms
with a unified Python interface.
"""

__version__ = "0.1.3"
__author__ = "IntelliStream Team"
__email__ = "shuhao_zhang@hust.edu.cn"

from .base import BaseANNSIndex
from .factory import create_index, list_algorithms, register_algorithm

__all__ = [
    "BaseANNSIndex",
    "create_index",
    "register_algorithm",
    "list_algorithms",
]
