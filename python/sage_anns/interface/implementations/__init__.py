"""Built-in ANN implementations.

Heavy dependencies live behind optional imports; registration is explicit via
``register_builtin()`` to avoid side effects on import.
"""

from __future__ import annotations

from .dummy import register_dummy

__all__ = ["register_dummy", "register_builtin"]


def register_builtin() -> None:
    """Register lightweight built-in ANN implementations."""

    register_dummy()
