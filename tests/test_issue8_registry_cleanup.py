"""Regression tests for issue #8: remove old alias and unify registry path."""

import pytest


def test_issue8_removed_alias_unavailable() -> None:
    """Removed alias candy_hnsw is unavailable in the unified registry."""
    from sage_anns import create_index, list_algorithms

    algorithms = list_algorithms()
    assert "candy_hnsw" not in algorithms

    with pytest.raises(ValueError, match="Unknown algorithm"):
        create_index("candy_hnsw", dimension=32)


def test_issue8_registry_is_stable_across_multiple_calls() -> None:
    """Default registry initialization should be idempotent."""
    from sage_anns import list_algorithms

    first = sorted(list_algorithms())
    second = sorted(list_algorithms())
    assert first == second
