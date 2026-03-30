"""DiskANN wrapper (experimental)."""

import warnings

import numpy as np

from ...base import BaseANNSIndex


class DiskANNIndex(BaseANNSIndex):
    """DiskANN implementation through PyCANDYAlgo.

    DiskANN is a disk-based ANNS algorithm from Microsoft that can handle
    billion-scale datasets with limited memory.

    Note: This wrapper is experimental. For full DiskANN functionality,
    use PyCANDYAlgo.diskannpy directly.

    Args:
        dimension: Vector dimension
        metric: Distance metric ('l2', 'inner_product')
        R: Graph degree (default: 64)
        L: Build complexity (default: 100)
        **kwargs: Additional parameters
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "l2",
        R: int = 64,  # noqa: N803
        L: int = 100,  # noqa: N803
        **kwargs,
    ):
        super().__init__(dimension, metric, **kwargs)
        self.R = R
        self.L = L
        self._index = None

        warnings.warn(
            "DiskANN wrapper is experimental. Full integration pending. "
            "Use PyCANDYAlgo.diskannpy directly for now.",
            UserWarning,
        )

    def build(self, data: np.ndarray) -> None:
        """Build DiskANN index."""
        raise NotImplementedError(
            "DiskANN build not yet fully integrated. Use PyCANDYAlgo.diskannpy directly for now."
        )

    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Add vectors."""
        raise NotImplementedError("DiskANN add not yet implemented")

    def search(
        self, queries: np.ndarray, k: int = 10, **search_params
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search DiskANN index."""
        raise NotImplementedError("DiskANN search not yet implemented")
