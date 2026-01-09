# CANDY Algorithms Module Structure

This directory contains CANDY algorithm wrappers for the sage-anns package.

## Directory Structure

```
candy/
├── __init__.py         # Module exports
├── utils.py            # Shared utilities (lazy imports)
├── base.py             # Base CANDYIndex class
├── flat.py             # Flat (brute-force) index
├── nndescent.py        # NN-Descent graph index
├── lshapg.py           # LSH-APG index
├── onlinepq.py         # Online PQ index
├── dpg.py              # DPG index
├── faiss_wrapper.py    # FAISS integration
├── diskann.py          # DiskANN wrapper (experimental)
└── sptag.py            # SPTAG wrapper (experimental)
```

## Available Algorithms

### Core CANDY Algorithms

All CANDY algorithms use the unified `AbstractIndex` interface from PyCANDYAlgo.

1. **CANDYFlatIndex** (`flat.py`)
   - Brute-force exact search
   - 100% recall, O(n) complexity
   - Best for small datasets or when exact results are required

2. **CANDYNNDescentIndex** (`nndescent.py`)
   - Graph-based approximate search
   - Incremental k-NN graph construction
   - Good for high-dimensional data

3. **CANDYLSHAPGIndex** (`lshapg.py`)
   - LSH + Proximity Graph hybrid
   - Fast candidate generation with graph refinement
   - Suitable for sparse high-dimensional data

4. **CANDYOnlinePQIndex** (`onlinepq.py`)
   - Product Quantization with online updates
   - Memory-efficient compression
   - Supports dynamic insertions

5. **CANDYDPGIndex** (`dpg.py`)
   - Dynamic Proximity Graph
   - Optimized for frequent updates
   - Good for streaming workloads

### FAISS Integration

6. **FAISSIndex** (`faiss_wrapper.py`)
   - Generic FAISS factory interface
   - Supports: Flat, IVF, HNSW, PQ, etc.
   - Flexible index configuration

7. **FAISSHNSWIndex** (`faiss_wrapper.py`)
   - Optimized FAISS HNSW implementation
   - Gorder graph reordering support
   - Better cache locality

### Experimental

8. **DiskANNIndex** (`diskann.py`)
   - Disk-based billion-scale ANNS
   - Work in progress

9. **SPTAGIndex** (`sptag.py`)
   - Microsoft SPTAG integration
   - Work in progress

## Usage Examples

### Using Generic CANDYIndex

```python
from sage_anns.algorithms.candy import CANDYIndex
import numpy as np

# Create index with any CANDY algorithm
index = CANDYIndex(
    algorithm="nnDescent",  # or "flat", "LSHAPG", etc.
    dimension=128,
    metric="l2",
    k=50,  # algorithm-specific parameter
    iterations=10
)

# Build with data
data = np.random.randn(10000, 128).astype('float32')
index.build(data)

# Search
queries = np.random.randn(10, 128).astype('float32')
distances, indices = index.search(queries, k=10)
```

### Using Specific Algorithm Classes

```python
from sage_anns.algorithms.candy import CANDYNNDescentIndex

# More convenient with specific class
index = CANDYNNDescentIndex(
    dimension=128,
    metric="l2",
    k=50,
    iterations=10,
    sample_rate=0.5
)

index.build(data)
distances, indices = index.search(queries, k=10)
```

### Using FAISS

```python
from sage_anns.algorithms.candy import FAISSHNSWIndex

# FAISS HNSW with Gorder optimization
index = FAISSHNSWIndex(
    dimension=128,
    metric="l2",
    M=32,
    ef_construction=200,
    ef_search=64,
    use_gorder=True
)

index.build(data)
distances, indices = index.search(queries, k=10, ef=100)
```

## Design Principles

1. **Modularity**: Each algorithm in its own file
2. **Lazy Loading**: PyCANDYAlgo imported only when needed
3. **Consistent Interface**: All classes inherit from BaseANNSIndex
4. **Clear Separation**: Base class (base.py) vs specific algorithms
5. **Documentation**: Each file contains detailed docstrings

## Adding New Algorithms

To add a new CANDY algorithm:

1. Create `new_algo.py` with your algorithm class
2. Import it in `__init__.py`
3. Add to `__all__` list
4. Register in `sage_anns/__init__.py` if needed

Example:

```python
# new_algo.py
from .base import CANDYIndex

class NewAlgoIndex(CANDYIndex):
    def __init__(self, dimension: int, metric: str = "l2", **kwargs):
        super().__init__(
            algorithm="newAlgo",  # must match IndexTable name
            dimension=dimension,
            metric=metric,
            **kwargs
        )
```

## Dependencies

- **PyCANDYAlgo**: Core C++ module (must be built)
- **PyTorch**: Required for tensor operations
- **NumPy**: Array operations

## Build Instructions

```bash
cd implementations/build
cmake .. && make -j$(nproc)
```

The built module (`PyCANDYAlgo*.so`) must be in the Python path.
