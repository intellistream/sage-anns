"""Test basic ANNS functionality."""

import numpy as np
import pytest


def _is_pycandy_available():
    """Check if PyCANDYAlgo is available."""
    try:
        import PyCANDYAlgo
        return True
    except ImportError:
        return False


def _is_vsag_available():
    """Check if pyvsag is available."""
    try:
        import pyvsag
        return True
    except ImportError:
        return False


def _is_gti_available():
    """Check if gti_wrapper is available."""
    try:
        import gti_wrapper
        return True
    except ImportError:
        return False


def _is_plsh_available():
    """Check if plsh_python is available."""
    try:
        import plsh_python
        return True
    except ImportError:
        return False


def test_import():
    """Test that sage_anns can be imported."""
    import sage_anns
    
    assert hasattr(sage_anns, '__version__')
    assert hasattr(sage_anns, 'create_index')
    assert hasattr(sage_anns, 'list_algorithms')
    assert hasattr(sage_anns, 'BaseANNSIndex')


def test_list_algorithms():
    """Test listing algorithms."""
    from sage_anns import list_algorithms
    
    algorithms = list_algorithms()
    assert isinstance(algorithms, list)
    # Should have at least some algorithms registered
    print(f"Available algorithms: {algorithms}")


def test_factory_unknown_algorithm():
    """Test that unknown algorithm raises error."""
    from sage_anns import create_index
    
    with pytest.raises(ValueError, match="Unknown algorithm"):
        create_index("nonexistent_algorithm")


@pytest.mark.skipif(
    not _is_pycandy_available(),
    reason="PyCANDYAlgo not built"
)
def test_candy_hnsw_basic():
    """Test CANDY HNSW basic functionality."""
    from sage_anns import create_index
    
    # Create small test dataset
    dimension = 32
    n_vectors = 100
    n_queries = 10
    k = 5
    
    np.random.seed(42)
    data = np.random.randn(n_vectors, dimension).astype('float32')
    queries = np.random.randn(n_queries, dimension).astype('float32')
    
    # Create and build index
    index = create_index(
        "candy_hnsw",
        dimension=dimension,
        metric="l2",
        M=8,
        ef_construction=50
    )
    
    assert not index.is_built
    index.build(data)
    assert index.is_built
    assert index.num_vectors == n_vectors
    
    # Search
    distances, indices = index.search(queries, k=k)
    
    assert distances.shape == (n_queries, k)
    assert indices.shape == (n_queries, k)
    assert np.all(indices >= 0)
    assert np.all(indices < n_vectors)


@pytest.mark.skipif(
    not _is_pycandy_available(),
    reason="PyCANDYAlgo not built"
)
def test_faiss_hnsw_basic():
    """Test FAISS HNSW basic functionality."""
    from sage_anns import create_index
    
    dimension = 32
    n_vectors = 100
    n_queries = 10
    k = 5
    
    np.random.seed(42)
    data = np.random.randn(n_vectors, dimension).astype('float32')
    queries = np.random.randn(n_queries, dimension).astype('float32')
    
    index = create_index(
        "faiss_hnsw",
        dimension=dimension,
        metric="l2",
        M=8
    )
    
    index.build(data)
    distances, indices = index.search(queries, k=k)
    
    assert distances.shape == (n_queries, k)
    assert indices.shape == (n_queries, k)


@pytest.mark.skipif(
    not _is_vsag_available(),
    reason="pyvsag not installed"
)
def test_vsag_hnsw_basic():
    """Test VSAG HNSW basic functionality."""
    from sage_anns import create_index
    
    dimension = 32
    n_vectors = 100
    n_queries = 10
    k = 5
    
    np.random.seed(42)
    data = np.random.randn(n_vectors, dimension).astype('float32')
    queries = np.random.randn(n_queries, dimension).astype('float32')
    
    index = create_index(
        "vsag_hnsw",
        dimension=dimension,
        metric="l2",
        M=8
    )
    
    index.build(data)
    distances, indices = index.search(queries, k=k)
    
    assert distances.shape == (n_queries, k)
    assert indices.shape == (n_queries, k)


@pytest.mark.skipif(
    not _is_gti_available(),
    reason="gti_wrapper not built"
)
def test_gti_basic():
    """Test GTI basic functionality."""
    from sage_anns import create_index
    
    dimension = 32
    n_vectors = 500  # GTI needs more data to build multi-level tree
    n_queries = 10
    k = 5
    
    np.random.seed(42)
    data = np.random.randn(n_vectors, dimension).astype('float32')
    queries = np.random.randn(n_queries, dimension).astype('float32')
    
    index = create_index(
        "gti",
        dimension=dimension,
        metric="l2",
        capacity_up_i=50,  # Smaller capacity to ensure tree splits
        capacity_up_l=50,
        m=8
    )
    
    index.build(data)
    distances, indices = index.search(queries, k=k)
    
    assert distances.shape == (n_queries, k)
    assert indices.shape == (n_queries, k)


@pytest.mark.skipif(
    not _is_plsh_available(),
    reason="plsh_python not built"
)
def test_plsh_basic():
    """Test PLSH basic functionality."""
    from sage_anns import create_index
    
    dimension = 32
    n_vectors = 100
    n_queries = 10
    k = 5
    
    np.random.seed(42)
    data = np.random.randn(n_vectors, dimension).astype('float32')
    queries = np.random.randn(n_queries, dimension).astype('float32')
    
    index = create_index(
        "plsh",
        dimension=dimension,
        metric="l2",
        k=4, 
        m=5
    )
    
    index.build(data)
    distances, indices = index.search(queries, k=k)
    
    assert distances.shape == (n_queries, k)
    assert indices.shape == (n_queries, k)
