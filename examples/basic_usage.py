"""
SAGE ANNS Usage Examples

This file demonstrates how to use the sage_anns package with different algorithms.
"""

import numpy as np
from sage_anns import create_index, list_algorithms


def example_basic_usage():
    """Basic usage example with CANDY HNSW."""
    print("=" * 60)
    print("Example 1: Basic Usage with CANDY HNSW")
    print("=" * 60)
    
    # Generate random data
    dimension = 128
    n_vectors = 10000
    n_queries = 100
    
    np.random.seed(42)
    data = np.random.randn(n_vectors, dimension).astype('float32')
    queries = np.random.randn(n_queries, dimension).astype('float32')
    
    # Create index
    index = create_index(
        algorithm="candy_hnsw",
        dimension=dimension,
        metric="l2",
        M=16,
        ef_construction=200,
        ef_search=50
    )
    
    print(f"Created index: {index}")
    
    # Build index
    print("Building index...")
    index.build(data)
    print(f"Index built: {index.num_vectors} vectors")
    
    # Search
    print(f"Searching for {n_queries} queries...")
    distances, indices = index.search(queries, k=10)
    
    print(f"Search results shape: {distances.shape}")
    print(f"Top-5 neighbors for first query:")
    print(f"  Indices: {indices[0, :5]}")
    print(f"  Distances: {distances[0, :5]}")
    print()


def example_faiss_hnsw():
    """Example with FAISS HNSW."""
    print("=" * 60)
    print("Example 2: FAISS HNSW")
    print("=" * 60)
    
    dimension = 64
    n_vectors = 5000
    
    np.random.seed(42)
    data = np.random.randn(n_vectors, dimension).astype('float32')
    queries = np.random.randn(10, dimension).astype('float32')
    
    # Create FAISS index with inner product metric
    index = create_index(
        algorithm="faiss_hnsw",
        dimension=dimension,
        metric="inner_product",  # For maximum inner product search
        M=32,
        ef_construction=200
    )
    
    index.build(data)
    print(f"FAISS index built with {index.num_vectors} vectors")
    
    # Search with custom ef parameter
    distances, indices = index.search(queries, k=5, ef=100)
    print(f"Found top-5 neighbors using ef=100")
    print()


def example_incremental_insertion():
    """Example of incremental insertion."""
    print("=" * 60)
    print("Example 3: Incremental Insertion")
    print("=" * 60)
    
    dimension = 32
    
    np.random.seed(42)
    initial_data = np.random.randn(1000, dimension).astype('float32')
    new_data = np.random.randn(500, dimension).astype('float32')
    
    # Build initial index
    index = create_index(
        algorithm="candy_hnsw",
        dimension=dimension,
        metric="l2"
    )
    
    index.build(initial_data)
    print(f"Initial index: {index.num_vectors} vectors")
    
    # Add more vectors
    index.add(new_data)
    print(f"After insertion: {index.num_vectors} vectors")
    
    # Search works with all vectors
    query = np.random.randn(1, dimension).astype('float32')
    distances, indices = index.search(query, k=10)
    print(f"Search successful across all {index.num_vectors} vectors")
    print()


def example_vsag():
    """Example with VSAG (if available)."""
    print("=" * 60)
    print("Example 4: VSAG HNSW")
    print("=" * 60)
    
    try:
        dimension = 128
        n_vectors = 5000
        
        np.random.seed(42)
        data = np.random.randn(n_vectors, dimension).astype('float32')
        queries = np.random.randn(10, dimension).astype('float32')
        
        index = create_index(
            algorithm="vsag_hnsw",
            dimension=dimension,
            metric="cosine",  # VSAG supports cosine similarity
            M=16,
            ef_construction=100
        )
        
        index.build(data)
        print(f"VSAG index built with {index.num_vectors} vectors")
        
        distances, indices = index.search(queries, k=10)
        print(f"Search completed successfully")
        print()
        
    except ValueError as e:
        if "Unknown algorithm" in str(e):
            print("VSAG not available (pyvsag not installed)")
            print("To install: cd implementations/vsag && make pyvsag && pip install wheelhouse/pyvsag*.whl")
        else:
            raise
    print()


def example_gti():
    """Example with GTI (if available)."""
    print("=" * 60)
    print("Example 5: GTI (Graph-based Tree Index)")
    print("=" * 60)
    
    try:
        dimension = 64
        n_vectors = 2000
        
        np.random.seed(42)
        data = np.random.randn(n_vectors, dimension).astype('float32')
        queries = np.random.randn(10, dimension).astype('float32')
        
        index = create_index(
            algorithm="gti",
            dimension=dimension,
            metric="l2",
            m=16,
            L=100
        )
        
        index.build(data)
        print(f"GTI index built with {index.num_vectors} vectors")
        
        # GTI supports efficient insertions
        new_data = np.random.randn(500, dimension).astype('float32')
        index.add(new_data)
        print(f"After insertion: {index.num_vectors} vectors")
        
        distances, indices = index.search(queries, k=10)
        print(f"Search completed successfully")
        print()
        
    except ValueError as e:
        if "Unknown algorithm" in str(e):
            print("GTI not available (gti_wrapper not built)")
            print("To build: cd implementations/gti/GTI/build && cmake .. && make gti_wrapper")
        else:
            raise
    print()


def example_plsh():
    """Example with PLSH (if available)."""
    print("=" * 60)
    print("Example 6: PLSH (Parallel LSH)")
    print("=" * 60)
    
    try:
        dimension = 64
        n_vectors = 3000
        
        np.random.seed(42)
        # PLSH is optimized for sparse vectors, but works with dense too
        data = np.random.randn(n_vectors, dimension).astype('float32')
        queries = np.random.randn(10, dimension).astype('float32')
        
        index = create_index(
            algorithm="plsh",
            dimension=dimension,
            metric="l2",
            k=10,  # Hash functions per table
            m=10,  # Number of hash tables
            num_threads=4
        )
        
        index.build(data)
        print(f"PLSH index built with {index.num_vectors} vectors")
        
        distances, indices = index.search(queries, k=10)
        print(f"Search completed successfully")
        print()
        
    except ValueError as e:
        if "Unknown algorithm" in str(e):
            print("PLSH not available (plsh_python not built)")
            print("To build: cd implementations/plsh/build && cmake .. && make")
        else:
            raise
    print()


def example_different_metrics():
    """Example showing different distance metrics."""
    print("=" * 60)
    print("Example 5: Different Distance Metrics")
    print("=" * 60)
    
    dimension = 64
    n_vectors = 1000
    
    np.random.seed(42)
    data = np.random.randn(n_vectors, dimension).astype('float32')
    query = np.random.randn(1, dimension).astype('float32')
    
    metrics = ["l2", "inner_product"]
    
    for metric in metrics:
        try:
            index = create_index(
                algorithm="faiss_hnsw",
                dimension=dimension,
                metric=metric,
                M=16
            )
            
            index.build(data)
            distances, indices = index.search(query, k=5)
            
            print(f"Metric: {metric}")
            print(f"  Top neighbor distances: {distances[0, :3]}")
        except Exception as e:
            print(f"Metric: {metric} - Error: {e}")
    
    print()


def list_available_algorithms():
    """List all available algorithms."""
    print("=" * 60)
    print("Available Algorithms")
    print("=" * 60)
    
    algorithms = list_algorithms()
    
    if algorithms:
        print(f"Found {len(algorithms)} algorithm(s):")
        for algo in algorithms:
            print(f"  - {algo}")
    else:
        print("No algorithms registered yet.")
        print("\nTo build algorithms:")
        print("  cd implementations && ./build_all.sh")
    
    print()


if __name__ == "__main__":
    # List available algorithms first
    list_available_algorithms()
    
    # Run examples if algorithms are available
    algorithms = list_algorithms()
    
    if "candy_hnsw" in algorithms:
        example_basic_usage()
        example_incremental_insertion()
    
    if "faiss_hnsw" in algorithms:
        example_faiss_hnsw()
        example_different_metrics()
    
    if "vsag_hnsw" in algorithms:
        example_vsag()
    
    if "gti" in algorithms:
        example_gti()
    
    if "plsh" in algorithms:
        example_plsh()
    
    if not algorithms:
        print("\n⚠️  No algorithms available yet.")
        print("Please build the algorithms first:")
        print("  cd implementations && ./build_all.sh")
