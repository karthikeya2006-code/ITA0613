"""
Scalability and Runtime/Memory Benchmarks for Instance-Based Pipeline
Analyzes computational complexity, memory scaling, and spatial indexing across 10^3 to 10^5 records.
Aligned with Course Outcomes CO6, CO7.
"""

import time
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.knn import compute_euclidean_distance


def benchmark_scalability(
    scale_sizes: List[int] = [1000, 5000, 10000, 50000, 100000],
    num_features: int = 14,
    num_query_repeats: int = 20,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Measure exact distance latency (ms) and feature matrix memory footprint (MB)
    as dataset size N scales from 10^3 to 10^5 records.
    """
    np.random.seed(random_state)
    results = []
    
    # Standard single query vector
    query_vector = np.random.randn(1, num_features).astype(np.float64)
    
    for N in scale_sizes:
        # Generate random feature matrix of size N x D
        X_matrix = np.random.randn(N, num_features).astype(np.float64)
        
        # Memory consumption in Megabytes (MB)
        mem_mb = round(X_matrix.nbytes / (1024 * 1024), 4)
        
        # Warm-up run
        _ = compute_euclidean_distance(X_matrix, query_vector)
        
        # Time distance computation across multiple runs
        latencies = []
        for _ in range(num_query_repeats):
            t_start = time.perf_counter()
            dists = compute_euclidean_distance(X_matrix, query_vector)
            _ = np.argpartition(dists[0], kth=min(11, N-1))[:11]
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0) # convert to milliseconds
            
        avg_latency_ms = float(np.mean(latencies))
        std_latency_ms = float(np.std(latencies))
        
        results.append({
            'Records_N': N,
            'Features_D': num_features,
            'Memory_MB': mem_mb,
            'Query_Latency_ms': round(avg_latency_ms, 4),
            'Latency_Std_ms': round(std_latency_ms, 4),
            'Theoretical_FLOPs': int(2 * N * num_features)
        })
        
    df_results = pd.DataFrame(results)
    return df_results


class SimpleKDNode:
    """Node for a pure Python/NumPy k-d Tree for spatial indexing."""
    def __init__(self, point_idx: int, split_dim: int, left=None, right=None):
        self.point_idx = point_idx
        self.split_dim = split_dim
        self.left = left
        self.right = right


class SimpleKDTree:
    """
    Pure NumPy k-d Tree prototype for spatial acceleration of k-NN search.
    Reduces average search complexity from O(Nd) to O(d log N) in moderate dimensions.
    """
    def __init__(self, X: np.ndarray, max_depth: int = 15):
        self.X = X
        self.max_depth = max_depth
        self.root = self._build_tree(np.arange(len(X)), depth=0)

    def _build_tree(self, indices: np.ndarray, depth: int):
        if len(indices) == 0 or depth >= self.max_depth:
            return None
        split_dim = depth % self.X.shape[1]
        
        # Sort indices along split dimension
        sorted_indices = indices[np.argsort(self.X[indices, split_dim])]
        median_idx = len(sorted_indices) // 2
        
        node = SimpleKDNode(
            point_idx=sorted_indices[median_idx],
            split_dim=split_dim,
            left=self._build_tree(sorted_indices[:median_idx], depth + 1),
            right=self._build_tree(sorted_indices[median_idx + 1:], depth + 1)
        )
        return node
