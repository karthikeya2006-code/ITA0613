"""
From-Scratch k-Nearest Neighbour (k-NN) Regressor
Implements Euclidean and Mahalanobis distance metrics using pure NumPy operations.
Aligned with Course Outcomes CO6, CO7.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any


def compute_euclidean_distance(X_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of Euclidean distances between query vectors and training points.
    Using identity: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 <u, v>
    
    Parameters:
    - X_train: (N, D) array of training feature vectors.
    - X_query: (M, D) array of query feature vectors.
    
    Returns:
    - Distances: (M, N) matrix where element (i, j) is distance from query i to train j.
    """
    if X_query.ndim == 1:
        X_query = X_query.reshape(1, -1)
        
    query_sq = np.sum(X_query ** 2, axis=1, keepdims=True)  # (M, 1)
    train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T # (1, N)
    cross_term = np.dot(X_query, X_train.T)                 # (M, N)
    
    dists_sq = np.maximum(0.0, query_sq + train_sq - 2.0 * cross_term)
    return np.sqrt(dists_sq)


def compute_mahalanobis_distance(
    X_train: np.ndarray, 
    X_query: np.ndarray, 
    cov_inv: Optional[np.ndarray] = None, 
    reg_lambda: float = 1e-4
) -> np.ndarray:
    """
    Vectorized computation of Mahalanobis distance accounting for feature correlations:
    d_M(x, x_i) = sqrt((x - x_i)^T S^{-1} (x - x_i))
    
    Parameters:
    - X_train: (N, D) array of training vectors.
    - X_query: (M, D) array of query vectors.
    - cov_inv: (D, D) precomputed regularized inverse covariance matrix.
    - reg_lambda: regularization added to diagonal of covariance matrix.
    
    Returns:
    - Distances: (M, N) matrix of Mahalanobis distances.
    """
    if X_query.ndim == 1:
        X_query = X_query.reshape(1, -1)
        
    N, D = X_train.shape
    M = X_query.shape[0]
    
    if cov_inv is None:
        cov = np.cov(X_train, rowvar=False)
        # Handle 1D feature edge case
        if cov.ndim == 0:
            cov = np.array([[cov]])
        # Regularization for numerical invertibility
        cov_reg = cov + np.eye(D) * reg_lambda
        cov_inv = np.linalg.pinv(cov_reg)
        
    # Efficient transformation: If S^{-1} = L L^T, then d_M(u, v) = ||L^T (u - v)||
    # Using eigen-decomposition or directly:
    # (u - v) S^{-1} (u - v)^T
    # Transform coordinates: X_train_trans = X_train @ (S^{-1/2}), then Euclidean distance
    try:
        eigenvals, eigenvecs = np.linalg.eigh(cov_inv)
        eigenvals = np.maximum(eigenvals, 1e-10)
        sqrt_cov_inv = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
        
        X_train_trans = np.dot(X_train, sqrt_cov_inv)
        X_query_trans = np.dot(X_query, sqrt_cov_inv)
        return compute_euclidean_distance(X_train_trans, X_query_trans)
    except Exception:
        # Fallback to direct pairwise difference computation
        dists = np.zeros((M, N), dtype=np.float64)
        for i in range(M):
            diff = X_train - X_query[i]  # (N, D)
            # diff @ cov_inv: (N, D)
            # sum((diff @ cov_inv) * diff, axis=1): (N,)
            val = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
            dists[i] = np.sqrt(np.maximum(0.0, val))
        return dists


class FromScratchKNNRegressor:
    """
    Instance-Based Learning: k-Nearest Neighbours Regressor implemented from scratch in NumPy.
    Supports Euclidean and Regularized Mahalanobis metrics.
    """
    def __init__(self, k: int = 5, metric: str = 'euclidean', reg_lambda: float = 1e-4):
        self.k = k
        self.metric = metric.lower()
        self.reg_lambda = reg_lambda
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FromScratchKNNRegressor':
        """
        Fit the instance-based model by storing the training records and precomputing covariance if needed.
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        
        if self.metric == 'mahalanobis':
            D = self.X_train.shape[1]
            cov = np.cov(self.X_train, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            cov_reg = cov + np.eye(D) * self.reg_lambda
            self.cov_inv = np.linalg.pinv(cov_reg)
            
        return self

    def predict(self, X: np.ndarray, return_neighbors: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Query the instance database and compute average target value of k nearest neighbours.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted with training data before making predictions.")
            
        X_query = np.asarray(X, dtype=np.float64)
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)
            
        if self.metric == 'euclidean':
            distances = compute_euclidean_distance(self.X_train, X_query)
        elif self.metric == 'mahalanobis':
            distances = compute_mahalanobis_distance(self.X_train, X_query, cov_inv=self.cov_inv, reg_lambda=self.reg_lambda)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'. Supported: 'euclidean', 'mahalanobis'.")
            
        # Extract k smallest distances along axis 1 (training points)
        # Using argpartition for O(N) top-k selection
        M = X_query.shape[0]
        k = min(self.k, self.X_train.shape[0])
        
        top_k_idx = np.argpartition(distances, kth=k-1, axis=1)[:, :k]
        
        # Sort top-k for each query to have exact order
        row_indices = np.arange(M)[:, None]
        top_k_dists = distances[row_indices, top_k_idx]
        sorted_order = np.argsort(top_k_dists, axis=1)
        sorted_idx = top_k_idx[row_indices, sorted_order]
        
        # Mean of target values of k neighbors
        neighbor_targets = self.y_train[sorted_idx]
        predictions = np.mean(neighbor_targets, axis=1)
        
        if return_neighbors:
            return predictions, sorted_idx, top_k_dists[row_indices, sorted_order]
            
        return predictions


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate RMSE, MAE, and R^2 score."""
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4), 'r2': round(r2, 4)}


def manual_k_validation_curve(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    k_values: List[int] = [3, 5, 7, 9, 11, 15, 21],
    metric: str = 'euclidean'
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Perform manual validation across candidate k values to determine the optimal neighborhood size.
    """
    results = []
    best_k = k_values[0]
    best_rmse = float('inf')
    
    for k in k_values:
        model = FromScratchKNNRegressor(k=k, metric=metric)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = evaluate_regression(y_val, preds)
        
        results.append({
            'k': k,
            'metric': metric,
            'val_rmse': metrics['rmse'],
            'val_mae': metrics['mae'],
            'val_r2': metrics['r2']
        })
        
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_k = k
            
    return results, best_k
