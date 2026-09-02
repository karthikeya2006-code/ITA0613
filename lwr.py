"""
From-Scratch Locally Weighted Regression (LWR)
Implements non-parametric local linear regression with Gaussian weighting kernel using NumPy.
Aligned with Course Outcomes CO6, CO7.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from src.knn import evaluate_regression


class FromScratchLWR:
    """
    Locally Weighted Linear Regression (LWR) using Gaussian Kernel:
    w_i(x) = exp(- ||x - x_i||^2 / (2 * tau^2))
    Local parameter solution: theta(x) = (X^T W X + lambda * I)^{-1} X^T W y
    Prediction: y_hat = [1, x]^T theta(x)
    """
    def __init__(self, tau: float = 1.0, reg_lambda: float = 1e-4, max_neighbors: Optional[int] = 1000):
        """
        Parameters:
        - tau: Bandwidth parameter controlling the locality of the weighting kernel.
        - reg_lambda: Ridge regularization for numerical stability of matrix inversion.
        - max_neighbors: Maximum active local neighbors to consider for fast vectorized solves.
        """
        self.tau = float(tau)
        self.reg_lambda = float(reg_lambda)
        self.max_neighbors = max_neighbors
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FromScratchLWR':
        """Store training data for query-time weighted regression."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        return self

    def _predict_single_point(self, x_query: np.ndarray) -> float:
        """
        Solve local normal equations for a single query point x_query.
        """
        # Squared Euclidean distances to all training points
        diffs = self.X_train - x_query  # (N, D)
        sq_dists = np.sum(diffs ** 2, axis=1)  # (N,)
        
        # Subselect top nearest points if dataset is very large to speed up matrix multiplication
        if self.max_neighbors is not None and self.max_neighbors < len(sq_dists):
            active_idx = np.argpartition(sq_dists, self.max_neighbors)[:self.max_neighbors]
            X_sub = self.X_train[active_idx]
            y_sub = self.y_train[active_idx]
            sq_dists_sub = sq_dists[active_idx]
        else:
            X_sub = self.X_train
            y_sub = self.y_train
            sq_dists_sub = sq_dists
            
        # Gaussian weights: w_i = exp(- dist^2 / (2 * tau^2))
        weights = np.exp(-sq_dists_sub / (2.0 * (self.tau ** 2)))  # (N_sub,)
        
        # Prevent degenerate weights
        if np.all(weights < 1e-15):
            weights = np.ones_like(weights)
            
        # Augment with intercept column: [1, x_1, x_2, ..., x_D]
        N_sub, D = X_sub.shape
        X_aug = np.hstack([np.ones((N_sub, 1)), X_sub])  # (N_sub, D + 1)
        x_query_aug = np.hstack([1.0, x_query])          # (D + 1,)
        
        # Weighted design matrix: X^T W X = X_aug^T * diag(weights) * X_aug = (X_aug * sqrt(w))^T (X_aug * sqrt(w))
        # Computationally efficient: (X_aug * weights[:, None])^T @ X_aug
        X_weighted = X_aug * weights[:, np.newaxis]
        XTWX = np.dot(X_weighted.T, X_aug)  # (D+1, D+1)
        XTWy = np.dot(X_weighted.T, y_sub)  # (D+1,)
        
        # Add ridge regularization
        XTWX_reg = XTWX + np.eye(D + 1) * self.reg_lambda
        
        # Solve linear system (XTWX_reg) theta = XTWy
        try:
            theta = np.linalg.solve(XTWX_reg, XTWy)
        except np.linalg.LinAlgError:
            theta = np.dot(np.linalg.pinv(XTWX_reg), XTWy)
            
        # Prediction
        return float(np.dot(x_query_aug, theta))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions for multiple query points.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("LWR model must be fitted before predict.")
            
        X_query = np.asarray(X, dtype=np.float64)
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)
            
        preds = np.zeros(X_query.shape[0], dtype=np.float64)
        for i in range(X_query.shape[0]):
            preds[i] = self._predict_single_point(X_query[i])
            
        return preds


def manual_tau_validation_curve(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    tau_values: List[float] = [0.6, 0.9, 1.2, 1.5, 2.0],
    max_eval_samples: int = 500
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Validation curve for bandwidth parameter tau in Locally Weighted Regression.
    """
    # Subsample validation set if large to maintain fast evaluation
    if len(X_val) > max_eval_samples:
        indices = np.random.RandomState(42).choice(len(X_val), max_eval_samples, replace=False)
        X_val_sub = X_val[indices]
        y_val_sub = y_val[indices]
    else:
        X_val_sub = X_val
        y_val_sub = y_val
        
    results = []
    best_tau = tau_values[0]
    best_rmse = float('inf')
    
    for tau in tau_values:
        model = FromScratchLWR(tau=tau)
        model.fit(X_train, y_train)
        preds = model.predict(X_val_sub)
        metrics = evaluate_regression(y_val_sub, preds)
        
        results.append({
            'tau': tau,
            'val_rmse': metrics['rmse'],
            'val_mae': metrics['mae'],
            'val_r2': metrics['r2']
        })
        
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_tau = tau
            
    return results, best_tau
