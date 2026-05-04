"""
Synthetic benchmark datasets for evaluating regularized regression solvers.

Three structured data-generating processes are provided, each designed to expose
a distinct difficulty regime encountered in real-world regression problems:

1. **High Correlation** — features share a latent signal, inducing near-linear
   dependence among columns of X.  Tests robustness under multicollinearity,
   where Elastic Net and Ridge are theoretically preferable to Lasso.

2. **High Dimensional** (p ≫ n) — a sparse true coefficient vector embedded in
   an ambient space far exceeding the sample size.  The canonical regime for
   compressed sensing and Lasso theory (Donoho, 2006; Candès & Tao, 2007).

3. **Near Singular** — X is constructed via SVD with a prescribed condition
   number, producing a design matrix whose smallest singular values approach
   zero.  Tests numerical stability of gradient computations and step-size
   estimation.

All datasets return (X, y, β_true) with X standardized to zero mean and unit
variance per column, matching the convention expected by the ISTA/FISTA solvers.

References
----------
Donoho, D.L. (2006). Compressed sensing. IEEE Transactions on Information
  Theory, 52(4), 1289–1306.
Zou, H. and Hastie, T. (2005). Regularization and variable selection via the
  Elastic Net. Journal of the Royal Statistical Society: Series B, 67(2), 301–320.
"""

import numpy as np
from typing import Tuple


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma


def high_correlation(
    n: int = 200,
    p: int = 50,
    rho: float = 0.95,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dataset where columns of X are highly correlated (corr ≈ rho).

    Each column = shared_signal + small_noise, so X[:,i] and X[:,j]
    have Pearson correlation ≈ rho.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    rho : float
        Target pairwise correlation (0 < rho < 1).
    noise : float
        Per-column noise standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n, p)  — standardized
    y : np.ndarray, shape (n,)
    beta_true : np.ndarray, shape (p,)
    """
    rng = np.random.default_rng(seed)
    base_signal = rng.standard_normal(n)
    # Each column: sqrt(rho)*base + sqrt(1-rho)*noise
    X = np.sqrt(rho) * base_signal[:, None] + np.sqrt(1.0 - rho) * rng.standard_normal(
        (n, p)
    )
    X = _standardize(X)

    beta_true = rng.standard_normal(p)
    y = X @ beta_true + noise * rng.standard_normal(n)
    return X, y, beta_true


def high_dimensional(
    n: int = 100,
    p: int = 500,
    sparsity: int = 10,
    noise: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    High-dimensional dataset (p >> n) with sparse true coefficient vector.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features (p >> n).
    sparsity : int
        Number of non-zero entries in beta_true.
    noise : float
        Observation noise standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n, p)  — standardized
    y : np.ndarray, shape (n,)
    beta_true : np.ndarray, shape (p,)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X = _standardize(X)

    beta_true = np.zeros(p)
    nz_idx = rng.choice(p, size=sparsity, replace=False)
    beta_true[nz_idx] = rng.standard_normal(sparsity)

    y = X @ beta_true + noise * rng.standard_normal(n)
    return X, y, beta_true


def near_singular(
    n: int = 200,
    p: int = 50,
    condition_number: float = 1e6,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dataset constructed via SVD with singular values spanning condition_number.

    X = U @ diag(linspace(1, 1/condition_number, p)) @ Vt

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    condition_number : float
        Ratio of largest to smallest singular value.
    noise : float
        Observation noise standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n, p)  — standardized
    y : np.ndarray, shape (n,)
    beta_true : np.ndarray, shape (p,)
    """
    rng = np.random.default_rng(seed)
    rank = min(n, p)

    # Random orthonormal bases via QR
    U, _ = np.linalg.qr(rng.standard_normal((n, rank)))
    Vt, _ = np.linalg.qr(rng.standard_normal((p, rank)))
    Vt = Vt.T  # shape (rank, p)

    singular_values = np.linspace(1.0, 1.0 / condition_number, rank)
    X = U @ (singular_values[:, None] * Vt)  # (n, p)
    X = _standardize(X)

    beta_true = rng.standard_normal(p)
    y = X @ beta_true + noise * rng.standard_normal(n)
    return X, y, beta_true
