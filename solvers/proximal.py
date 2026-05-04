"""
Closed-form proximal operators for Ridge, Lasso, and Elastic Net regularization.

Each operator solves the proximal subproblem

    prox_{t·g}(v) = argmin_u { (1/2)‖u − v‖² + t·g(u) }

analytically, enabling exact proximal gradient steps without inner optimization.
All three operators are derived from the convex composite objective

    F(β) = (1/2n) ‖y − Xβ‖² + λ Ω(β)

where Ω(β) is the regularizer and t = 1/L is the gradient step size.

References
----------
Parikh, N. and Boyd, S. (2014). Proximal Algorithms. Foundations and Trends in
  Optimization, 1(3), 127–239.
"""

import numpy as np


def prox_ridge(beta: np.ndarray, lam: float, n: int) -> np.ndarray:
    """
    Proximal operator for Ridge (L2) regularization.

    Solves: argmin_u { (1/2)||u - v||² + λ||u||² }

    The 2n scaling arises from the objective form F(β) = (1/2n)||r||² + λ||β||²,
    giving an effective ridge parameter λ' = 2nλ internally.

    Parameters
    ----------
    beta : np.ndarray, shape (p,)
        Input vector (gradient-step iterate v).
    lam : float
        Regularization parameter λ (public interface, not scaled).
    n : int
        Number of observations (rows of X).

    Returns
    -------
    np.ndarray, shape (p,)
        beta / (1 + 2 * n * lam)
    """
    return beta / (1.0 + 2.0 * n * lam)


def prox_lasso(v: np.ndarray, lam: float) -> np.ndarray:
    """
    Proximal operator for Lasso (L1) regularization — soft thresholding.

    Solves: argmin_u { (1/2)||u - v||² + λ||u||₁ }
    Solution: sign(v) * max(|v| - λ, 0)

    Parameters
    ----------
    v : np.ndarray, shape (p,)
        Input vector.
    lam : float
        Regularization parameter λ.

    Returns
    -------
    np.ndarray, shape (p,)
        Soft-thresholded vector.
    """
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def prox_elasticnet(v: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """
    Proximal operator for Elastic Net regularization.

    Solves: argmin_u { (1/2)||u - v||² + λ1||u||₁ + λ2||u||² }
    Solution: soft_threshold(v, λ1) / (1 + 2*λ2)

    Parameters
    ----------
    v : np.ndarray, shape (p,)
        Input vector.
    lam1 : float
        L1 regularization parameter λ1.
    lam2 : float
        L2 regularization parameter λ2.

    Returns
    -------
    np.ndarray, shape (p,)
        Elastic Net proximal output.
    """
    return prox_lasso(v, lam1) / (1.0 + 2.0 * lam2)
