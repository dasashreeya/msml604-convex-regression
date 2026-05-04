"""
Backtracking line search for adaptive step-size selection in ISTA/FISTA.

Finds the smallest Lipschitz estimate L (equivalently the largest step size
1/L) satisfying the sufficient descent condition

    f(β_new) ≤ Q_L(β_new, β)

where Q_L is the quadratic upper bound on the smooth loss:

    Q_L(u, β) = f(β) + ∇f(β)ᵀ(u − β) + (L/2) ‖u − β‖²

Starting from an initial estimate L_init, the algorithm multiplies L by a
factor η > 1 (default η = 1.5) until the condition is satisfied.  This
amortized strategy avoids paying the full cost of computing σ_max(XᵀX) at
every iteration while guaranteeing sufficient decrease.

References
----------
Beck, A. and Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding
  Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences,
  2(1), 183–202.  (Algorithm 2 — Backtracking ISTA/FISTA)
"""

import numpy as np
from typing import Callable


def backtracking_line_search(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    grad: np.ndarray,
    prox_fn: Callable[[np.ndarray, float], np.ndarray],
    L_init: float,
    eta: float = 1.5,
) -> float:
    """
    Backtracking line search to find a valid Lipschitz estimate L.

    At each candidate L, computes β_new = prox(β - (1/L)*grad) and checks:

        F(β_new) ≤ Q_L(β_new, β)

    where the quadratic upper bound is:

        Q_L(u, β) = F(β) + ∇L(β)ᵀ(u - β) + (L/2)||u - β||²

    F here is the *smooth* data-fit term only: (1/2n)||y - Xβ||².
    The prox handles the regularization part.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,)
    beta : np.ndarray, shape (p,)
        Current iterate.
    grad : np.ndarray, shape (p,)
        Gradient of smooth loss at beta: -(1/n) Xᵀ(y - Xβ).
    prox_fn : callable
        prox_fn(v, step) → proximal output given step size 1/L.
        The caller must wrap the model-specific prox into this signature.
    L_init : float
        Initial Lipschitz estimate.
    eta : float
        Backtracking multiplier (L ← L * eta). Default 1.5.

    Returns
    -------
    float
        Updated L satisfying the descent condition.
    """
    n = X.shape[0]
    L = L_init

    r_beta = y - X @ beta
    f_beta = (1.0 / (2.0 * n)) * np.dot(r_beta, r_beta)

    for _ in range(100):  # hard cap to prevent infinite loop
        step = 1.0 / L
        beta_new = prox_fn(beta - step * grad, step)
        diff = beta_new - beta

        r_new = y - X @ beta_new
        f_new = (1.0 / (2.0 * n)) * np.dot(r_new, r_new)

        # Quadratic upper bound (smooth part only)
        q_bound = (
            f_beta
            + np.dot(grad, diff)
            + (L / 2.0) * np.dot(diff, diff)
        )

        if f_new <= q_bound + 1e-12:
            break
        L *= eta

    return L
