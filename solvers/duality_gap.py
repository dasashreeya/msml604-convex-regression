"""
Fenchel duality gaps for Lasso and Elastic Net.

Primal objectives use the form F(beta) = (1/2n)||y - X*beta||^2 + lam*Omega(beta).
The gap provides a certificate of suboptimality: gap >= 0, and gap=0 at optimum.
"""

import numpy as np


def lasso_duality_gap(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    lam: float,
) -> float:
    """
    Fenchel duality gap for the Lasso.

    Primal:
        P(beta) = (1/2n)||y - X*beta||^2 + lam*||beta||_1

    Dual variable (feasibility-projected residual):
        theta = r / (n * max(1, ||X^T r||_inf / (n*lam)))
        where r = y - X*beta

    Dual objective:
        D(theta) = (1/2n)||y||^2 - (1/(2n))||y - n*theta||^2

    Gap = P(beta) - D(theta)

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,)
    beta : np.ndarray, shape (p,)
    lam : float
        Regularization parameter lambda.

    Returns
    -------
    float
        Duality gap (>= 0).
    """
    n = X.shape[0]
    r = y - X @ beta

    # Primal value
    primal = (1.0 / (2.0 * n)) * np.dot(r, r) + lam * np.sum(np.abs(beta))

    # Dual variable: project r to satisfy ||X^T theta||_inf <= lam
    Xtr = X.T @ r
    scale = max(1.0, np.max(np.abs(Xtr)) / (n * lam))
    theta = r / (n * scale)

    # Dual objective
    dual = (1.0 / (2.0 * n)) * np.dot(y, y) - (1.0 / (2.0 * n)) * np.dot(
        y - n * theta, y - n * theta
    )

    gap = primal - dual
    assert gap >= -1e-10, f"Duality gap is negative: {gap}"
    return float(gap)


def elasticnet_duality_gap(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    lam1: float,
    lam2: float,
) -> float:
    """
    Fenchel duality gap for the Elastic Net.

    Primal:
        P(beta) = (1/2n)||y - X*beta||^2 + lam1*||beta||_1 + lam2*||beta||^2

    Uses the exact Fenchel-Rockafellar dual with theta = r/n (no projection
    needed since lam2 > 0 makes g* always finite):

        f(z) = (1/2n)||y - z||^2  =>  f*(v) = y^T v + (n/2)||v||^2
        g(beta) = lam1*||beta||_1 + lam2*||beta||^2
               =>  g*(v) = sum_j max(|v_j| - lam1, 0)^2 / (4*lam2)

        D(theta) = -f*(-theta) - g*(X^T theta)
                 = y^T theta - (n/2)||theta||^2
                   - sum_j max(|X^T theta_j| - lam1, 0)^2 / (4*lam2)

    With theta = r/n, at the EN optimum beta*:
        KKT gives X^T r* / n = lam1 * s* + 2*lam2 * beta*
        => g*(X^T r*/n) = lam2 * ||beta*||^2
        => gap = P(beta*) - D(r*/n) = 0  (verified by substitution)

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,)
    beta : np.ndarray, shape (p,)
    lam1 : float
        L1 regularization parameter lambda1.
    lam2 : float
        L2 regularization parameter lambda2 (must be > 0).

    Returns
    -------
    float
        Duality gap (>= 0).
    """
    if lam2 == 0.0:
        return lasso_duality_gap(X, y, beta, lam1)

    n = X.shape[0]
    r = y - X @ beta

    # Primal value
    primal = (
        (1.0 / (2.0 * n)) * np.dot(r, r)
        + lam1 * np.sum(np.abs(beta))
        + lam2 * np.dot(beta, beta)
    )

    # Dual variable: theta = r/n  (exact dual point, no projection for lam2 > 0)
    theta = r / n  # shape (n,)
    Xt_theta = X.T @ theta  # = X^T r / n, shape (p,)

    # g*(X^T theta) = sum_j max(|X^T theta_j| - lam1, 0)^2 / (4*lam2)
    g_conj = np.sum(np.maximum(np.abs(Xt_theta) - lam1, 0.0) ** 2) / (4.0 * lam2)

    # -f*(-theta) = y^T theta - (n/2)||theta||^2
    neg_f_conj = np.dot(y, theta) - (n / 2.0) * np.dot(theta, theta)

    dual = neg_f_conj - g_conj

    gap = primal - dual
    assert gap >= -1e-8, f"Elastic Net duality gap is negative: {gap}"
    return float(gap)
