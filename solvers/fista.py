"""
FISTA (Fast ISTA) with Nesterov momentum for regularized regression.

Achieves O(1/k²) convergence vs ISTA's O(1/k) by using the momentum update:
    t_{k+1} = (1 + sqrt(1 + 4*t_k²)) / 2
    y_k = β_k + ((t_{k-1} - 1) / t_k) * (β_k - β_{k-1})
    β_{k+1} = prox(y_k - (1/L) ∇L(y_k))
"""

import numpy as np

from .ista import ISTA


class FISTA(ISTA):
    """
    FISTA solver — same interface as ISTA, with Nesterov momentum.

    Inherits all parameters from ISTA. The only difference is the fit()
    method, which applies the momentum extrapolation step.

    Parameters
    ----------
    model : str
        One of 'ridge', 'lasso', 'elasticnet'.
    lam : float
        Primary regularization parameter λ.
    lam2 : float
        Secondary L2 parameter for Elastic Net (λ2).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    line_search : bool
        If True, use backtracking line search.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FISTA":
        """
        Run FISTA with Nesterov momentum.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix.
        y : np.ndarray, shape (n,)
            Response vector.

        Returns
        -------
        self
        """
        n, p = X.shape
        beta = np.zeros(p)
        beta_prev = np.zeros(p)
        t = 1.0
        L = self._lipschitz(X)

        self.loss_history_ = []
        self.gap_history_ = []

        def prox_step(v: np.ndarray, step: float) -> np.ndarray:
            return self._prox(v, step, n)

        for k in range(self.max_iter):
            # Nesterov momentum extrapolation
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            momentum = (t - 1.0) / t_new
            y_momentum = beta + momentum * (beta - beta_prev)

            # Gradient at momentum point
            grad = self._gradient(X, y, y_momentum)

            if self.line_search:
                L = backtracking_line_search_fista(
                    X, y, y_momentum, grad, prox_step, L
                )

            step = 1.0 / L
            beta_new = self._prox(y_momentum - step * grad, step, n)

            obj = self._objective(X, y, beta_new)
            self.loss_history_.append(obj)

            gap = self._duality_gap(X, y, beta_new)
            self.gap_history_.append(gap if gap is not None else float("nan"))

            # Update state
            beta_prev = beta
            beta = beta_new
            t = t_new

            # Convergence check
            converged = False
            if gap is not None:
                converged = gap < self.tol
            else:
                # Ridge: ||full gradient||^2 < tol
                r_new = y - X @ beta_new
                grad_full = -(1.0 / n) * (X.T @ r_new) + 2.0 * self.lam * beta_new
                converged = float(np.dot(grad_full, grad_full)) < self.tol

            if converged:
                self.n_iter_ = k + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = beta
        return self


# ---------------------------------------------------------------------------
# Local import to avoid circular dependency with line_search module
# ---------------------------------------------------------------------------

from .line_search import backtracking_line_search as backtracking_line_search_fista
