"""
ISTA (Iterative Shrinkage-Thresholding Algorithm) for regularized regression.

Implements proximal gradient descent for the composite convex objective

    F(β) = (1/2n) ‖y − Xβ‖² + λ Ω(β)

via the fixed-point iteration

    β_{k+1} = prox_{(1/L)·λΩ}( β_k − (1/L) ∇f(β_k) )

where ∇f(β) = −(1/n) Xᵀ(y − Xβ) is the gradient of the smooth data-fit term
and L = σ_max(XᵀX)/n is the Lipschitz constant of ∇f.  An optional
backtracking line search adaptively estimates L at each iteration, yielding
larger effective step sizes on well-conditioned problems.

Convergence rate: O(1/k) in the objective gap F(β_k) − F(β*).

References
----------
Beck, A. and Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding
  Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences,
  2(1), 183–202.
"""

import numpy as np
from typing import List, Optional

from .proximal import prox_ridge, prox_lasso, prox_elasticnet
from .line_search import backtracking_line_search
from .duality_gap import lasso_duality_gap, elasticnet_duality_gap


class ISTA:
    """
    ISTA solver for Ridge, Lasso, and Elastic Net regression.

    Parameters
    ----------
    model : str
        One of 'ridge', 'lasso', 'elasticnet'.
    lam : float
        Primary regularization parameter λ.
    lam2 : float
        Secondary L2 parameter for Elastic Net (λ2). Ignored for other models.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the duality gap (Lasso/ElasticNet) or
        relative change in objective (Ridge).
    line_search : bool
        If True, use backtracking line search to adapt L each iteration.
    """

    def __init__(
        self,
        model: str,
        lam: float,
        lam2: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        line_search: bool = True,
    ) -> None:
        assert model in ("ridge", "lasso", "elasticnet"), (
            f"Unknown model '{model}'. Choose from 'ridge', 'lasso', 'elasticnet'."
        )
        self.model = model
        self.lam = lam
        self.lam2 = lam2
        self.max_iter = max_iter
        self.tol = tol
        self.line_search = line_search

        # Set after fit()
        self.coef_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []
        self.gap_history_: List[float] = []
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _objective(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        n = X.shape[0]
        r = y - X @ beta
        smooth = (1.0 / (2.0 * n)) * np.dot(r, r)
        if self.model == "ridge":
            reg = self.lam * np.dot(beta, beta)
        elif self.model == "lasso":
            reg = self.lam * np.sum(np.abs(beta))
        else:  # elasticnet
            reg = self.lam * np.sum(np.abs(beta)) + self.lam2 * np.dot(beta, beta)
        return float(smooth + reg)

    def _gradient(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        r = y - X @ beta
        return -(1.0 / n) * (X.T @ r)

    def _lipschitz(self, X: np.ndarray) -> float:
        n = X.shape[0]
        XtX = X.T @ X
        return float(np.linalg.eigvalsh(XtX).max()) / n

    def _prox(self, v: np.ndarray, step: float, n: int) -> np.ndarray:
        """Apply the model-specific proximal operator."""
        if self.model == "ridge":
            # prox_ridge expects lam scaled by step and needs n for 2n factor
            # prox_ridge(v, lam, n) = v/(1+2*n*lam); pass lam*step/n so
            # 2*n*(lam*step/n) = 2*lam*step = 2*lam/L (correct proximal scaling)
            return prox_ridge(v, self.lam * step / n, n)
        elif self.model == "lasso":
            return prox_lasso(v, self.lam * step)
        else:  # elasticnet
            return prox_elasticnet(v, self.lam * step, self.lam2 * step)

    def _duality_gap(
        self, X: np.ndarray, y: np.ndarray, beta: np.ndarray
    ) -> Optional[float]:
        if self.model == "lasso":
            return lasso_duality_gap(X, y, beta, self.lam)
        elif self.model == "elasticnet":
            return elasticnet_duality_gap(X, y, beta, self.lam, self.lam2)
        return None  # Ridge has no gap

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ISTA":
        """
        Run ISTA to fit the regularized regression model.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix. Should be standardized (zero mean, unit variance).
        y : np.ndarray, shape (n,)
            Response vector.

        Returns
        -------
        self
        """
        n, p = X.shape
        beta = np.zeros(p)
        L = self._lipschitz(X)

        self.loss_history_ = []
        self.gap_history_ = []

        # Wrap prox for line search (step size 1/L is baked in at call time)
        def prox_step(v: np.ndarray, step: float) -> np.ndarray:
            return self._prox(v, step, n)

        for k in range(self.max_iter):
            grad = self._gradient(X, y, beta)

            if self.line_search:
                L = backtracking_line_search(X, y, beta, grad, prox_step, L)

            step = 1.0 / L
            beta_new = self._prox(beta - step * grad, step, n)

            obj = self._objective(X, y, beta_new)
            self.loss_history_.append(obj)

            gap = self._duality_gap(X, y, beta_new)
            self.gap_history_.append(gap if gap is not None else float("nan"))

            beta = beta_new

            # Convergence check
            converged = False
            if gap is not None:
                converged = gap < self.tol
            else:
                # Ridge: ||full gradient||^2 < tol (gradient of entire objective)
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
