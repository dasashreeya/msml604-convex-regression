"""
Proximal gradient solver library for regularized linear regression.

Provides from-scratch implementations of ISTA and FISTA for Ridge, Lasso, and
Elastic Net regression, together with supporting utilities for proximal operators,
backtracking line search, and Fenchel duality gap certificates.

All solvers minimize the composite objective

    F(β) = (1/2n) ‖y − Xβ‖² + λ Ω(β)

where Ω is the regularizer of choice (‖·‖², ‖·‖₁, or a convex combination),
n is the number of observations, and λ > 0 is the regularization strength.

Modules
-------
proximal      : closed-form proximal operators for Ridge, Lasso, Elastic Net
ista          : ISTA (proximal gradient descent) solver class
fista         : FISTA (Nesterov-accelerated ISTA) solver class
line_search   : backtracking line search for adaptive step-size selection
duality_gap   : Fenchel duality gap certificates for Lasso and Elastic Net
"""

from .ista import ISTA
from .fista import FISTA
from .proximal import prox_ridge, prox_lasso, prox_elasticnet
from .duality_gap import lasso_duality_gap, elasticnet_duality_gap
from .line_search import backtracking_line_search

__all__ = [
    "ISTA",
    "FISTA",
    "prox_ridge",
    "prox_lasso",
    "prox_elasticnet",
    "lasso_duality_gap",
    "elasticnet_duality_gap",
    "backtracking_line_search",
]
