# Regularized Regression Solver Library

Proximal gradient solvers (ISTA and FISTA) for Ridge, Lasso, and Elastic Net regression.

## Objective Function

All solvers minimize:

```
F(beta) = (1/2n) * ||y - X*beta||^2 + lambda * Omega(beta)
```

- `n`: number of observations (rows of X)
- `p`: number of features (columns of X)
- `r = y - X*beta`: residuals
- `L = sigma_max(X^T X) / n`: initial Lipschitz constant of the gradient

Gradient of smooth part: `∇L(beta) = -(1/n) * X^T * (y - X*beta)`

---

## Proximal Operators (`proximal.py`)

The proximal operator for regularizer `g` with step size `t = 1/L` is:

```
prox_{t*g}(v) = argmin_u { (1/2)||u - v||^2 + t * g(u) }
```

### Ridge (L2)

```
g(beta) = lambda * ||beta||^2
prox_{t*g}(v) = v / (1 + 2*t*lambda)
```

**Implementation note**: `prox_ridge(v, lam, n)` returns `v / (1 + 2*n*lam)`.
The `n` factor encodes the convention that callers pass `lam * step / n`, so the
effective scaling `2 * n * (lam * step / n) = 2 * lam * step` is correct.
This keeps the public interface uniform: all solvers accept the objective-level `lambda`.

### Lasso (L1) — Soft Thresholding

```
g(beta) = lambda * ||beta||_1
prox_{t*g}(v)_j = sign(v_j) * max(|v_j| - t*lambda, 0)
```

This is the elementwise soft-thresholding operator.

### Elastic Net (L1 + L2)

```
g(beta) = lambda1 * ||beta||_1 + lambda2 * ||beta||^2
prox_{t*g}(v) = soft_threshold(v, t*lambda1) / (1 + 2*t*lambda2)
```

The L1 and L2 proximal operators compose exactly in this separable form.

---

## Duality Gaps (`duality_gap.py`)

The duality gap provides an *exact, certificate* of sub-optimality: `gap >= 0` always,
and `gap = 0` at the optimum (under strong duality).

### Lasso Duality Gap

**Primal**: `P(beta) = (1/2n)||y-X*beta||^2 + lambda*||beta||_1`

**Dual variable** (project residuals to feasibility):
```
theta = r / (n * max(1, ||X^T r||_inf / (n*lambda)))
```
where `r = y - X*beta`. This ensures `||X^T theta||_inf <= lambda`.

**Dual objective**:
```
D(theta) = (1/2n)||y||^2 - (1/(2n))||y - n*theta||^2
```

**Gap** = P(beta) - D(theta) >= 0.

### Elastic Net Duality Gap

**Primal**: `P(beta) = (1/2n)||y-X*beta||^2 + lambda1*||beta||_1 + lambda2*||beta||^2`

Uses Fenchel-Rockafellar duality. The conjugate of the regularizer:
```
g*(v) = sum_j max(|v_j| - lambda1, 0)^2 / (4*lambda2)
```

**Dual variable** (no projection needed since `g*` is always finite for `lambda2 > 0`):
```
theta = r / n
```

**Dual objective**:
```
D(theta) = y^T theta - (n/2)||theta||^2 - sum_j max(|X^T theta_j| - lambda1, 0)^2 / (4*lambda2)
```

**Correctness at optimum**: KKT gives `X^T r* / n = lambda1 * s* + 2*lambda2 * beta*`, so:
`g*(X^T theta*) = lambda2 * ||beta*||^2` and `P(beta*) - D(theta*) = 0` (verified by substitution).

---

## Algorithms

### ISTA

Proximal gradient descent:
```
1. grad = -(1/n) * X^T * (y - X*beta)
2. v = beta - (1/L) * grad
3. beta_new = prox_{(1/L)*g}(v)
```
Convergence rate: O(1/k) in objective gap.

### FISTA

Nesterov momentum acceleration:
```
t_new = (1 + sqrt(1 + 4*t^2)) / 2
y_k = beta + ((t_old - 1) / t_new) * (beta - beta_prev)
grad = -(1/n) * X^T * (y - X*y_k)
beta_new = prox_{(1/L)*g}(y_k - (1/L) * grad)
```
Convergence rate: O(1/k^2) in objective gap — quadratically faster than ISTA.

### Backtracking Line Search

Finds the smallest L satisfying the descent condition:
```
f(beta_new) <= f(beta) + grad^T (beta_new - beta) + (L/2)||beta_new - beta||^2
```
where `f` is the smooth loss only. Updates `L <- L * eta` (default `eta=1.5`) until satisfied.

---

## File Structure

```
solvers/
  __init__.py      exports ISTA, FISTA, prox_*, *_duality_gap
  proximal.py      prox operators for Ridge, Lasso, Elastic Net
  ista.py          ISTA solver class
  fista.py         FISTA solver class (inherits ISTA)
  line_search.py   backtracking line search
  duality_gap.py   Fenchel duality gaps
  README.md        this file

experiments/
  synthetic.py     hard dataset generators
  convergence.py   ISTA vs FISTA convergence plots
  plots/           saved figures

tests/
  test_all.py      sklearn validation suite
```
