# Solver Library: ISTA and FISTA for Regularized Regression

From-scratch implementations of proximal gradient descent (ISTA) and its
Nesterov-accelerated variant (FISTA) for Ridge, Lasso, and Elastic Net
regression.  All solvers minimize the composite convex objective

```
F(β) = (1/2n) ‖y − Xβ‖²  +  λ Ω(β)
```

where `n` is the number of observations, `p` is the number of features, and
`Ω` is the chosen regularizer.  The gradient of the smooth data-fit term is

```
∇f(β) = −(1/n) Xᵀ(y − Xβ)
```

with Lipschitz constant `L = σ_max(XᵀX) / n`.

---

## Proximal Operators (`proximal.py`)

Each proximal operator solves the subproblem

```
prox_{t·g}(v) = argmin_u { (1/2)‖u − v‖²  +  t · g(u) }
```

analytically, enabling exact gradient steps without any inner optimization loop.

### Ridge (L2)

```
g(β) = λ ‖β‖²
prox_{t·g}(v) = v / (1 + 2t λ)
```

**Implementation note**: `prox_ridge(v, lam, n)` returns `v / (1 + 2·n·lam)`.
The caller passes `lam * step / n` so the effective denominator `1 + 2·n·(lam·step/n)
= 1 + 2·lam·step` is correct.  This keeps the public interface uniform across
all three regularizers.

### Lasso (L1) — Soft Thresholding

```
g(β) = λ ‖β‖₁
prox_{t·g}(v)_j = sign(v_j) · max(|v_j| − tλ, 0)
```

The elementwise soft-thresholding operator.

### Elastic Net (L1 + L2)

```
g(β) = λ₁ ‖β‖₁  +  λ₂ ‖β‖²
prox_{t·g}(v) = soft_threshold(v, t λ₁) / (1 + 2t λ₂)
```

The L1 and L2 proximal operators compose exactly in this separable form because
the L2 part is strictly smooth and the L1 part acts element-wise independently.

---

## Duality Gap Certificates (`duality_gap.py`)

The Fenchel duality gap provides a computable, exact certificate of sub-optimality:

```
gap(β) = P(β) − D(θ(β))  ≥  F(β) − F(β*)  ≥  0
```

`gap = 0` if and only if `β = β*` (strong duality holds for both objectives).

### Lasso Duality Gap

**Primal**: `P(β) = (1/2n) ‖y − Xβ‖²  +  λ ‖β‖₁`

**Dual variable** (project residual `r = y − Xβ` to feasibility):
```
θ = r / (n · max(1, ‖Xᵀr‖∞ / (nλ)))
```
This ensures the dual constraint `‖Xᵀθ‖∞ ≤ λ`.

**Dual objective**:
```
D(θ) = (1/2n) ‖y‖² − (1/2n) ‖y − nθ‖²
```

### Elastic Net Duality Gap

**Primal**: `P(β) = (1/2n) ‖y − Xβ‖²  +  λ₁ ‖β‖₁  +  λ₂ ‖β‖²`

Uses Fenchel–Rockafellar duality.  Conjugate of the regularizer (for λ₂ > 0):
```
g*(v) = Σ_j  max(|v_j| − λ₁, 0)² / (4λ₂)
```

**Dual variable** (no projection needed since `g*` is everywhere finite):
```
θ = r / n
```

**Dual objective**:
```
D(θ) = yᵀθ − (n/2) ‖θ‖²  −  g*(Xᵀθ)
```

**Correctness at optimum**: KKT gives `Xᵀr*/n = λ₁ s* + 2λ₂ β*`, so
`g*(Xᵀθ*) = λ₂ ‖β*‖²` and `P(β*) − D(θ*) = 0` (verified by substitution).

---

## Algorithms

### ISTA (O(1/k) convergence)

Standard proximal gradient descent:

```
1.  grad  = −(1/n) Xᵀ(y − Xβ_k)
2.  v     = β_k − (1/L) · grad
3.  β_{k+1} = prox_{(1/L)·λΩ}(v)
```

### FISTA (O(1/k²) convergence)

Nesterov momentum acceleration applied to ISTA:

```
t_{k+1}  = (1 + √(1 + 4 t_k²)) / 2
y_k      = β_k + ((t_k − 1) / t_{k+1}) · (β_k − β_{k−1})
grad     = −(1/n) Xᵀ(y − X y_k)
β_{k+1}  = prox_{(1/L)·λΩ}(y_k − (1/L) · grad)
```

The extrapolated point `y_k` is used for both the gradient and the proximal
step.  The momentum coefficient `(t_k − 1) / t_{k+1}` grows monotonically
toward 1, driving progressively stronger extrapolation as iterates converge.

### Backtracking Line Search (`line_search.py`)

Avoids the cost of computing `σ_max(XᵀX)` at each iteration by starting from
the previous `L` and multiplying by `η` (default `η = 1.5`) until the
sufficient-descent condition is satisfied:

```
f(β_new) ≤ f(β) + ∇f(β)ᵀ(β_new − β) + (L/2) ‖β_new − β‖²
```

where `f` denotes the smooth data-fit term only; the proximal step handles the
regularization.