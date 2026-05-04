# Proximal Gradient Methods for Regularized Regression

A research code repository implementing and empirically evaluating ISTA and FISTA
— the canonical proximal gradient algorithms for convex composite minimization —
applied to Ridge, Lasso, and Elastic Net regression.

All solvers are written from scratch in NumPy.  The implementation covers the
complete algorithmic stack: proximal operators, backtracking line search, Fenchel
duality gap certificates, and walk-forward cross-validation on a real supply-chain
dataset.

---

## Problem Formulation

All solvers minimize the composite convex objective

```
F(β) = (1/2n) ‖y − Xβ‖²  +  λ Ω(β)
```

where

| Symbol | Meaning |
|--------|---------|
| `n`, `p` | number of observations, features |
| `X ∈ ℝⁿˣᵖ` | standardized design matrix |
| `y ∈ ℝⁿ` | response vector |
| `β ∈ ℝᵖ` | coefficient vector (decision variable) |
| `λ > 0` | regularization strength |
| `Ω(β)` | regularizer: `‖β‖²` (Ridge), `‖β‖₁` (Lasso), or their convex combination (Elastic Net) |

---

## Repository Structure

```
msml604-convex-regression/
│
├── solvers/                    Core solver library
│   ├── __init__.py             Public API
│   ├── proximal.py             Closed-form proximal operators
│   ├── ista.py                 ISTA solver (O(1/k) convergence)
│   ├── fista.py                FISTA solver (O(1/k²) convergence)
│   ├── line_search.py          Backtracking line search
│   ├── duality_gap.py          Fenchel duality gap certificates
│   └── README.md               Detailed algorithm documentation
│
├── experiments/                Experiment infrastructure
│   ├── synthetic.py            Synthetic benchmark data generators
│   ├── data_pipeline.py        DataCo supply-chain preprocessing pipeline
│   ├── evaluation.py           CV, regularization path, bias-variance, timing
│   ├── convergence.py          Convergence figure generator (run standalone)
│   └── plots/                  Output directory (git-ignored)
│
├── notebooks/
│   ├── 01_eda.ipynb            Exploratory data analysis of the supply-chain dataset
│   ├── 02_real_experiments.ipynb   Regression experiments on real data
│   └── 03_synthetic_experiments.ipynb  Algorithm benchmarks on synthetic data
│
├── data/
│   ├── DataCoSupplyChainDataset.csv   Raw dataset (see Data section below)
│   ├── X_processed.npy                Preprocessed design matrix
│   ├── y_processed.npy                Preprocessed response vector
│   └── feature_names.npy              Feature name array
│
├── tests/
│   └── test_all.py             Correctness and property validation suite
│
└── README.md                   This file
```

---

## Installation

### Prerequisites

- Python 3.9 or later
- pip

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd msml604-convex-regression

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install numpy scipy pandas matplotlib scikit-learn jupyter
```

No additional build steps are required — the solver library is pure Python/NumPy.

---

## Data

### DataCo Smart Supply Chain Dataset

The real-data experiments use the publicly available **DataCo Smart Supply Chain
for Big Data Analysis** dataset (Konstantinou, Kaggle).

**Download**: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

After downloading, place `DataCoSupplyChainDataset.csv` in the `data/` directory:

```
data/DataCoSupplyChainDataset.csv
```

The preprocessing pipeline (`experiments/data_pipeline.py`) handles all feature
engineering automatically.  Preprocessed arrays (`X_processed.npy`, etc.) are
included in the repository for convenience and allow the notebooks to run without
re-processing the raw CSV.

---

## Reproducing Results

### 1. Run the Test Suite

Validates solver correctness against scikit-learn and checks algorithmic properties
(duality gap monotonicity, sparsity recovery, FISTA iteration counts, etc.):

```bash
python tests/test_all.py
```

Expected output: all sklearn comparison tests pass with max coefficient deviation
< 0.01, and all additional property tests pass.

### 2. Generate Convergence Plots

Produces 9 convergence figures (3 datasets × 3 regularizers) comparing ISTA and
FISTA on synthetic benchmark problems.  Figures are saved to `experiments/plots/`.

```bash
python experiments/convergence.py
```

Each figure shows:
- Objective value `F(β_k)` vs iteration on a semi-log scale
- Fenchel duality gap vs iteration (Lasso / Elastic Net only)
- O(1/k) and O(1/k²) reference curves

### 3. Run the Jupyter Notebooks

Launch JupyterLab (or classic Notebook) and open the notebooks in order:

```bash
jupyter lab
```

| Notebook | Contents |
|----------|----------|
| `01_eda.ipynb` | Exploratory data analysis: feature distributions, target distribution, correlation structure, leakage detection |
| `02_real_experiments.ipynb` | Full pipeline on the DataCo dataset: preprocessing, lambda tuning via walk-forward CV, regularization paths, bias-variance curves, model comparison table, ISTA vs FISTA timing |
| `03_synthetic_experiments.ipynb` | Algorithm benchmarks on all three synthetic datasets: convergence curves, iteration counts, coefficient recovery |

All notebooks are self-contained and can be run cell-by-cell from top to bottom.

### 4. Use the Solver Library Directly

```python
import numpy as np
from solvers import ISTA, FISTA

# Simulated data
rng = np.random.default_rng(0)
n, p = 200, 50
X = rng.standard_normal((n, p))
X = (X - X.mean(0)) / X.std(0)
beta_true = rng.standard_normal(p)
y = X @ beta_true + 0.1 * rng.standard_normal(n)

# Fit Lasso with FISTA
solver = FISTA(model="lasso", lam=0.1, max_iter=1000, tol=1e-6, line_search=True)
solver.fit(X, y)

print(f"Converged in {solver.n_iter_} iterations")
print(f"Final duality gap: {solver.gap_history_[-1]:.2e}")
print(f"Non-zero coefficients: {(abs(solver.coef_) > 1e-6).sum()}")

# Fit Elastic Net with ISTA
solver_en = ISTA(model="elasticnet", lam=0.1, lam2=0.05,
                 max_iter=2000, tol=1e-6, line_search=True)
solver_en.fit(X, y)
```

---

## Algorithm Overview

### ISTA — Iterative Shrinkage-Thresholding Algorithm

Applies proximal gradient descent with step size `1/L`:

```
β_{k+1} = prox_{(1/L)·λΩ}( β_k − (1/L) ∇f(β_k) )
```

Convergence rate: **O(1/k)** in the objective gap `F(β_k) − F(β*)`.

### FISTA — Fast ISTA (Nesterov Acceleration)

Introduces a momentum extrapolation step before each proximal update:

```
t_{k+1} = (1 + √(1 + 4 t_k²)) / 2
y_k      = β_k + ((t_k − 1) / t_{k+1}) · (β_k − β_{k−1})
β_{k+1} = prox_{(1/L)·λΩ}( y_k − (1/L) ∇f(y_k) )
```

Convergence rate: **O(1/k²)** in the objective gap — the optimal rate for
first-order methods on smooth + non-smooth composite problems (Nesterov, 1983).

### Convergence Certificates

For Lasso and Elastic Net, the **Fenchel duality gap** provides a rigorous,
computable bound on suboptimality at each iteration, enabling principled early
stopping without knowledge of the optimal value.  Ridge convergence is monitored
via the squared full-gradient norm.

---

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| 1/(2n) data-fit scaling | Keeps λ interpretation invariant to sample size |
| Backtracking line search | Avoids expensive eigenvalue computation at each step |
| Walk-forward CV | Respects temporal ordering; prevents future-data leakage |
| Duality-gap stopping | Certified optimality without knowledge of F(β*) |
| FISTA inherits ISTA | Single code path for all proximal/gap/objective logic |

---

## References

Beck, A. and Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm
  for Linear Inverse Problems. *SIAM Journal on Imaging Sciences*, **2**(1), 183–202.

Nesterov, Y. (1983). A method of solving a convex programming problem with convergence
  rate O(1/k²). *Soviet Mathematics Doklady*, **27**(2), 372–376.

Parikh, N. and Boyd, S. (2014). Proximal Algorithms. *Foundations and Trends in
  Optimization*, **1**(3), 127–239.

Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of
  the Royal Statistical Society: Series B*, **58**(1), 267–288.

Zou, H. and Hastie, T. (2005). Regularization and variable selection via the Elastic
  Net. *Journal of the Royal Statistical Society: Series B*, **67**(2), 301–320.
