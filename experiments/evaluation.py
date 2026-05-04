"""
Evaluation utilities for regularized regression experiments.

Provides a self-contained suite of functions for assessing solver quality on
both synthetic and real-world data:

- **Cross-validation** (`cv_score`, `lambda_cv_search`): walk-forward (expanding
  window) CV that respects the temporal ordering of supply-chain records,
  preventing future-data leakage during hyperparameter selection.

- **Regularization path** (`regularization_path`): coefficient trajectories
  across a data-driven λ grid, enabling visual inspection of variable selection.

- **Bias–variance decomposition** (`bias_variance_sweep`): train/test RMSE as a
  function of regularization strength, isolating underfitting from overfitting.

- **Model comparison** (`model_comparison_table`): tabular cross-model benchmarks
  using a unified evaluation protocol.

- **Solver timing** (`time_solver`): wall-clock comparison of ISTA vs FISTA to
  complement the theoretical convergence analysis.

- **Plot helpers**: publication-ready matplotlib figures for all of the above.

All standardization is fitted exclusively on training folds; test folds are
transformed using training-fold statistics to prevent information leakage.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from solvers import FISTA, ISTA
from experiments.data_pipeline import standardize, time_series_cv_splits


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cv_score(
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    lam: float,
    lam2: float = 0.0,
    n_splits: int = 5,
    solver_cls=FISTA,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Time-series walk-forward CV for a single (model, lam) configuration.

    Standardization is fitted on each training fold independently to prevent
    leakage.

    Returns
    -------
    dict with keys: rmse_mean, rmse_std, mae_mean, mae_std, n_iter_mean
    """
    splits = time_series_cv_splits(len(y), n_splits=n_splits)
    rmse_scores, mae_scores, iter_counts = [], [], []

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr_std, X_te_std, _, _ = standardize(X_tr, X_te)
        y_mean = float(y_tr.mean())
        y_tr_c = y_tr - y_mean

        solver = solver_cls(
            model=model,
            lam=lam,
            lam2=lam2,
            max_iter=max_iter,
            tol=tol,
            line_search=True,
        )
        solver.fit(X_tr_std, y_tr_c)
        y_pred = X_te_std @ solver.coef_ + y_mean

        rmse_scores.append(rmse(y_te, y_pred))
        mae_scores.append(mae(y_te, y_pred))
        iter_counts.append(solver.n_iter_)

    return {
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "n_iter_mean": float(np.mean(iter_counts)),
    }


# ---------------------------------------------------------------------------
# Regularization path
# ---------------------------------------------------------------------------

def regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    lam_grid: np.ndarray,
    lam2_ratio: float = 0.5,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coefficient path across lambda values (full data fit).

    Iterates from largest to smallest lambda (cold-start each time).
    For ElasticNet: lam2 = lam * lam2_ratio (fixed ratio across path).

    Returns
    -------
    coef_path : np.ndarray, shape (n_lam, p)
    lam_grid  : np.ndarray, shape (n_lam,)  — same input grid
    """
    p = X.shape[1]
    coef_path = np.zeros((len(lam_grid), p))
    y_c = y - float(y.mean())   # center y (non-penalized intercept)

    for i, lam in enumerate(lam_grid):
        lam2 = lam * lam2_ratio if model == "elasticnet" else 0.0
        solver = FISTA(
            model=model,
            lam=lam,
            lam2=lam2,
            max_iter=max_iter,
            tol=tol,
            line_search=True,
        )
        solver.fit(X, y_c)
        coef_path[i] = solver.coef_

    return coef_path, lam_grid


# ---------------------------------------------------------------------------
# Lambda grid search via CV
# ---------------------------------------------------------------------------

def lambda_cv_search(
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    lam_grid: np.ndarray,
    lam2_ratio: float = 0.5,
    n_splits: int = 5,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> Tuple[float, pd.DataFrame]:
    """
    Grid search over lam_grid using time-series CV.

    Returns
    -------
    best_lam : float
    results_df : pd.DataFrame with columns [lambda, rmse_mean, rmse_std, mae_mean, n_iter_mean]
    """
    rows = []
    for lam in lam_grid:
        lam2 = lam * lam2_ratio if model == "elasticnet" else 0.0
        scores = cv_score(
            X, y, model=model, lam=lam, lam2=lam2,
            n_splits=n_splits, max_iter=max_iter, tol=tol,
        )
        rows.append({"lambda": lam, **scores})

    results_df = pd.DataFrame(rows)
    best_idx = results_df["rmse_mean"].idxmin()
    best_lam = float(results_df.loc[best_idx, "lambda"])
    return best_lam, results_df


# ---------------------------------------------------------------------------
# Bias-variance sweep
# ---------------------------------------------------------------------------

def bias_variance_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: str,
    lam_grid: np.ndarray,
    lam2_ratio: float = 0.5,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    For each lambda, fit on (X_train, y_train) and score on both splits.

    Both X_train and X_test must already be standardized (via standardize()).

    Returns
    -------
    pd.DataFrame with columns:
        lambda, train_rmse, test_rmse, n_nonzero, coef_norm_l1, coef_norm_l2
    """
    y_mean = float(y_train.mean())
    y_train_c = y_train - y_mean

    rows = []
    for lam in lam_grid:
        lam2 = lam * lam2_ratio if model == "elasticnet" else 0.0
        solver = FISTA(
            model=model,
            lam=lam,
            lam2=lam2,
            max_iter=max_iter,
            tol=tol,
            line_search=True,
        )
        solver.fit(X_train, y_train_c)
        beta = solver.coef_

        train_pred = X_train @ beta + y_mean
        test_pred = X_test @ beta + y_mean

        rows.append({
            "lambda": lam,
            "train_rmse": rmse(y_train, train_pred),
            "test_rmse": rmse(y_test, test_pred),
            "n_nonzero": int(np.sum(np.abs(beta) > 1e-6)),
            "coef_norm_l1": float(np.sum(np.abs(beta))),
            "coef_norm_l2": float(np.sqrt(np.dot(beta, beta))),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def model_comparison_table(
    X: np.ndarray,
    y: np.ndarray,
    configs: List[Dict],
    n_splits: int = 5,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Compare multiple (model, lam, lam2) configs via time-series CV.

    Parameters
    ----------
    configs : list of dicts, each with keys "model", "lam", "lam2", "label"

    Returns
    -------
    pd.DataFrame indexed by "label", sorted by rmse_mean ascending.
    """
    rows = []
    for cfg in configs:
        scores = cv_score(
            X, y,
            model=cfg["model"],
            lam=cfg["lam"],
            lam2=cfg.get("lam2", 0.0),
            n_splits=n_splits,
            max_iter=max_iter,
            tol=tol,
        )
        rows.append({"label": cfg["label"], **scores})

    df = pd.DataFrame(rows).set_index("label")
    return df.sort_values("rmse_mean")


# ---------------------------------------------------------------------------
# Solver timing
# ---------------------------------------------------------------------------

def time_solver(
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    lam: float,
    lam2: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    n_runs: int = 3,
) -> Dict[str, float]:
    """
    Time ISTA and FISTA on (X, y) and return comparison metrics.

    Runs n_runs times and averages to reduce noise.

    Returns
    -------
    dict: ista_time_sec, fista_time_sec, speedup_ratio,
          ista_n_iter, fista_n_iter, iter_ratio
    """
    ista_times, fista_times = [], []
    ista_iters, fista_iters = [], []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        s = ISTA(model=model, lam=lam, lam2=lam2, max_iter=max_iter, tol=tol, line_search=True)
        s.fit(X, y)
        ista_times.append(time.perf_counter() - t0)
        ista_iters.append(s.n_iter_)

        t0 = time.perf_counter()
        s = FISTA(model=model, lam=lam, lam2=lam2, max_iter=max_iter, tol=tol, line_search=True)
        s.fit(X, y)
        fista_times.append(time.perf_counter() - t0)
        fista_iters.append(s.n_iter_)

    ista_t = float(np.mean(ista_times))
    fista_t = float(np.mean(fista_times))
    ista_it = float(np.mean(ista_iters))
    fista_it = float(np.mean(fista_iters))

    return {
        "ista_time_sec": ista_t,
        "fista_time_sec": fista_t,
        "speedup_ratio": ista_t / (fista_t + 1e-12),
        "ista_n_iter": ista_it,
        "fista_n_iter": fista_it,
        "iter_ratio": ista_it / (fista_it + 1e-12),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_regularization_path(
    coef_path: np.ndarray,
    lam_grid: np.ndarray,
    feature_names: List[str],
    model: str,
    ax=None,
    top_k: int = 10,
) -> None:
    """
    Plot coefficient path vs log(lambda).

    Only labels the top_k features by maximum absolute coefficient value.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    max_abs = np.max(np.abs(coef_path), axis=0)
    top_idx = np.argsort(max_abs)[::-1][:top_k]

    for j in range(coef_path.shape[1]):
        if j in top_idx:
            label = feature_names[j] if j < len(feature_names) else f"f{j}"
            ax.semilogx(lam_grid, coef_path[:, j], lw=1.5, label=label)
        else:
            ax.semilogx(lam_grid, coef_path[:, j], color="lightgray", lw=0.7, alpha=0.5)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"{model.capitalize()} Regularization Path")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_bias_variance(
    bv_df: pd.DataFrame,
    model: str,
    best_lam: Optional[float] = None,
    ax=None,
) -> None:
    """
    Plot train RMSE and test RMSE vs log(lambda).

    A vertical dashed line is drawn at best_lam if provided.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.semilogx(bv_df["lambda"], bv_df["train_rmse"], "b-o", ms=3, label="Train RMSE")
    ax.semilogx(bv_df["lambda"], bv_df["test_rmse"], "r-o", ms=3, label="Test RMSE")

    if best_lam is not None:
        ax.axvline(best_lam, color="green", ls="--", lw=1.5, label=f"Best λ={best_lam:.4f}")

    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{model.capitalize()} — Bias-Variance Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_cv_curve(
    cv_df: pd.DataFrame,
    model: str,
    best_lam: Optional[float] = None,
    ax=None,
) -> None:
    """
    Plot CV RMSE mean ± 1 std vs log(lambda).

    A vertical dashed line is drawn at best_lam if provided.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    lam = cv_df["lambda"].values
    mu = cv_df["rmse_mean"].values
    sigma = cv_df["rmse_std"].values

    ax.semilogx(lam, mu, "b-o", ms=4, label="CV RMSE (mean)")
    ax.fill_between(lam, mu - sigma, mu + sigma, alpha=0.2, color="blue", label="±1 std")

    if best_lam is not None:
        ax.axvline(best_lam, color="green", ls="--", lw=1.5, label=f"Best λ={best_lam:.4f}")

    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("CV RMSE")
    ax.set_title(f"{model.capitalize()} — Lambda Tuning via Walk-forward CV")
    ax.legend()
    ax.grid(True, alpha=0.3)
