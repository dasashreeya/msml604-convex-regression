"""
Convergence plots comparing ISTA vs FISTA on synthetic datasets.

Generates 9 plots (3 datasets × 3 models):
  - Objective value vs iteration (log scale)
  - Duality gap vs iteration (log scale, Lasso/ElasticNet only)
  - Reference lines: O(1/k) and O(1/k²)

Output: experiments/plots/convergence_{model}_{dataset}.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solvers import ISTA, FISTA
from experiments.synthetic import high_correlation, high_dimensional, near_singular


DATASETS = {
    "high_corr": high_correlation,
    "high_dim": high_dimensional,
    "near_singular": near_singular,
}

MODELS = ["ridge", "lasso", "elasticnet"]
LAM = 0.1
LAM2 = 0.05  # elastic net L2 param


def _ref_lines(k_max: int, v0: float):
    """Return iteration array and O(1/k), O(1/k²) reference curves."""
    ks = np.arange(1, k_max + 1, dtype=float)
    return ks, v0 / ks, v0 / ks ** 2


def plot_convergence(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    model: str = "lasso",
    dataset_name: str = "dataset",
    out_dir: str = None,
) -> None:
    """
    Run ISTA and FISTA on (X, y) and save convergence plots.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,)
    lam : float
        Regularization parameter.
    model : str
        One of 'ridge', 'lasso', 'elasticnet'.
    dataset_name : str
        Used in the output filename.
    out_dir : str
        Directory for saving plots. Defaults to experiments/plots/.
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    lam2 = LAM2 if model == "elasticnet" else 0.0

    ista = ISTA(model=model, lam=lam, lam2=lam2, max_iter=500, tol=1e-10,
                line_search=True)
    fista = FISTA(model=model, lam=lam, lam2=lam2, max_iter=500, tol=1e-10,
                  line_search=True)

    ista.fit(X, y)
    fista.fit(X, y)

    has_gap = model in ("lasso", "elasticnet")
    n_plots = 2 if has_gap else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # --- Objective plot ---
    ax = axes[0]
    iters_i = np.arange(1, len(ista.loss_history_) + 1)
    iters_f = np.arange(1, len(fista.loss_history_) + 1)

    ax.semilogy(iters_i, ista.loss_history_, label="ISTA", color="tab:blue")
    ax.semilogy(iters_f, fista.loss_history_, label="FISTA", color="tab:orange")

    # Reference lines anchored to ISTA's first value
    k_max = max(len(ista.loss_history_), len(fista.loss_history_))
    ks, ref1, ref2 = _ref_lines(k_max, ista.loss_history_[0])
    ax.semilogy(ks, ref1, "k--", alpha=0.4, label="O(1/k)")
    ax.semilogy(ks, ref2, "k:", alpha=0.4, label="O(1/k²)")

    ax.set_xlabel("Iteration k")
    ax.set_ylabel("Objective F(β)")
    ax.set_title(f"{model.capitalize()} — {dataset_name}\nObjective vs Iteration")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # --- Duality gap plot ---
    if has_gap:
        ax2 = axes[1]
        gap_i = [g for g in ista.gap_history_ if not np.isnan(g)]
        gap_f = [g for g in fista.gap_history_ if not np.isnan(g)]

        # Clip near-zero for log scale
        gap_i = np.maximum(gap_i, 1e-16)
        gap_f = np.maximum(gap_f, 1e-16)

        ax2.semilogy(np.arange(1, len(gap_i) + 1), gap_i,
                     label="ISTA gap", color="tab:blue")
        ax2.semilogy(np.arange(1, len(gap_f) + 1), gap_f,
                     label="FISTA gap", color="tab:orange")

        k_max_g = max(len(gap_i), len(gap_f))
        ks_g, ref1_g, ref2_g = _ref_lines(k_max_g, gap_i[0])
        ax2.semilogy(ks_g, ref1_g, "k--", alpha=0.4, label="O(1/k)")
        ax2.semilogy(ks_g, ref2_g, "k:", alpha=0.4, label="O(1/k²)")

        ax2.set_xlabel("Iteration k")
        ax2.set_ylabel("Duality Gap")
        ax2.set_title(f"{model.capitalize()} — {dataset_name}\nDuality Gap vs Iteration")
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fname = f"convergence_{model}_{dataset_name}.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    print(f"  Saved: {fpath}")


def run_all() -> None:
    """Generate all 9 convergence plots across 3 datasets × 3 models."""
    dataset_fns = {
        "high_corr": high_correlation,
        "high_dim": high_dimensional,
        "near_singular": near_singular,
    }

    for ds_name, ds_fn in dataset_fns.items():
        X, y, _ = ds_fn()
        for model in MODELS:
            print(f"Plotting: {model} on {ds_name} ...")
            plot_convergence(X, y, lam=LAM, model=model, dataset_name=ds_name)


if __name__ == "__main__":
    run_all()
