"""
Streamlit dashboard for interactive regularized regression exploration.

Usage:
    streamlit run dashboard.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from solvers import FISTA, ISTA
from experiments.synthetic import high_correlation, high_dimensional, near_singular
from experiments.data_pipeline import standardize

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Convex Regression Explorer",
    page_icon="📉",
    layout="wide",
)

st.title("Ridge, Lasso & Elastic Net — Convex Optimization Explorer")
st.caption(
    "Pick a dataset, choose a model, and drag λ to watch coefficients "
    "and convergence update live."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Controls")

DATASET_OPTIONS = {
    "DataCo Supply Chain (real)": "dataco",
    "Synthetic — High Correlation (ρ=0.95)": "high_corr",
    "Synthetic — High Dimensional (p=500, s=10)": "high_dim",
    "Synthetic — Near Singular (κ=10⁶)": "near_singular",
}

MODEL_OPTIONS = ["ridge", "lasso", "elasticnet"]
MODEL_LABELS = {"ridge": "Ridge (L2)", "lasso": "Lasso (L1)", "elasticnet": "Elastic Net (L1+L2)"}

dataset_label = st.sidebar.selectbox("Dataset", list(DATASET_OPTIONS.keys()))
dataset_key = DATASET_OPTIONS[dataset_label]

model = st.sidebar.selectbox(
    "Model",
    MODEL_OPTIONS,
    format_func=lambda m: MODEL_LABELS[m],
)

log_lam = st.sidebar.slider(
    "log₁₀(λ)  — regularization strength",
    min_value=-3.0, max_value=1.0, value=-1.0, step=0.05,
)
lam = float(10 ** log_lam)
st.sidebar.caption(f"λ = {lam:.5f}")

if model == "elasticnet":
    lam2_ratio = st.sidebar.slider("Elastic Net α (L1 ratio)", 0.0, 1.0, 0.5, 0.05)
    lam2 = lam * lam2_ratio
else:
    lam2 = 0.0

solver_choice = st.sidebar.radio("Solver", ["FISTA", "ISTA", "Both"])

max_iter = st.sidebar.slider("Max iterations", 50, 2000, 500, 50)
tol = st.sidebar.select_slider(
    "Convergence tolerance",
    options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    value=1e-6,
)

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Top-k coefficients to label", 5, 20, 10)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading DataCo dataset…")
def load_dataco():
    data_dir = os.path.join(ROOT, "data")
    X = np.load(os.path.join(data_dir, "X_processed.npy"))
    y = np.load(os.path.join(data_dir, "y_processed.npy"))
    feat = list(np.load(os.path.join(data_dir, "feature_names.npy"), allow_pickle=True))
    # chronological 80/20 split, standardize on train
    n_train = int(0.8 * len(y))
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    X_tr_std, X_te_std, _, _ = standardize(X_tr, X_te)
    return X_tr_std, y_tr, X_te_std, y_te, feat


@st.cache_data(show_spinner="Generating synthetic dataset…")
def load_synthetic(key):
    if key == "high_corr":
        X, y, beta_true = high_correlation(n=300, p=50, rho=0.95, seed=42)
    elif key == "high_dim":
        X, y, beta_true = high_dimensional(n=100, p=500, sparsity=10, seed=42)
    else:
        X, y, beta_true = near_singular(n=200, p=50, condition_number=1e6, seed=42)
    feat = [f"x{i}" for i in range(X.shape[1])]
    n_train = int(0.8 * len(y))
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    return X_tr, y_tr, X_te, y_te, feat, beta_true


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

beta_true = None
if dataset_key == "dataco":
    X_train, y_train, X_test, y_test, feature_names = load_dataco()
else:
    X_train, y_train, X_test, y_test, feature_names, beta_true = load_synthetic(dataset_key)

p = X_train.shape[1]
y_mean = float(y_train.mean())
y_train_c = y_train - y_mean

# ---------------------------------------------------------------------------
# Fit solver(s)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fit_solver(cls_name, model, lam, lam2, max_iter, tol, X_tr_bytes, y_tr_bytes):
    X_tr = np.frombuffer(X_tr_bytes, dtype=np.float64).reshape(-1, X_train.shape[1])
    y_tr = np.frombuffer(y_tr_bytes, dtype=np.float64)
    cls = FISTA if cls_name == "FISTA" else ISTA
    solver = cls(model=model, lam=lam, lam2=lam2, max_iter=max_iter, tol=tol, line_search=True)
    solver.fit(X_tr, y_tr)
    return solver.coef_, solver.loss_history_, solver.gap_history_, solver.n_iter_


X_bytes = X_train.astype(np.float64).tobytes()
y_bytes = y_train_c.astype(np.float64).tobytes()

solvers_to_run = (
    ["FISTA", "ISTA"] if solver_choice == "Both"
    else [solver_choice]
)

results = {}
for name in solvers_to_run:
    coef, loss_hist, gap_hist, n_iter = fit_solver(
        name, model, lam, lam2, max_iter, tol, X_bytes, y_bytes
    )
    results[name] = dict(coef=coef, loss=loss_hist, gap=gap_hist, n_iter=n_iter)

# Use primary solver for metrics
primary = results[solvers_to_run[0]]
beta_hat = primary["coef"]
y_pred_test = X_test @ beta_hat + y_mean
y_pred_train = X_train @ beta_hat + y_mean

rmse_train = float(np.sqrt(np.mean((y_train - y_pred_train) ** 2)))
rmse_test = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
n_nonzero = int(np.sum(np.abs(beta_hat) > 1e-6))
ss_res = np.sum((y_test - y_pred_test) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2_test = float(1.0 - ss_res / (ss_tot + 1e-12))

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Test RMSE", f"{rmse_test:.3f}")
col2.metric("Train RMSE", f"{rmse_train:.3f}")
col3.metric("Test R²", f"{r2_test:.4f}")
col4.metric("Nonzero coefs", f"{n_nonzero} / {p}")
col5.metric("Iterations", primary["n_iter"])

st.markdown("---")

# ---------------------------------------------------------------------------
# Main plots — two columns
# ---------------------------------------------------------------------------

left, right = st.columns(2)

# ---- Coefficient bar chart ----
with left:
    st.subheader("Coefficient Vector")

    fig_coef, ax = plt.subplots(figsize=(7, 4))
    fig_coef.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Show top_k by magnitude; rest as grey
    order = np.argsort(np.abs(beta_hat))[::-1]
    colors = ["#4c9be8" if i < top_k else "#3a3a4a" for i in range(len(order))]
    sorted_colors = [colors[rank] for rank in np.argsort(order)]
    ax.bar(range(p), beta_hat, color=sorted_colors, width=1.0)

    # Label top_k
    for rank, idx in enumerate(order[:top_k]):
        label = feature_names[idx] if idx < len(feature_names) else f"x{idx}"
        ax.annotate(
            label,
            xy=(idx, beta_hat[idx]),
            xytext=(0, 4 if beta_hat[idx] >= 0 else -10),
            textcoords="offset points",
            fontsize=6,
            color="white",
            ha="center",
            rotation=45,
        )

    if beta_true is not None:
        ax.step(range(p), beta_true, color="#f4a261", lw=1.2, label="β_true", where="mid")
        ax.legend(fontsize=8, facecolor="#1e1e2e", labelcolor="white")

    ax.axhline(0, color="white", lw=0.5, alpha=0.4)
    ax.set_xlabel("Feature index", color="white", fontsize=9)
    ax.set_ylabel("Coefficient", color="white", fontsize=9)
    ax.set_title(
        f"{MODEL_LABELS[model]}  |  λ={lam}  |  {n_nonzero}/{p} nonzero",
        color="white", fontsize=10,
    )
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    st.pyplot(fig_coef)
    plt.close(fig_coef)

# ---- Convergence curve ----
with right:
    st.subheader("Convergence Curve")

    fig_conv, axes = plt.subplots(1, 2, figsize=(7, 4))
    fig_conv.patch.set_facecolor("#0e1117")

    COLORS = {"FISTA": "#4c9be8", "ISTA": "#f4a261"}

    for name, res in results.items():
        loss = res["loss"]
        gap = [g for g in res["gap"] if np.isfinite(g)]

        # Objective
        ax_obj = axes[0]
        ax_obj.set_facecolor("#0e1117")
        if loss:
            f_star = min(loss)
            excess = [max(v - f_star, 1e-16) for v in loss]
            ax_obj.semilogy(excess, color=COLORS[name], lw=1.5, label=name)

        # Duality gap
        ax_gap = axes[1]
        ax_gap.set_facecolor("#0e1117")
        if gap:
            ax_gap.semilogy(gap, color=COLORS[name], lw=1.5, label=name)

    # Reference lines O(1/k) and O(1/k²)
    if results:
        k_ref = np.arange(1, len(results[solvers_to_run[0]]["loss"]) + 1)
        c0 = axes[0].get_lines()[0].get_ydata()[0] if axes[0].get_lines() else 1.0
        axes[0].semilogy(k_ref, c0 / k_ref, "w--", lw=0.7, alpha=0.4, label="O(1/k)")
        axes[0].semilogy(k_ref, c0 / k_ref**2, "w:", lw=0.7, alpha=0.4, label="O(1/k²)")

    for ax, title, ylabel in [
        (axes[0], "Objective − F*", "F(β_k) − F*"),
        (axes[1], "Duality Gap", "Gap G_k"),
    ]:
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("Iteration k", color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        ax.legend(fontsize=8, facecolor="#1e1e2e", labelcolor="white")
        ax.grid(True, alpha=0.15, color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    plt.tight_layout()
    st.pyplot(fig_conv)
    plt.close(fig_conv)

# ---------------------------------------------------------------------------
# Lambda path (secondary section, collapsible)
# ---------------------------------------------------------------------------

with st.expander("Regularization Path across λ grid", expanded=False):
    st.caption(
        "Coefficients fitted on the full training set for each λ in the grid. "
        "The vertical dashed line marks the currently selected λ."
    )

    @st.cache_data(show_spinner="Computing regularization path…")
    def compute_path(model, lam2_ratio, max_iter, tol, X_bytes, y_bytes, p_):
        X_tr = np.frombuffer(X_bytes, dtype=np.float64).reshape(-1, p_)
        y_tr = np.frombuffer(y_bytes, dtype=np.float64)
        lam_max = float(np.max(np.abs(X_tr.T @ y_tr))) / X_tr.shape[0]
        lam_grid = np.logspace(np.log10(lam_max), np.log10(1e-3 * lam_max), 40)
        path = np.zeros((len(lam_grid), p_))
        for i, lv in enumerate(lam_grid):
            l2 = lv * lam2_ratio if model == "elasticnet" else 0.0
            s = FISTA(model=model, lam=lv, lam2=l2, max_iter=max_iter, tol=tol, line_search=True)
            s.fit(X_tr, y_tr)
            path[i] = s.coef_
        return path, lam_grid

    path, lam_grid = compute_path(model, lam2_ratio if model == "elasticnet" else 0.5,
                                  max_iter, tol, X_bytes, y_bytes, p)

    fig_path, ax_path = plt.subplots(figsize=(12, 4))
    fig_path.patch.set_facecolor("#0e1117")
    ax_path.set_facecolor("#0e1117")

    max_abs = np.max(np.abs(path), axis=0)
    top_idx = set(np.argsort(max_abs)[::-1][:top_k])
    cmap = plt.cm.get_cmap("tab10", top_k)

    color_counter = 0
    for j in range(p):
        if j in top_idx:
            label = feature_names[j] if j < len(feature_names) else f"x{j}"
            ax_path.semilogx(lam_grid, path[:, j], lw=1.5,
                             color=cmap(color_counter), label=label)
            color_counter += 1
        else:
            ax_path.semilogx(lam_grid, path[:, j], color="#3a3a4a", lw=0.6, alpha=0.5)

    ax_path.axvline(lam, color="#f4a261", ls="--", lw=1.5, label=f"Current λ={lam}")
    ax_path.axhline(0, color="white", lw=0.5, alpha=0.3)
    ax_path.set_xlabel("λ (log scale)", color="white", fontsize=9)
    ax_path.set_ylabel("Coefficient", color="white", fontsize=9)
    ax_path.set_title(f"{MODEL_LABELS[model]} — Regularization Path", color="white", fontsize=10)
    ax_path.legend(loc="upper left", fontsize=7, ncol=3,
                   facecolor="#1e1e2e", labelcolor="white")
    ax_path.tick_params(colors="white", labelsize=7)
    ax_path.grid(True, alpha=0.15, color="white")
    for spine in ax_path.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    st.pyplot(fig_path)
    plt.close(fig_path)

# ---------------------------------------------------------------------------
# Dataset info footer
# ---------------------------------------------------------------------------

st.markdown("---")
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.caption(
        f"**Dataset**: {dataset_label}  |  "
        f"n_train={X_train.shape[0]:,}  n_test={X_test.shape[0]:,}  p={p}"
    )
with info_col2:
    st.caption(
        f"**Model**: {MODEL_LABELS[model]}  |  λ={lam}"
        + (f"  λ₂={lam2:.5f}" if model == "elasticnet" else "")
        + f"  |  solver={solver_choice}"
    )
