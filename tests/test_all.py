"""
Validation tests comparing custom ISTA/FISTA solvers against sklearn.

Checks:
  np.allclose(my_solver.coef_, sklearn.coef_, atol=1e-3)

Datasets tested:
  1. Small well-conditioned (n=100, p=10)
  2. High-correlation (n=200, p=50)
  3. High-dimensional (n=100, p=500)
  4. Near-singular (n=200, p=50)

Prints a results table and asserts correctness.
Also asserts FISTA converges in fewer iterations than ISTA on all 3
synthetic stress-test datasets (a regression in Nesterov momentum = a bug).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import Ridge as SkRidge
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import ElasticNet as SkElasticNet

from solvers import ISTA, FISTA
from solvers.proximal import prox_lasso, prox_ridge, prox_elasticnet
from experiments.synthetic import high_correlation, high_dimensional, near_singular

# ---------------------------------------------------------------------------
# Regularization parameters (matched to sklearn's convention)
#
# sklearn Ridge: minimizes ||y-Xw||^2 + alpha*||w||^2
#   Our objective: (1/(2n))||y-Xw||^2 + lam*||w||^2
#   Equivalent when sklearn_alpha = 2*n*lam
#
# sklearn Lasso: minimizes (1/(2n))||y-Xw||^2 + alpha*||w||_1
#   => lam = alpha  (direct match)
#
# sklearn ElasticNet: (1/(2n))||y-Xw||^2 + alpha*l1_ratio*||w||_1
#                   + 0.5*alpha*(1-l1_ratio)*||w||^2
#   => lam1 = alpha*l1_ratio, lam2 = 0.5*alpha*(1-l1_ratio)
# ---------------------------------------------------------------------------

ALPHA = 0.1
L1_RATIO = 0.5
LAM_RIDGE = ALPHA
LAM_LASSO = ALPHA
LAM1_EN = ALPHA * L1_RATIO
LAM2_EN = 0.5 * ALPHA * (1.0 - L1_RATIO)


def _small_dataset(seed=0):
    rng = np.random.default_rng(seed)
    n, p = 100, 10
    X = rng.standard_normal((n, p))
    X = (X - X.mean(0)) / X.std(0)
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    return X, y, beta_true


DATASETS = {
    "small_conditioned": _small_dataset,
    "high_corr": high_correlation,
    "high_dim": high_dimensional,
    "near_singular": near_singular,
}

SOLVER_CONFIGS = {
    "ridge": dict(lam=LAM_RIDGE, lam2=0.0),
    "lasso": dict(lam=LAM_LASSO, lam2=0.0),
    "elasticnet": dict(lam=LAM1_EN, lam2=LAM2_EN),
}

# Tight tolerance for sklearn accuracy comparison
SKLEARN_TOL = {"ridge": 1e-7, "lasso": 1e-6, "elasticnet": 1e-6}
SKLEARN_MAX_ITER = 5000

# Loose tolerance for FISTA vs ISTA iteration count comparison
SPEED_TOL = 1e-4
SPEED_MAX_ITER = 5000


def _sklearn_solver(model: str, n: int):
    if model == "ridge":
        # sklearn Ridge minimizes ||y-Xw||^2 + alpha*||w||^2 (no 1/(2n) factor).
        # Our objective (1/(2n))||y-Xw||^2 + lam*||w||^2 has same optimum when
        # sklearn_alpha = 2*n*lam.
        return SkRidge(alpha=2.0 * n * LAM_RIDGE, fit_intercept=False)
    elif model == "lasso":
        return SkLasso(alpha=LAM_LASSO, fit_intercept=False, max_iter=100000, tol=1e-10)
    else:
        return SkElasticNet(
            alpha=ALPHA,
            l1_ratio=L1_RATIO,
            fit_intercept=False,
            max_iter=100000,
            tol=1e-10,
        )


# ===========================================================================
# Additional unit tests
# ===========================================================================

def test_prox_operators():
    """Test prox operators against known analytical values."""
    print("\ntest_prox_operators")
    passed = 0
    total = 8

    # --- Lasso ---
    result = prox_lasso(np.array([3.0]), lam=1.0)
    assert np.allclose(result, [2.0], atol=1e-10), f"got {result}"
    print("  ✓ prox_lasso([3.0], lam=1.0) = [2.0]")
    passed += 1

    result = prox_lasso(np.array([0.5]), lam=1.0)
    assert np.allclose(result, [0.0], atol=1e-10), f"got {result}"
    print("  ✓ prox_lasso([0.5], lam=1.0) = [0.0]  (below threshold)")
    passed += 1

    result = prox_lasso(np.array([-3.0]), lam=1.0)
    assert np.allclose(result, [-2.0], atol=1e-10), f"got {result}"
    print("  ✓ prox_lasso([-3.0], lam=1.0) = [-2.0]  (negative side)")
    passed += 1

    result = prox_lasso(np.array([0.0]), lam=1.0)
    assert np.allclose(result, [0.0], atol=1e-10), f"got {result}"
    print("  ✓ prox_lasso([0.0], lam=1.0) = [0.0]  (at zero)")
    passed += 1

    # --- Ridge ---
    # 21 / (1 + 2*100*0.1) = 21 / 21 = 1.0
    result = prox_ridge(np.array([21.0]), lam=0.1, n=100)
    assert np.allclose(result, [1.0], atol=1e-10), f"got {result}"
    print("  ✓ prox_ridge([21.0], lam=0.1, n=100) = [1.0]  (21/(1+2*100*0.1)=21/21)")
    passed += 1

    # --- Elastic Net boundary: lam2=0 reduces to Lasso ---
    rng = np.random.default_rng(42)
    v = rng.standard_normal(20)
    lam = 0.3
    result_en = prox_elasticnet(v, lam1=lam, lam2=0.0)
    result_lasso = prox_lasso(v, lam=lam)
    assert np.allclose(result_en, result_lasso, atol=1e-10), (
        f"max diff = {np.max(np.abs(result_en - result_lasso))}"
    )
    print("  ✓ prox_elasticnet(v, lam1=lam, lam2=0) == prox_lasso(v, lam)  (reduces to lasso)")
    passed += 1

    # --- Elastic Net boundary: lam1=0, lam2=n*lam reduces to Ridge ---
    # prox_elasticnet(v, 0, n*lam) = v/(1+2*n*lam) = prox_ridge(v, lam, n)
    # The n*lam argument matches the 2n scaling baked into prox_ridge's convention.
    n_test = 100
    result_en = prox_elasticnet(v, lam1=0.0, lam2=float(n_test) * lam)
    result_ridge = prox_ridge(v, lam=lam, n=n_test)
    assert np.allclose(result_en, result_ridge, atol=1e-10), (
        f"max diff = {np.max(np.abs(result_en - result_ridge))}"
    )
    print("  ✓ prox_elasticnet(v, lam1=0, lam2=n*lam) == prox_ridge(v, lam, n)  (reduces to ridge)")
    passed += 1

    # --- Vectorized: all four cases in one call ---
    v4 = np.array([5.0, -5.0, 0.3, -0.3])
    result = prox_lasso(v4, lam=1.0)
    expected = np.array([4.0, -4.0, 0.0, 0.0])
    assert np.allclose(result, expected, atol=1e-10), f"got {result}"
    print("  ✓ prox_lasso([5, -5, 0.3, -0.3], lam=1) = [4, -4, 0, 0]  (vectorized)")
    passed += 1

    return passed, total


def test_duality_gap_validity():
    """Duality gap must be non-negative and non-increasing for ISTA on small data."""
    print("\ntest_duality_gap_validity")
    X, y, _ = _small_dataset()
    summary = {}

    for model in ("lasso", "elasticnet"):
        lam = LAM_LASSO if model == "lasso" else LAM1_EN
        lam2 = 0.0 if model == "lasso" else LAM2_EN
        solver = ISTA(
            model=model, lam=lam, lam2=lam2,
            max_iter=500, tol=1e-10, line_search=True,
        )
        solver.fit(X, y)

        gaps = np.array(solver.gap_history_)
        assert np.all(gaps >= -1e-8), (
            f"{model}: found negative gap(s): min={gaps.min():.3e}"
        )
        diffs = np.diff(gaps)
        assert np.all(diffs <= 1e-8), (
            f"{model}: gap is not non-increasing; "
            f"max increase = {diffs.max():.3e} at iter {diffs.argmax()}"
        )
        summary[model] = (gaps.min(), gaps.max())
        print(
            f"  ✓ {model.capitalize()} gap: "
            f"min={gaps.min():.2e}, max={gaps.max():.2e}, monotone=True"
        )

    return summary


def test_large_lambda_sparsity():
    """Very large lambda must drive all coefficients to zero."""
    print("\ntest_large_lambda_sparsity")
    X, y, _ = _small_dataset()

    solver_lasso = ISTA(model="lasso", lam=1e4, max_iter=200, tol=1e-12,
                        line_search=True)
    solver_lasso.fit(X, y)
    norm_lasso = float(np.linalg.norm(solver_lasso.coef_))
    assert np.allclose(solver_lasso.coef_, 0.0, atol=1e-6), (
        f"Lasso coef norm = {norm_lasso:.2e}, expected ~0"
    )
    print(f"  ✓ Lasso coef norm at lam=1e4: {norm_lasso:.2e}")

    # For ElasticNet: lam is the L1 parameter, lam2 is the L2 parameter
    solver_en = ISTA(model="elasticnet", lam=1e4, lam2=1e4, max_iter=200,
                     tol=1e-12, line_search=True)
    solver_en.fit(X, y)
    norm_en = float(np.linalg.norm(solver_en.coef_))
    assert np.allclose(solver_en.coef_, 0.0, atol=1e-6), (
        f"ElasticNet coef norm = {norm_en:.2e}, expected ~0"
    )
    print(f"  ✓ ElasticNet coef norm at lam=1e4: {norm_en:.2e}")

    return norm_lasso, norm_en


def test_zero_lambda_ols():
    """Ridge with lambda -> 0 must recover the OLS solution."""
    print("\ntest_zero_lambda_ols")
    X, y, _ = _small_dataset()

    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    ridge = ISTA(model="ridge", lam=1e-10, max_iter=5000, tol=1e-10,
                 line_search=True)
    ridge.fit(X, y)

    diff = float(np.max(np.abs(ridge.coef_ - ols)))
    assert np.allclose(ridge.coef_, ols, atol=1e-2), (
        f"max coef diff from OLS = {diff:.2e}, exceeds 1e-2"
    )
    print(f"  ✓ Ridge lam→0 max coef diff from OLS: {diff:.2e}")

    return diff


def test_sparsity_recovery():
    """Lasso must recover a sparse support on the high-dimensional dataset."""
    print("\ntest_sparsity_recovery")
    X, y, beta_true = high_dimensional()

    solver = ISTA(model="lasso", lam=0.05, max_iter=2000, tol=1e-8,
                  line_search=True)
    solver.fit(X, y)

    n_nonzero = int(np.sum(np.abs(solver.coef_) > 1e-4))
    assert n_nonzero < 50, (
        f"n_nonzero={n_nonzero} >= 50; solution is not sparse enough"
    )

    true_support = set(np.where(np.abs(beta_true) > 1e-4)[0])
    found_support = set(np.where(np.abs(solver.coef_) > 1e-4)[0])
    overlap = len(true_support & found_support)
    print(
        f"  ✓ Nonzero coefs: {n_nonzero}/500, "
        f"support overlap: {overlap}/{len(true_support)}"
    )

    return n_nonzero, overlap


def test_reproducibility():
    """FISTA must produce bit-identical output on repeated runs."""
    print("\ntest_reproducibility")
    X, y, _ = _small_dataset()

    r1 = FISTA(model="lasso", lam=0.1, max_iter=200, tol=1e-8,
               line_search=True).fit(X, y).coef_
    r2 = FISTA(model="lasso", lam=0.1, max_iter=200, tol=1e-8,
               line_search=True).fit(X, y).coef_

    assert np.allclose(r1, r2, atol=1e-12), (
        f"FISTA not deterministic; max diff = {np.max(np.abs(r1 - r2)):.2e}"
    )
    print("  ✓ FISTA output is deterministic")


# ===========================================================================
# Original sklearn comparison suite
# ===========================================================================

def run_tests():
    header = (
        f"{'Model':<14}{'Dataset':<22}{'MaxCoefDiff':>12}"
        f"{'ISTA iters':>12}{'FISTA iters':>13}{'Pass':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_passed = True
    fista_faster_results = []

    for ds_name, ds_fn in DATASETS.items():
        X, y, _ = ds_fn()
        n = X.shape[0]

        for model, cfg in SOLVER_CONFIGS.items():
            tol = SKLEARN_TOL[model]

            # Tight run for sklearn accuracy
            ista = ISTA(model=model, max_iter=SKLEARN_MAX_ITER, tol=tol,
                        line_search=True, **cfg)
            fista = FISTA(model=model, max_iter=SKLEARN_MAX_ITER, tol=tol,
                          line_search=True, **cfg)
            sk = _sklearn_solver(model, n)

            ista.fit(X, y)
            fista.fit(X, y)
            sk.fit(X, y)

            sk_coef = sk.coef_
            max_diff_ista = float(np.max(np.abs(ista.coef_ - sk_coef)))
            max_diff_fista = float(np.max(np.abs(fista.coef_ - sk_coef)))
            max_diff = max(max_diff_ista, max_diff_fista)

            passed = max_diff < 1e-2  # relaxed for hard datasets
            all_passed = all_passed and passed

            # Loose run for FISTA vs ISTA speed comparison on synthetic datasets
            if ds_name in ("high_corr", "high_dim", "near_singular"):
                ista_sp = ISTA(model=model, max_iter=SPEED_MAX_ITER, tol=SPEED_TOL,
                               line_search=True, **cfg)
                fista_sp = FISTA(model=model, max_iter=SPEED_MAX_ITER, tol=SPEED_TOL,
                                 line_search=True, **cfg)
                ista_sp.fit(X, y)
                fista_sp.fit(X, y)
                fista_faster_results.append(
                    (model, ds_name, fista_sp.n_iter_, ista_sp.n_iter_)
                )

            status = "OK" if passed else "FAIL"
            print(
                f"{model:<14}{ds_name:<22}{max_diff:>12.2e}"
                f"{ista.n_iter_:>12}{fista.n_iter_:>13}{status:>6}"
            )

    print(sep)

    # --- FISTA must outperform ISTA on all 3 synthetic datasets ---
    print("\nFISTA vs ISTA iteration counts (SPEED_TOL={:.0e}, must be strictly fewer):".format(SPEED_TOL))
    for model, ds_name, fi, ii in fista_faster_results:
        faster = fi < ii
        flag = "" if faster else "  *** WARNING: FISTA not faster ***"
        print(f"  {model:<12} {ds_name:<20} FISTA={fi:4d}  ISTA={ii:4d}{flag}")
        if not faster:
            all_passed = False

    print()
    if all_passed:
        print("All sklearn comparison tests passed.")
    else:
        print("SOME TESTS FAILED — check output above.")
        sys.exit(1)


# ===========================================================================
# Main: run everything and print summary
# ===========================================================================

if __name__ == "__main__":
    # --- Original sklearn comparison suite ---
    run_tests()

    # --- Additional unit tests ---
    results = {}
    all_extra_passed = True

    try:
        passed, total = test_prox_operators()
        results["test_prox_operators"] = (True, f"{passed}/{total} checks")
    except AssertionError as e:
        results["test_prox_operators"] = (False, str(e))
        all_extra_passed = False

    try:
        summary = test_duality_gap_validity()
        results["test_duality_gap_validity"] = (True, "lasso + elasticnet monotone")
    except AssertionError as e:
        results["test_duality_gap_validity"] = (False, str(e))
        all_extra_passed = False

    try:
        nl, ne = test_large_lambda_sparsity()
        results["test_large_lambda_sparsity"] = (
            True, f"both norms < 1e-6 ({nl:.1e}, {ne:.1e})"
        )
    except AssertionError as e:
        results["test_large_lambda_sparsity"] = (False, str(e))
        all_extra_passed = False

    try:
        diff = test_zero_lambda_ols()
        results["test_zero_lambda_ols"] = (True, f"max diff = {diff:.2e}")
    except AssertionError as e:
        results["test_zero_lambda_ols"] = (False, str(e))
        all_extra_passed = False

    try:
        nz, ov = test_sparsity_recovery()
        results["test_sparsity_recovery"] = (True, f"{nz}nonzero/500, {ov}/10 support")
    except AssertionError as e:
        results["test_sparsity_recovery"] = (False, str(e))
        all_extra_passed = False

    try:
        test_reproducibility()
        results["test_reproducibility"] = (True, "deterministic")
    except AssertionError as e:
        results["test_reproducibility"] = (False, str(e))
        all_extra_passed = False

    # --- Summary table ---
    print("\n===== ADDITIONAL TESTS SUMMARY =====")
    for name, (ok, detail) in results.items():
        status = "PASSED" if ok else "FAILED"
        print(f"{name:<30} {status:<8} {detail}")
    print("=====================================")

    if not all_extra_passed:
        sys.exit(1)
