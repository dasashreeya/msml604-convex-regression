"""
Data loading, feature engineering, and design matrix construction
for the DataCo Smart Supply Chain regression task.

This module prepares the real-world evaluation dataset used in the empirical
experiments of Section 4.  The target is order-item sales revenue (USD), and
the design matrix is constructed from operational features that are plausibly
available at order-placement time, with strict exclusion of post-hoc outcome
variables to prevent target leakage.

Dataset
-------
DataCo Smart Supply Chain for Big Data Analysis (Konstantinos Konstantinou, Kaggle).
URL: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Target: Sales  — order-item revenue in USD

CSV columns used (53 total):
    order date (DateOrders)       — datetime → temporal features + row ordering
    Days for shipment (scheduled) — numeric feature
    Order Item Discount           — numeric feature (USD discount amount)
    Order Item Discount Rate      — numeric feature (0–1 fraction)
    Order Item Product Price      — numeric feature (unit price in USD)
    Order Item Quantity           — numeric feature
    Latitude / Longitude          — numeric features (customer location)
    Type                          — categorical (payment method)
    Shipping Mode                 — categorical
    Customer Segment              — categorical
    Market                        — categorical
    Delivery Status               — categorical
    Department Name               — categorical (product department)

Excluded from X — target leakage (algebraically derived from Sales):
    Order Item Total         — ≈ Sales + Discount  (r=0.99)
    Sales per customer       — aggregation of Sales (r=0.99)
    Benefit per order        — Sales − Cost
    Order Profit Per Order   — Sales − Cost
    Order Item Profit Ratio  — Profit / Sales

Excluded — PII / identifiers (no predictive signal):
    Customer Email, Customer Fname, Customer Lname, Customer Password,
    Customer Street, Customer City, Customer State, Customer Zipcode,
    Order Id, Order Item Id, Customer Id, Order Customer Id,
    Product Card Id, Product Category Id, Category Id, Department Id,
    Order Zipcode, Product Image, Product Name, Product Description

Excluded — duplicate:
    Product Price  — identical to Order Item Product Price (diff = 0)
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "DataCoSupplyChainDataset.csv")

TARGET_COL = "Sales"

DATE_COL = "order date (DateOrders)"

# Columns derived from the target — never include in X.
LABEL_COLS = {
    "Order Item Total",
    "Sales per customer",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
}

# Raw numeric feature columns (explicitly enumerated, no leakage)
FEATURE_COLS = [
    "Days for shipment (scheduled)",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Quantity",
    "Latitude",
    "Longitude",
]

# Low-cardinality categorical columns to one-hot encode
CAT_FEATURE_COLS = [
    "Type",              # 4 values: DEBIT, TRANSFER, CASH, PAYMENT
    "Shipping Mode",     # 4 values: Standard Class, First Class, Second Class, Same Day
    "Customer Segment",  # 3 values: Consumer, Home Office, Corporate
    "Market",            # 5 values: Pacific Asia, USCA, Africa, Europe, LATAM
    "Delivery Status",   # 4 values: Advance shipping, Late delivery, On time, Canceled
    "Department Name",   # 11 values: Fitness, Apparel, Golf, …
]

# All non-feature columns to exclude from the design matrix
_EXCLUDE_COLS = LABEL_COLS | {
    TARGET_COL, DATE_COL, "shipping date (DateOrders)",
    # PII / identifiers
    "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
    "Customer Street", "Customer City", "Customer State", "Customer Zipcode",
    "Customer Country", "Order City", "Order Country", "Order State",
    "Order Zipcode", "Order Region",
    "Customer Id", "Order Id", "Order Item Id", "Order Item Cardprod Id",
    "Order Customer Id", "Product Card Id", "Product Category Id",
    "Category Id", "Department Id",
    # Duplicate / redundant
    "Product Price",    # identical to Order Item Product Price
    "Product Status",
    "Product Image",
    "Product Name",
    "Product Description",
    "Order Status",     # post-order outcome, not known at prediction time
    "Late_delivery_risk",
    "Days for shipping (real)",  # actual outcome (not scheduled)
}


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the DataCo CSV and perform minimal type coercion.

    - Uses latin-1 encoding (required for this dataset)
    - Parses `order date (DateOrders)` as datetime
    - Drops rows where Sales (target) is NaN
    """
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%m/%d/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df


def engineer_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Create derived features from the raw DataCo DataFrame.

    Temporal features (from order date):
        order_hour, order_day_of_week, order_month, order_quarter

    One-hot encoded categoricals (drop_first=True, separator '__'):
        Type__*, Shipping Mode__*, Customer Segment__*,
        Market__*, Delivery Status__*, Department Name__*

    Lag features of the target (shift to avoid leakage):
        lag1_target          — previous row's Sales value
        lag2_target          — two rows back
        rolling3_mean_target — 3-step rolling mean (shifted by 1)

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_raw().
    target_col : str
        Name of the regression target column.

    Returns
    -------
    pd.DataFrame sorted by order date with new columns; first 2 NaN lag rows dropped.
    """
    df = df.copy().sort_values(DATE_COL).reset_index(drop=True)

    # --- Temporal features ---
    dt = df[DATE_COL]
    df["order_hour"]        = dt.dt.hour.astype(float)
    df["order_day_of_week"] = dt.dt.dayofweek.astype(float)
    df["order_month"]       = dt.dt.month.astype(float)
    df["order_quarter"]     = dt.dt.quarter.astype(float)

    # --- One-hot encode low-cardinality categoricals ---
    cats_present = [c for c in CAT_FEATURE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cats_present, drop_first=True, prefix_sep="__",
                        dtype=float)

    # --- Lag features of the target ---
    s = df[target_col]
    df["lag1_target"]          = s.shift(1)
    df["lag2_target"]          = s.shift(2)
    df["rolling3_mean_target"] = s.shift(1).rolling(3, min_periods=1).mean()

    # Drop the first two rows that have NaN lags
    df = df.dropna(subset=["lag1_target", "lag2_target"]).reset_index(drop=True)

    return df


def build_design_matrix(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Split engineered DataFrame into (X, y, feature_names).

    Label columns (Order Item Total, Sales per customer, etc.) are always
    excluded regardless of feature_cols.

    Parameters
    ----------
    df : pd.DataFrame
        Output of engineer_features().
    target_col : str
        Regression target column.
    feature_cols : list of str or None
        Explicit feature list. If None, auto-selects all numeric columns
        minus target, date, PII, leakage, and ID columns.

    Returns
    -------
    X : np.ndarray, shape (n, p), float64  — NOT standardized
    y : np.ndarray, shape (n,), float64
    feature_names : list of str
    """
    if feature_cols is None:
        # All numeric columns in the df (after one-hot encoding, dummies are float)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            c for c in numeric_cols
            if c not in _EXCLUDE_COLS and c != target_col
        ]

    # Enforce leakage exclusion unconditionally
    feature_cols = [c for c in feature_cols if c not in LABEL_COLS and c != target_col]

    X = df[feature_cols].astype(np.float64).values
    y = df[target_col].astype(np.float64).values

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    return X, y, list(feature_cols)


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit mean/std on X_train only, then apply to both splits.

    Columns with zero variance (e.g., one-hot dummies with all-same value)
    are left with sigma=1 (standardized value = 0).

    Returns
    -------
    X_train_std, X_test_std, mu, sigma
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def time_series_cv_splits(
    n: int,
    n_splits: int = 5,
    min_train_size: int = 200,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward (expanding window) cross-validation for time series.

    Divides indices 0..n-1 into n_splits+1 equal blocks.
    Split i uses blocks 0..i as train, block i+1 as validation.
    Validation is always strictly after training — no leakage.

    Returns list of (train_idx, val_idx) tuples.
    """
    block_size = n // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        train_end = (i + 1) * block_size
        test_end  = (i + 2) * block_size if i < n_splits - 1 else n
        if train_end < min_train_size:
            continue
        splits.append((np.arange(train_end), np.arange(train_end, test_end)))
    return splits


def get_lambda_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_lam: int = 40,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Data-driven lambda grid from lambda_max down to eps * lambda_max.

    lambda_max = max(|X^T y|) / n — the smallest lambda that zeros all
    Lasso coefficients.

    Returns log-spaced array (descending).
    """
    lam_max = float(np.max(np.abs(X.T @ y))) / X.shape[0]
    return np.logspace(np.log10(lam_max), np.log10(eps * lam_max), n_lam)
