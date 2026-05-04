"""
Experiment modules for regularized regression evaluation.

This package contains all experiment infrastructure used in the empirical study:

synthetic
    Structured data-generating processes (high-correlation, high-dimensional,
    near-singular) for algorithm stress-testing and convergence analysis.

data_pipeline
    End-to-end preprocessing pipeline for the DataCo Smart Supply Chain dataset,
    including feature engineering, leakage-free design matrix construction, and
    walk-forward cross-validation splits.

evaluation
    Evaluation utilities: cross-validation scoring, regularization path
    computation, bias-variance sweeps, model comparison tables, and timing.

convergence
    Script that generates the ISTA vs FISTA convergence figures (objective and
    duality gap vs iteration) across all 9 dataset-model combinations.
"""
