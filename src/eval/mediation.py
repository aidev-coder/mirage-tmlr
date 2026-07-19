"""
Stage 3b — Mediation: does truth still explain the probe score once typicality
is partialled out?

Model: probe_score ~ truth + typicality (standardized OLS).
If truth's coefficient dies when typicality enters, the probe is confounded.

Assumption flag: OLS mediation assumes an additive-linear score surface. Stage 3a
(stratified AUROC) is the nonparametric triangulation partner; per the project's standing directive §3
Stage 3, direction-disagreement between them must be investigated, not averaged.
"""

from __future__ import annotations

import numpy as np


def _std(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def mediation(scores: np.ndarray, truth: np.ndarray, typicality: np.ndarray,
              covariates: dict[str, np.ndarray] | None = None) -> dict:
    """Standardized coefficients for score ~ truth (+ typicality + covariates),
    with the truth-coefficient shrinkage ratio as the headline diagnostic.

    D-011: fragmentation is partialled out as a covariate alongside typicality,
    exactly as typicality is — confounds are controlled analytically, never by
    distorting the corpus. Pass it (and any further confound) via `covariates`."""
    import statsmodels.api as sm

    s, y, t = _std(scores), _std(truth), _std(typicality)
    cov_names = list(covariates or {})
    cov_cols = [_std(covariates[k]) for k in cov_names]

    marginal = sm.OLS(s, sm.add_constant(np.column_stack([y]))).fit()
    joint = sm.OLS(s, sm.add_constant(np.column_stack([y, t, *cov_cols]))).fit()

    b_truth_marginal = float(marginal.params[1])
    b_truth_joint = float(joint.params[1])
    b_typ_joint = float(joint.params[2])

    def ci(fit, i):
        lo, hi = fit.conf_int()[i]
        return [round(float(lo), 4), round(float(hi), 4)]

    shrink = (1.0 - b_truth_joint / b_truth_marginal) if abs(b_truth_marginal) > 1e-9 else None
    out = {
        "n": int(len(s)),
        "truth_beta_marginal": round(b_truth_marginal, 4),
        "truth_beta_marginal_ci": ci(marginal, 1),
        "truth_beta_partialled": round(b_truth_joint, 4),
        "truth_beta_partialled_ci": ci(joint, 1),
        "typicality_beta": round(b_typ_joint, 4),
        "typicality_beta_ci": ci(joint, 2),
        "truth_beta_shrinkage": None if shrink is None else round(float(shrink), 4),
        "r2_truth_only": round(float(marginal.rsquared), 4),
        "r2_joint": round(float(joint.rsquared), 4),
        "covariates": cov_names,
        "assumption": "additive-linear; triangulate with stage 3a",
    }
    for j, name in enumerate(cov_names):
        i = 3 + j
        out[f"{name}_beta"] = round(float(joint.params[i]), 4)
        out[f"{name}_beta_ci"] = ci(joint, i)
    return out
