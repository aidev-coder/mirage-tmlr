"""
Shared statistics. House rule (.5): every AUROC ships with a
bootstrap CI (≥1000 resamples), and overlapping CIs are never called a difference.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT_DEFAULT = 1000


def auroc_with_ci(labels: np.ndarray, scores: np.ndarray,
                  n_boot: int = N_BOOT_DEFAULT, seed: int = 0,
                  alpha: float = 0.05) -> dict:
    """Point AUROC + percentile bootstrap CI over the eval set."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "ci": [None, None], "n": int(len(labels)),
                "note": "single-class eval set"}

    point = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    boots = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        boots.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auroc": round(point, 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "n": int(n), "n_boot": len(boots)}


def gap_with_ci(labels_a, scores_a, labels_b, scores_b,
                n_boot: int = N_BOOT_DEFAULT, seed: int = 0,
                alpha: float = 0.05) -> dict:
    """Bootstrap CI on AUROC(a) − AUROC(b) — the diagnosis-table 'gap' column.

    Resamples both eval sets independently per replicate. A CI excluding 0 is
    the criterion for calling the gap real.
    """
    la, sa = np.asarray(labels_a), np.asarray(scores_a)
    lb, sb = np.asarray(labels_b), np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(la), len(la))
        ib = rng.integers(0, len(lb), len(lb))
        if len(np.unique(la[ia])) < 2 or len(np.unique(lb[ib])) < 2:
            continue
        diffs.append(roc_auc_score(la[ia], sa[ia]) - roc_auc_score(lb[ib], sb[ib]))
    point = float(roc_auc_score(la, sa) - roc_auc_score(lb, sb))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"gap": round(point, 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": len(diffs)}
