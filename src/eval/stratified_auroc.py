"""
Stage 3a — Stratified AUROC: truth classification WITHIN matched-typicality bands.

If a probe's headline AUROC comes from reading typicality, then inside a band
where every item has (approximately) the same typicality there is nothing left
to read, and within-band AUROC collapses toward 0.5. If the probe reads truth,
within-band AUROC survives.

Nonparametric counterpart of the Stage-3b mediation — makes no linearity
assumption about score ~ typicality.
"""

from __future__ import annotations

import numpy as np

from ..stats import auroc_with_ci, gap_with_ci


def stratified_auroc(scores: np.ndarray, truth: np.ndarray, typicality: np.ndarray,
                     n_bands: int = 4, min_per_class: int = 5,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Quantile-band typicality stratification.

    Returns per-band AUROC+CI, the item-weighted mean within-band AUROC, the
    pooled headline AUROC on the same items, and the headline-vs-stratified gap
    with a bootstrap CI (the number that goes in the diagnosis table).
    """
    scores, truth, typ = map(np.asarray, (scores, truth, typicality))
    edges = np.quantile(typ, np.linspace(0, 1, n_bands + 1))
    edges[-1] += 1e-9
    band_of = np.clip(np.searchsorted(edges, typ, side="right") - 1, 0, n_bands - 1)

    bands, usable = [], []
    for b in range(n_bands):
        m = band_of == b
        n_pos, n_neg = int(truth[m].sum()), int((~truth[m].astype(bool)).sum())
        entry = {"band": b, "typicality_range": [float(edges[b]), float(edges[b + 1])],
                 "n": int(m.sum()), "n_true": n_pos, "n_false": n_neg}
        if min(n_pos, n_neg) >= min_per_class:
            entry.update(auroc_with_ci(truth[m], scores[m], n_boot=n_boot, seed=seed))
            usable.append((m.sum(), entry["auroc"]))
        else:
            entry.update({"auroc": None, "ci": [None, None],
                          "note": f"skipped: <{min_per_class} per class"})
        bands.append(entry)

    headline = auroc_with_ci(truth, scores, n_boot=n_boot, seed=seed)
    result = {"headline_pooled": headline, "bands": bands,
              "n_bands_usable": len(usable)}

    if usable:
        w = np.array([u[0] for u in usable], dtype=float)
        a = np.array([u[1] for u in usable])
        result["within_band_auroc_weighted"] = round(float((w * a).sum() / w.sum()), 4)
        # gap CI: bootstrap the full dataset, recomputing both quantities per replicate
        rng = np.random.default_rng(seed)
        gaps = []
        n = len(scores)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            g = _gap_once(scores[idx], truth[idx], typ[idx], n_bands, min_per_class)
            if g is not None:
                gaps.append(g)
        if gaps:
            lo, hi = np.percentile(gaps, [2.5, 97.5])
            result["gap"] = {
                "point": round(headline["auroc"] - result["within_band_auroc_weighted"], 4),
                "ci": [round(float(lo), 4), round(float(hi), 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
    return result


def _gap_once(scores, truth, typ, n_bands, min_per_class):
    """headline − weighted-within-band AUROC for one bootstrap replicate."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(truth)) < 2:
        return None
    edges = np.quantile(typ, np.linspace(0, 1, n_bands + 1))
    edges[-1] += 1e-9
    band_of = np.clip(np.searchsorted(edges, typ, side="right") - 1, 0, n_bands - 1)
    ws, as_ = [], []
    for b in range(n_bands):
        m = band_of == b
        if min(truth[m].sum(), (~truth[m].astype(bool)).sum()) >= min_per_class:
            ws.append(m.sum())
            as_.append(roc_auc_score(truth[m], scores[m]))
    if not ws:
        return None
    return roc_auc_score(truth, scores) - float(np.average(as_, weights=ws))
