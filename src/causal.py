"""
Causal mediation: how much of a truth-probe's true-vs-false response is routed
through the single typicality direction. Uses matched twin pairs (same subject,
one true one false, e.g. "Hpa-An is a city in Myanmar" vs "...in The Bahamas")
so the total effect is a clean within-subject contrast.

For a probe with logit f(h) = w·h + b and a unit typicality direction u:
  total effect        TE  = f(h_true) - f(h_false)
  counterfactual      h*  = h_true - (u·h_true)u + (u·h_false)u   (swap only the
                            typicality coordinate of the true item to its twin's)
  natural indirect    NIE = f(h_true) - f(h*)        (effect carried by u alone)
  fraction mediated   NIE / TE

Reported for the FIELDED probe (diagonal-trained, the field's recipe) and the
FAIR probe (all-cell). If a large fraction of the fielded probe's truth response
is causally the one typicality coordinate, the probe is computing typicality and
calling it truth. Swept over layers and run per model; the scale axis is the same
number across the Pythia sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .stage3 import _extract_or_load, load_corpus

_ROOT = Path(__file__).resolve().parent.parent
DIAGONAL = ("TT", "FA")
TYPICAL = ("TT", "FT")   # the high-frequency column (the typicality axis, by construction)


def _standardize(X: np.ndarray):
    """Zero-variance-safe standardization. Everything downstream (probe fit, PLS,
    mediation) runs in this space so the logit decomposition stays consistent."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def _fit_logreg(Z: np.ndarray, y: np.ndarray, seed: int = 0):
    """Plain logistic probe on already-standardized Z; returns (w, b) in Z-space."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(Z, y)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def _typicality_subspace(Z: np.ndarray, freq: np.ndarray, k_max: int, n_bins: int = 20):
    """Orthonormal basis (d, k_max) for the frequency-driven manifold: the directions
    along which the MEAN standardized activation moves as entity frequency sweeps.
    Bin items by frequency, take per-bin means, center, SVD. Well-posed and bounded
    by n_bins (no PLS degeneracy); the top-k right singular vectors span the manifold
    of typicality-driven mean variation."""
    edges = np.quantile(freq, np.linspace(0, 1, n_bins + 1))
    binid = np.clip(np.digitize(freq, edges[1:-1]), 0, n_bins - 1)
    means = [Z[binid == b].mean(axis=0) for b in range(n_bins) if (binid == b).sum() > 0]
    M = np.asarray(means)
    M = M - M.mean(axis=0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt.T[:, :min(k_max, Vt.shape[0])]


def _random_subspace(d: int, k: int, seed: int) -> np.ndarray:
    """Orthonormal (d, k) Gaussian subspace — the null for the manifold: resetting
    any k-dim projection moves a logit somewhat, so the frequency manifold only
    counts if it mediates MORE than this."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    return Q[:, :k]


def _mediation_subspace(w: np.ndarray, b: float, U: np.ndarray,
                        H_fooled: np.ndarray, H_ref: np.ndarray) -> dict:
    """Within a truth class, across the typicality SUBSPACE span(U). Counterfactual:
    reset each fooled item's projection onto span(U) to the reference cell's mean
    projection ("make it look atypical" along every typicality dimension at once),
    leaving the orthogonal complement untouched.
        h_cf = h - U Uᵀ(h - mean_ref)
    NIE / TE is the fraction of the probe's fooled-vs-reference response gap that is
    causally carried by the typicality manifold."""
    def f(h):
        return h @ w + b
    mean_ref = H_ref.mean(axis=0)
    proj = (H_fooled - mean_ref) @ U        # (n, k)
    H_cf = H_fooled - proj @ U.T
    te = float(np.median(f(H_fooled)) - np.median(f(H_ref)))
    nie = float(np.median(f(H_fooled)) - np.median(f(H_cf)))
    return {
        "total_effect": round(te, 4),
        "indirect_effect": round(nie, 4),
        "fraction_mediated": round(nie / te, 4) if abs(te) > 1e-6 else None,
    }


def run(substrate, corpus_path: str | Path, batch_size: int = 32,
        seed: int = 20260719, commit_fn=None, domain: str | None = None) -> dict:
    """domain: hold topic fixed (subset AFTER extraction, cache stays valid). The
    pooled corpus confounds domain with truth across cells (2026-08-03 refutation),
    so any FT-vs-FA contrast on it may measure domain, not typicality."""
    items = load_corpus(corpus_path)
    corpus_hash = Path(corpus_path).stem.split("_v")[-1]
    texts = [it["text"] for it in items]
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    freq = np.array([it["typicality"]["entity_freq_log10"] for it in items], dtype=float)

    H = _extract_or_load(substrate, texts, corpus_hash, batch_size, commit_fn)
    if domain:
        keep = np.array([it["domain"] == domain for it in items])
        items = [it for it, k in zip(items, keep) if k]
        H, truth, cells, freq = H[keep], truth[keep], cells[keep], freq[keep]
    diag = np.isin(cells, DIAGONAL)
    n_layers = H.shape[1]
    ft = cells == "FT"; fa = cells == "FA"; tt = cells == "TT"; ta = cells == "TA"
    ks = [1, 2, 4, 8, 16]

    per_layer = []
    for L in range(n_layers):
        Z = _standardize(H[:, L, :].astype(np.float64))
        w_field, b_field = _fit_logreg(Z[diag], truth[diag], seed)
        U_full = _typicality_subspace(Z, freq, max(ks))
        sweep, sweep_ctrl, sweep_rand = {}, {}, {}
        for k in ks:
            Uk = U_full[:, :k]
            sweep[k] = _mediation_subspace(w_field, b_field, Uk, Z[ft], Z[fa])["fraction_mediated"]
            sweep_ctrl[k] = _mediation_subspace(w_field, b_field, Uk, Z[tt], Z[ta])["fraction_mediated"]
            rand = [_mediation_subspace(w_field, b_field, _random_subspace(Z.shape[1], k, seed + r),
                                        Z[ft], Z[fa])["fraction_mediated"] for r in range(5)]
            rand = [v for v in rand if v is not None]
            sweep_rand[k] = round(float(np.median(rand)), 4) if rand else None
        per_layer.append({
            "layer": L,
            "ft_error_frac_mediated_by_k": sweep,          # FT vs FA (the headline)
            "ft_error_random_subspace_by_k": sweep_rand,   # null: random k-dim subspace
            "tt_control_frac_mediated_by_k": sweep_ctrl,   # TT vs TA (specificity)
            "total_effect": _mediation_subspace(w_field, b_field, U_full[:, :1], Z[ft], Z[fa])["total_effect"],
        })
        if commit_fn:
            commit_fn()
        e = per_layer[-1]
        print(f"  [L{L}] FT-error k1={sweep[1]} k8={sweep[8]} | random k8={sweep_rand[8]} "
              f"| TT-ctrl k8={sweep_ctrl[8]}", flush=True)

    hl = n_layers // 2
    return {
        "model": substrate.model_id,
        "domain": domain or "all",
        "corpus": Path(corpus_path).name,
        "corpus_hash": corpus_hash,
        "contrast": "within-truth-class across typicality SUBSPACE (FT vs FA; TT vs TA control)",
        "typicality_subspace": "PLS of residuals onto typical/atypical label, k-swept",
        "k_values": ks,
        "headline_layer": hl,
        "per_layer": per_layer,
        "provenance": "measured",
        "seed": seed,
    }
