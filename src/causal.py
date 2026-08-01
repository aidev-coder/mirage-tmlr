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
from collections import defaultdict
from pathlib import Path

import numpy as np

from .stage3 import _extract_or_load, load_corpus

_ROOT = Path(__file__).resolve().parent.parent
DIAGONAL = ("TT", "FA")
TYPICAL = ("TT", "FT")   # the high-frequency column (the typicality axis, by construction)


def _fit_logreg(X: np.ndarray, y: np.ndarray, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(sc.transform(X), y)
    # fold the scaler into a single (w, b) acting on raw h
    w = (clf.coef_[0] / sc.scale_)
    b = float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())
    return w.astype(np.float64), b


def _twin_pairs(items: list[dict]) -> list[tuple[int, int]]:
    by_subj = defaultdict(lambda: {"t": [], "f": []})
    for i, it in enumerate(items):
        subj = it["entities"][0] if it.get("entities") else it.get("entity")
        by_subj[subj]["t" if it["truth"] else "f"].append(i)
    pairs = []
    for d in by_subj.values():
        for ti in d["t"]:
            for fi in d["f"]:
                pairs.append((ti, fi))
    return pairs


def _mediation(w: np.ndarray, b: float, u: np.ndarray,
               H_true: np.ndarray, H_false: np.ndarray) -> dict:
    def f(h):
        return h @ w + b
    ut, uf = H_true @ u, H_false @ u
    H_cf = H_true - np.outer(ut, u) + np.outer(uf, u)   # swap only the u-coordinate
    te = f(H_true) - f(H_false)
    nie = f(H_true) - f(H_cf)
    keep = np.abs(te) > 1e-6
    frac = float(np.median(nie[keep] / te[keep]))
    return {
        "total_effect_median": round(float(np.median(te)), 4),
        "indirect_effect_median": round(float(np.median(nie)), 4),
        "fraction_mediated_median": round(frac, 4),
        "fraction_mediated_mean": round(float(np.mean(nie[keep] / te[keep])), 4),
        "n_pairs": int(len(te)),
    }


def run(substrate, corpus_path: str | Path, batch_size: int = 32,
        seed: int = 20260719, commit_fn=None) -> dict:
    items = load_corpus(corpus_path)
    corpus_hash = Path(corpus_path).stem.split("_v")[-1]
    texts = [it["text"] for it in items]
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    typical = np.isin(cells, TYPICAL)
    diag = np.isin(cells, DIAGONAL)

    H = _extract_or_load(substrate, texts, corpus_hash, batch_size, commit_fn)
    n_layers = H.shape[1]
    pairs = _twin_pairs(items)
    ti = np.array([p[0] for p in pairs]); fi = np.array([p[1] for p in pairs])

    per_layer = []
    for L in range(n_layers):
        Xl = H[:, L, :].astype(np.float64)
        w_fair, b_fair = _fit_logreg(Xl, truth, seed)
        w_field, b_field = _fit_logreg(Xl[diag], truth[diag], seed)
        w_typ, _ = _fit_logreg(Xl, typical, seed)
        u = w_typ / (np.linalg.norm(w_typ) + 1e-12)
        Ht, Hf = Xl[ti], Xl[fi]
        per_layer.append({
            "layer": L,
            "fielded": _mediation(w_field, b_field, u, Ht, Hf),
            "fair": _mediation(w_fair, b_fair, u, Ht, Hf),
        })
        if commit_fn:
            commit_fn()
        e = per_layer[-1]
        print(f"  [L{L}] fielded frac_mediated={e['fielded']['fraction_mediated_median']} "
              f"| fair={e['fair']['fraction_mediated_median']} "
              f"(TE_field={e['fielded']['total_effect_median']})", flush=True)

    hl = n_layers // 2
    return {
        "model": substrate.model_id,
        "corpus": Path(corpus_path).name,
        "corpus_hash": corpus_hash,
        "n_pairs": len(pairs),
        "typicality_direction": "logreg on TT+FT vs TA+FA (frequency column)",
        "headline_layer": hl,
        "per_layer": per_layer,
        "provenance": "measured",
        "seed": seed,
    }
