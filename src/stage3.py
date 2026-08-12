"""
Stage 3 — the three orthogonal decorrelation tests, run on the finalized 2x2
corpus (data/corpus/mirage_2x2_v<hash>.jsonl) for one detector on one substrate.

  3a stratified AUROC   truth classification within matched-typicality bands
  3b mediation          score ~ truth + typicality + fragmentation (D-011 covariate)
  3c adversarial split  train on the diagonal, test only on the off-diagonal (TA+FT)

Probe scores feeding 3a/3b are OUT-OF-FOLD (K-fold cross-fitted): every item is
scored by a probe that never saw it, so the stratified/mediation numbers are not
train-set-optimistic. 3c does its own diagonal->off-diagonal split.

Layers are swept in full (D-004 / the project's standing directive §4.3); the mid-depth canary layer is
reported as the headline but never as a cherry-picked best.

Output: results/stage3_<detector>_<model>_<date>.json — full per-layer curves for
all three tests, each value with its CI, provenance "measured".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .eval.adversarial_split import adversarial_split
from .eval.mediation import mediation
from .eval.stratified_auroc import stratified_auroc

_ROOT = Path(__file__).resolve().parent.parent


def load_corpus(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _extract_or_load(substrate, texts, corpus_hash, batch_size, commit_fn=None) -> np.ndarray:
    mh = hashlib.sha256(substrate.model_id.encode()).hexdigest()[:12]
    shard = Path(substrate.cache_dir) / "stage3" / f"{mh}_{corpus_hash}.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    if shard.exists():
        a = np.load(shard)
        if a.shape[0] == len(texts):
            print(f"  [cache] hidden states loaded {a.shape}", flush=True)
            return a
        print(f"  [cache] stale ({a.shape[0]} != {len(texts)}), re-extracting", flush=True)
    a = substrate.hidden_states_matrix(texts, batch_size=batch_size)
    np.save(shard, a)
    if commit_fn:
        commit_fn()
    print(f"  [extract] hidden states saved {a.shape}", flush=True)
    return a


def _oof_scores(X: np.ndarray, y: np.ndarray, device, n_folds: int, seed: int) -> np.ndarray:
    """K-fold cross-fitted P(true): each item scored by a probe trained without it."""
    from .probes.torch_mlp import _pick_device, _train_one
    dev = _pick_device(device)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), n_folds)
    scores = np.zeros(len(y), dtype=np.float64)
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        scores[te] = _train_one(X[tr].astype(np.float32), y[tr].astype(np.float32),
                                X[te].astype(np.float32), dev, seed=seed)
    return scores


def _edit_oof(X: np.ndarray, edited: np.ndarray, device, n_folds: int, seed: int) -> np.ndarray:
    """Out-of-fold P(edited) from hidden states at one layer — the edit-signature
    the §4.2 canary flags. Partialled out in Stage-3 so truth results are net of any
    edit component (extends D-011 from fragmentation to the edit axis; the swap
    signature is faint but seed-variable, notebook 2026-07-20)."""
    from .probes.torch_mlp import _pick_device, _train_one
    dev = _pick_device(device)
    edited = np.asarray(edited, dtype=np.float32)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(edited)), n_folds)
    scores = np.zeros(len(edited), dtype=np.float64)
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        scores[te] = _train_one(X[tr].astype(np.float32), edited[tr],
                                X[te].astype(np.float32), dev, seed=seed)
    return scores


def _fielded_oof_scores(X: np.ndarray, truth: np.ndarray, cells: np.ndarray,
                        device, n_folds: int, seed: int) -> np.ndarray:
    """The field's instrument, scored on the whole corpus: probes are trained ONLY
    on the diagonal (TT+FA) — the confounded recipe every benchmark uses. Diagonal
    items get K-fold cross-fitted scores; off-diagonal items are scored by the mean
    of the K diagonal-trained probes. This is the instrument 3c indicts, now made
    available to 3a/3b so all three tests audit the SAME probe."""
    from .eval.adversarial_split import DIAGONAL
    from .probes.torch_mlp import _pick_device, _train_one
    dev = _pick_device(device)
    diag = np.flatnonzero(np.isin(cells, DIAGONAL))
    off = np.flatnonzero(~np.isin(cells, DIAGONAL))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(diag), n_folds)
    scores = np.zeros(len(truth), dtype=np.float64)
    off_acc = np.zeros(len(off), dtype=np.float64)
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        ytr = truth[tr].astype(np.float32)
        scores[te] = _train_one(X[tr].astype(np.float32), ytr, X[te].astype(np.float32), dev, seed=seed)
        off_acc += _train_one(X[tr].astype(np.float32), ytr, X[off].astype(np.float32), dev, seed=seed)
    scores[off] = off_acc / n_folds
    return scores


def _cell_score_summary(scores: np.ndarray, cells: np.ndarray) -> dict:
    out = {}
    for c in ("TT", "TA", "FT", "FA"):
        s = scores[cells == c]
        if len(s):
            q = np.percentile(s, [10, 50, 90])
            out[c] = {"n": int(len(s)), "mean": round(float(s.mean()), 4),
                      "q10": round(float(q[0]), 4), "q50": round(float(q[1]), 4),
                      "q90": round(float(q[2]), 4)}
    return out


def _fragmentation_oof(items: list[dict], truth: np.ndarray, tokenizer,
                       n_folds: int, seed: int) -> np.ndarray:
    """Out-of-fold P(true) from tokenization features ALONE — the fragmentation
    confound signal (C3). Partialled out in 3b so 3a/3b judge truth net of it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from .corpus_build import fragmentation_features
    feats = fragmentation_features([it["text"] for it in items],
                                   [it["entity"] for it in items], tokenizer)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(truth)), n_folds)
    scores = np.zeros(len(truth), dtype=np.float64)
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=500, random_state=seed).fit(sc.transform(feats[tr]), truth[tr])
        scores[te] = clf.predict_proba(sc.transform(feats[te]))[:, 1]
    return scores


def run(substrate, corpus_path: str | Path, detector: str = "saplma",
        n_folds: int = 5, batch_size: int = 32, device: str | None = None,
        seed: int = 20260719, commit_fn=None, domain: str | None = None) -> dict:
    """domain: restrict to one corpus domain. Only 'cities' populates all four
    cells; inventions/elements have no atypical column, so the pooled off-diagonal
    mixes a cities-only TA against a multi-domain FT. Running cities-only holds
    domain fixed and checks the collapse is typicality, not topic."""
    items = load_corpus(corpus_path)
    corpus_hash = Path(corpus_path).stem.split("_v")[-1]
    texts = [it["text"] for it in items]
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    typicality = np.array([it["typicality"]["entity_freq_log10"] for it in items], dtype=float)

    edited = np.array([bool(it.get("edited")) for it in items])
    H = _extract_or_load(substrate, texts, corpus_hash, batch_size, commit_fn)
    if domain:   # subset AFTER extraction so the full-corpus activation cache stays valid
        keep = np.array([it["domain"] == domain for it in items])
        items = [it for it, k in zip(items, keep) if k]
        H, truth, cells, typicality, edited = (H[keep], truth[keep], cells[keep],
                                               typicality[keep], edited[keep])
    n_layers = H.shape[1]
    frag_oof = _fragmentation_oof(items, truth, substrate.tokenizer, n_folds, seed)
    edit_oof = _edit_oof(H[:, n_layers // 2, :], edited, device, n_folds, seed)

    def factory(_seed=seed):
        from .probes.saplma import SaplmaProbe
        return SaplmaProbe(seed=_seed)

    headline_layer = n_layers // 2
    fielded_headline = None
    per_layer = []
    for L in range(n_layers):
        Xl = H[:, L, :]
        oof_all = _oof_scores(Xl, truth, device, n_folds, seed)
        oof_fielded = _fielded_oof_scores(Xl, truth, cells, device, n_folds, seed)
        if L == headline_layer:
            fielded_headline = oof_fielded
        cov = {"fragmentation": frag_oof, "edit": edit_oof}
        from .eval.adversarial_split import OFF_DIAGONAL
        from .stats import auroc_with_ci
        off_idx = np.flatnonzero(np.isin(cells, OFF_DIAGONAL))
        entry = {
            "layer": L,
            "allcell_off_diagonal": auroc_with_ci(truth[off_idx], oof_all[off_idx], seed=seed),
            "stratified_allcell": stratified_auroc(oof_all, truth, typicality, seed=seed),
            "mediation_allcell": mediation(oof_all, truth, typicality, covariates=cov),
            "stratified_fielded": stratified_auroc(oof_fielded, truth, typicality, seed=seed),
            "mediation_fielded": mediation(oof_fielded, truth, typicality, covariates=cov),
            "adversarial": adversarial_split(Xl, truth, cells, factory, seed=seed),
            "fielded_cell_scores": _cell_score_summary(oof_fielded, cells),
        }
        per_layer.append(entry)
        if commit_fn:
            commit_fn()
        gf = entry["stratified_fielded"].get("gap", {})
        print(f"  [L{L}] fielded: strat_gap={gf.get('point')} {gf.get('ci')} "
              f"truth_beta {entry['mediation_fielded']['truth_beta_marginal']}->"
              f"{entry['mediation_fielded']['truth_beta_partialled']} | "
              f"adv_off={entry['adversarial']['off_diagonal'].get('auroc')} | "
              f"allcell_truth_beta_partialled={entry['mediation_allcell']['truth_beta_partialled']}", flush=True)

    return {
        "detector": detector,
        "model": substrate.model_id,
        "corpus": Path(corpus_path).name,
        "corpus_hash": corpus_hash,
        "domain": domain or "all",
        "n": len(items),
        "cell_counts": {c: int((cells == c).sum()) for c in ("TT", "TA", "FT", "FA")},
        "n_folds": n_folds,
        "typicality_axis": "entity_freq_log10",
        "mediation_covariates": ["typicality", "fragmentation", "edit"],
        "headline_layer": headline_layer,
        "fielded_scores_headline": {
            "layer": headline_layer,
            "score": [round(float(s), 6) for s in fielded_headline],
            "cell": cells.tolist(),
            "truth": [bool(t) for t in truth],
        },
        "per_layer": per_layer,
        "provenance": "measured",
        "seed": seed,
    }


def run_transfer(substrate, corpus_path, am_dir=None, layer=None, seed=0,
                 batch_size=32, commit_fn=None) -> dict:
    """Train the probe on the FIELD'S dataset, test on our crossed corpus.

    Every other result here trains on our own congruent cells, where entity
    frequency and truth agree because we made them agree. That is a statement
    about what the standard recipe does when the alignment is present, not about
    the published datasets. This asks the other question: a probe trained exactly
    as the field trains one, on Azaria and Mitchell's true/false statements, is
    then asked (a) can its score separate common from rare among items that are
    ALL TRUE, and (b) what does it do on the disagreement cells.

    The row-swap construction behind that dataset roughly preserves entity
    frequency across truth, so a probe reading only frequency has nothing to gain
    there. If frequency reading shows up anyway, it was not taught by the training
    distribution.
    """
    import numpy as np

    from .probes.saplma import SaplmaProbe
    from .stats import auroc_with_ci

    am_dir = Path(am_dir or (_ROOT / "data" / "raw" / "azaria_mitchell"))
    train_texts, train_y = [], []
    for csv in sorted(am_dir.glob("*_true_false.csv")):
        if csv.name.startswith("neg_"):
            continue
        with open(csv, encoding="utf-8-sig") as fh:
            import csv as _csv
            for row in _csv.DictReader(fh):
                s = (row.get("statement") or "").strip()
                lab = (row.get("label") or "").strip()
                if s and lab in ("0", "1"):
                    train_texts.append(s)
                    train_y.append(lab == "1")
    train_y = np.array(train_y)
    print(f"[transfer] training on {len(train_texts)} A&M statements "
          f"({train_y.mean():.1%} true)", flush=True)

    items = load_corpus(corpus_path)
    test_texts = [it["text"] for it in items]
    cells = np.array([it["cell"] for it in items])
    truth = np.array([bool(it["truth"]) for it in items])

    H_tr = substrate.hidden_states_matrix(train_texts, batch_size=batch_size)
    H_te = substrate.hidden_states_matrix(test_texts, batch_size=batch_size)
    L = layer if layer is not None else H_tr.shape[1] // 2
    if commit_fn:
        commit_fn()

    probe = SaplmaProbe(seed=seed).fit(H_tr[:, L, :].astype(np.float64), train_y)
    s = np.asarray(probe.score(H_te[:, L, :].astype(np.float64)))

    # held-out slice of the training distribution, for a like-for-like headline
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_texts))
    cut = int(0.8 * len(idx))
    p2 = SaplmaProbe(seed=seed).fit(H_tr[idx[:cut], L, :].astype(np.float64), train_y[idx[:cut]])
    in_dist = auroc_with_ci(train_y[idx[cut:]],
                            np.asarray(p2.score(H_tr[idx[cut:], L, :].astype(np.float64))), seed=seed)

    true_only = np.isin(cells, ["TT", "TA"])
    freq_read = auroc_with_ci((cells[true_only] == "TT"), s[true_only], seed=seed)
    off = np.isin(cells, ["TA", "FT"])
    off_auroc = auroc_with_ci(truth[off], s[off], seed=seed)

    out = {"experiment": "transfer_am_to_crossed", "model": substrate.model_id,
           "layer": int(L), "corpus": Path(corpus_path).name,
           "n_train": len(train_texts), "n_test": len(items),
           "in_distribution_on_am": in_dist,
           "frequency_read_among_true_only": freq_read,
           "disagreement_auroc": off_auroc,
           "cell_means": {c: round(float(s[cells == c].mean()), 4)
                          for c in ("TT", "TA", "FT", "FA")},
           "provenance": "measured"}
    print(f"[transfer] A&M held-out {in_dist['auroc']:.3f} | "
          f"frequency-read among TRUE only {freq_read['auroc']:.3f} {freq_read['ci']} | "
          f"disagreement {off_auroc['auroc']:.3f} | cells {out['cell_means']}", flush=True)
    return out
