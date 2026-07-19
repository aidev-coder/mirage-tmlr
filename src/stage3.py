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
        seed: int = 20260719, commit_fn=None) -> dict:
    items = load_corpus(corpus_path)
    corpus_hash = Path(corpus_path).stem.split("_v")[-1]
    texts = [it["text"] for it in items]
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    typicality = np.array([it["typicality"]["entity_freq_log10"] for it in items], dtype=float)

    H = _extract_or_load(substrate, texts, corpus_hash, batch_size, commit_fn)
    n_layers = H.shape[1]
    frag_oof = _fragmentation_oof(items, truth, substrate.tokenizer, n_folds, seed)

    def factory(_seed=seed):
        from .probes.saplma import SaplmaProbe
        return SaplmaProbe(seed=_seed)

    per_layer = []
    for L in range(n_layers):
        oof = _oof_scores(H[:, L, :], truth, device, n_folds, seed)
        entry = {
            "layer": L,
            "stratified": stratified_auroc(oof, truth, typicality, seed=seed),
            "mediation": mediation(oof, truth, typicality,
                                   covariates={"fragmentation": frag_oof}),
            "adversarial": adversarial_split(H[:, L, :], truth, cells, factory, seed=seed),
        }
        per_layer.append(entry)
        if commit_fn:
            commit_fn()
        g = entry["stratified"].get("gap", {})
        print(f"  [L{L}] strat_gap={g.get('point')} {g.get('ci')} "
              f"truth_beta {entry['mediation']['truth_beta_marginal']}->"
              f"{entry['mediation']['truth_beta_partialled']} "
              f"adv_off={entry['adversarial']['off_diagonal'].get('auroc')}", flush=True)

    headline_layer = n_layers // 2
    return {
        "detector": detector,
        "model": substrate.model_id,
        "corpus": Path(corpus_path).name,
        "corpus_hash": corpus_hash,
        "n": len(items),
        "cell_counts": {c: int((cells == c).sum()) for c in ("TT", "TA", "FT", "FA")},
        "n_folds": n_folds,
        "typicality_axis": "entity_freq_log10",
        "mediation_covariates": ["typicality", "fragmentation"],
        "headline_layer": headline_layer,
        "per_layer": per_layer,
        "provenance": "measured",
        "seed": seed,
    }
