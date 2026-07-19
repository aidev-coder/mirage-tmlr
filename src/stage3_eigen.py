"""
Stage 3, EigenScore variant. EigenScore (INSIDE, Chen et al. 2024) is unsupervised
so there is no diagonal training: the score is a property of each statement's own
token-embedding spread. The audit is the same question in a different shape, does
the score separate truth on the diagonal (TT vs FA) yet fail on the off-diagonal
(TA vs FT), and is it really tracking typicality once you partial that out.

Signal convention: signal = -eigenscore, so higher means lower dispersion means
more likely true. AUROC handles the sign; a value below 0.5 means the score runs
against truth, which we report as-is.

Output mirrors the SAPLMA stage3 schema so report/make_tables.py reads both.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .eval.mediation import mediation
from .eval.stratified_auroc import stratified_auroc
from .stage3 import _cell_score_summary, _fragmentation_oof, load_corpus
from .stats import auroc_with_ci, gap_with_ci

_ROOT = Path(__file__).resolve().parent.parent
DIAGONAL = ("TT", "FA")
OFF = ("TA", "FT")


def _extract_or_load(substrate, texts, corpus_hash, batch_size, commit_fn=None) -> np.ndarray:
    mh = hashlib.sha256(substrate.model_id.encode()).hexdigest()[:12]
    shard = Path(substrate.cache_dir) / "stage3_eigen" / f"{mh}_{corpus_hash}.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    if shard.exists():
        a = np.load(shard)
        if a.shape[0] == len(texts):
            print(f"  [cache] eigenscore matrix loaded {a.shape}", flush=True)
            return a
    a = substrate.eigenscore_matrix(texts, batch_size=batch_size)
    np.save(shard, a)
    if commit_fn:
        commit_fn()
    print(f"  [eigenscore] saved {a.shape}", flush=True)
    return a


def run(substrate, corpus_path, n_folds: int = 5, batch_size: int = 16,
        seed: int = 20260719, commit_fn=None, **_) -> dict:
    items = load_corpus(corpus_path)
    corpus_hash = Path(corpus_path).stem.split("_v")[-1]
    texts = [it["text"] for it in items]
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    typ = np.array([it["typicality"]["entity_freq_log10"] for it in items], dtype=float)
    diag = np.flatnonzero(np.isin(cells, DIAGONAL))
    off = np.flatnonzero(np.isin(cells, OFF))

    eig = _extract_or_load(substrate, texts, corpus_hash, batch_size, commit_fn)
    frag_oof = _fragmentation_oof(items, truth, substrate.tokenizer, n_folds, seed)
    cov = {"fragmentation": frag_oof}

    per_layer = []
    for L in range(eig.shape[1]):
        sig = -eig[:, L]
        entry = {
            "layer": L,
            "adversarial": {
                "headline_heldout_diagonal": auroc_with_ci(truth[diag], sig[diag], seed=seed),
                "off_diagonal": auroc_with_ci(truth[off], sig[off], seed=seed),
                "gap": gap_with_ci(truth[diag], sig[diag], truth[off], sig[off], seed=seed),
            },
            "stratified_fielded": stratified_auroc(sig, truth, typ, seed=seed),
            "mediation_fielded": mediation(sig, truth, typ, covariates=cov),
            "fielded_cell_scores": _cell_score_summary(sig, cells),
            "mediation_allcell": {"truth_beta_partialled": None},
        }
        per_layer.append(entry)
        if commit_fn:
            commit_fn()
        a = entry["adversarial"]
        print(f"  [L{L}] diag_auroc={a['headline_heldout_diagonal'].get('auroc')} "
              f"off={a['off_diagonal'].get('auroc')} gap={a['gap'].get('gap')}", flush=True)

    return {
        "detector": "eigenscore",
        "model": substrate.model_id,
        "corpus": Path(corpus_path).name,
        "corpus_hash": corpus_hash,
        "n": len(items),
        "cell_counts": {c: int((cells == c).sum()) for c in ("TT", "TA", "FT", "FA")},
        "typicality_axis": "entity_freq_log10",
        "signal": "-eigenscore (higher=less dispersed=more likely true)",
        "headline_layer": eig.shape[1] // 2,
        "per_layer": per_layer,
        "provenance": "measured",
        "seed": seed,
    }
