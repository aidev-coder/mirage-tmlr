"""
Stage 3c — Adversarial split: train the probe the STANDARD way (on diagonal
cells, where truth and typicality are confounded exactly as in the field's
benchmarks), then test ONLY on the off-diagonal cells:

    TA  rare-but-true          (atypical, true)
    FT  fluent confident lie   (typical, false)

The off-diagonal AUROC is the honest truth-detection number. The drop from the
in-distribution headline is the artifact.

Cells follow configs/corpus.yaml: diagonal = {TT, FA}, off-diagonal = {TA, FT}.
"""

from __future__ import annotations

import numpy as np

from ..stats import auroc_with_ci, gap_with_ci

DIAGONAL = ("TT", "FA")
OFF_DIAGONAL = ("TA", "FT")


def adversarial_split(X: np.ndarray, truth: np.ndarray, cells: np.ndarray,
                      probe_factory, train_frac: float = 0.8,
                      n_boot: int = 1000, seed: int = 0) -> dict:
    """probe_factory() -> object with .fit(X, y) and .score(X) -> P(true).

    Trains on a train_frac split of the diagonal; reports
      headline      AUROC on held-out diagonal (what the field measures)
      off_diagonal  AUROC on TA+FT (what honest truth detection achieves)
      gap           headline − off_diagonal, bootstrap CI
    """
    X, truth, cells = np.asarray(X), np.asarray(truth), np.asarray(cells)
    rng = np.random.default_rng(seed)

    diag_idx = np.flatnonzero(np.isin(cells, DIAGONAL))
    off_idx = np.flatnonzero(np.isin(cells, OFF_DIAGONAL))
    if len(diag_idx) == 0 or len(off_idx) == 0:
        raise ValueError("corpus must contain both diagonal and off-diagonal cells")

    perm = rng.permutation(diag_idx)
    n_train = int(train_frac * len(perm))
    train_idx, heldout_idx = perm[:n_train], perm[n_train:]

    probe = probe_factory()
    probe.fit(X[train_idx], truth[train_idx])

    s_heldout = probe.score(X[heldout_idx])
    s_off = probe.score(X[off_idx])

    headline = auroc_with_ci(truth[heldout_idx], s_heldout, n_boot=n_boot, seed=seed)
    off = auroc_with_ci(truth[off_idx], s_off, n_boot=n_boot, seed=seed)
    gap = gap_with_ci(truth[heldout_idx], s_heldout, truth[off_idx], s_off,
                      n_boot=n_boot, seed=seed)

    per_cell = {}
    for c in OFF_DIAGONAL:
        m = off_idx[cells[off_idx] == c]
        per_cell[c] = {"n": int(len(m)),
                       "mean_score": round(float(probe.score(X[m]).mean()), 4) if len(m) else None}

    return {"train_n": int(n_train), "headline_heldout_diagonal": headline,
            "off_diagonal": off, "gap": gap, "off_diagonal_cells": per_cell}
