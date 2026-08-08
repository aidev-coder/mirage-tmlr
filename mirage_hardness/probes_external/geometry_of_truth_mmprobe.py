"""
Adapter: Marks & Tegmark's mass-mean probe (MMProbe), unmodified, wrapped to the
MIRAGE auditor's fit(X, y) / score(X) contract.

This is the first EXTERNAL probe run through mirage_hardness/audit_probe.py — every
prior number in this project used our own SAPLMA reimplementation. MMProbe is a
different architecture (whitened mean-difference direction, not a trained MLP) from
a paper this project already cites as the optimistic baseline. Source: unmodified,
see vendor/geometry_of_truth/PROVENANCE.md.

FEATURE SCALING (2026-08-07, adapter correction, logged here rather than silently):
`MMProbe.forward` with iid=False just does sigmoid(x @ direction) on whatever
activations it is given. The original repo's own utils.collect_acts() defaults to
center=True and offers scale=True precisely because raw residual-stream activations
carry "rogue"/outlier feature dimensions (seen here on gemma-2-9b-it: max abs value
276 vs a mean of 2.4) that dominate a raw dot product and saturate the sigmoid,
destroying rank information regardless of the true signal. Feeding the auditor's raw
[n, d] activations straight into MMProbe produced a degenerate AUROC of exactly 0.5
on gemma — not a "no confound" finding, a saturated score. We standardize features
(z-score, fit on the training split only, matching what our own built-in SAPLMA does
via sklearn's StandardScaler) before calling into the vendored, otherwise-unmodified
MMProbe code. This is the minimum change needed for the paper's own method to
function as its authors intend on activations it was not pre-normalized for; it is
not a tuning of the confound estimate itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "vendor" / "geometry_of_truth"))


class GeometryOfTruthMMProbe:
    """fit(X, y) -> self; score(X) -> [n] P(true), the auditor's two-method contract."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._probe = None
        self._mu = None
        self._sd = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeometryOfTruthMMProbe":
        import torch as t

        from probes import MMProbe  # vendored, unmodified

        X = np.asarray(X, dtype=np.float64)
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd[self._sd < 1e-8] = 1.0
        Xz = (X - self._mu) / self._sd

        acts = t.tensor(Xz, dtype=t.float32)
        labels = t.tensor(np.asarray(y), dtype=t.float32)
        self._probe = MMProbe.from_data(acts, labels)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch as t

        X = np.asarray(X, dtype=np.float64)
        Xz = (X - self._mu) / self._sd
        acts = t.tensor(Xz, dtype=t.float32)
        with t.no_grad():
            p = self._probe(acts, iid=False)
        return p.numpy().astype(np.float64)
