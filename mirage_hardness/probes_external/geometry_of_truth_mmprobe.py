"""
Adapter: Marks & Tegmark's mass-mean probe (MMProbe), unmodified, wrapped to the
MIRAGE auditor's fit(X, y) / score(X) contract.

This is the first EXTERNAL probe run through mirage_hardness/audit_probe.py — every
prior number in this project used our own SAPLMA reimplementation. MMProbe is a
different architecture (whitened mean-difference direction, not a trained MLP) from
a paper this project already cites as the optimistic baseline. Source: unmodified,
see vendor/geometry_of_truth/PROVENANCE.md.
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeometryOfTruthMMProbe":
        import torch as t

        from probes import MMProbe  # vendored, unmodified

        acts = t.tensor(np.asarray(X), dtype=t.float32)
        labels = t.tensor(np.asarray(y), dtype=t.float32)
        self._probe = MMProbe.from_data(acts, labels)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch as t

        acts = t.tensor(np.asarray(X), dtype=t.float32)
        with t.no_grad():
            p = self._probe(acts, iid=False)
        return p.numpy().astype(np.float64)
