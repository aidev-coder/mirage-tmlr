"""
Adapter: Marks & Tegmark's LRProbe (gradient-trained logistic regression, AdamW +
weight decay), unmodified, wrapped to the MIRAGE auditor's fit/score contract.

Second external architecture, after MMProbe. The three probes now audited span the
plausible design space for a linear truth reader:
  - our SAPLMA          : trained MLP (nonlinear head)
  - MMProbe             : closed-form whitened mean-difference direction
  - LRProbe (this file) : gradient-descent logistic regression with weight decay

If the confound were a property of one optimizer or one architecture it should not
survive all three. Source unmodified, see vendor/geometry_of_truth/PROVENANCE.md.

Feature scaling: same z-scoring as the MMProbe adapter and for the same reason —
raw residual-stream activations carry outlier dimensions that dominate an
unnormalized linear layer, and the source repo's own utils.collect_acts() centers by
default. Fit on the training split only. See the MMProbe adapter for the full note.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "vendor" / "geometry_of_truth"))


class GeometryOfTruthLRProbe:
    """fit(X, y) -> self; score(X) -> [n] P(true)."""

    def __init__(self, seed: int = 0, epochs: int = 1000,
                 lr: float = 0.001, weight_decay: float = 0.1):
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self._probe = None
        self._mu = None
        self._sd = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeometryOfTruthLRProbe":
        import torch as t

        from probes import LRProbe  # vendored, unmodified

        t.manual_seed(self.seed)   # LRProbe inits randomly; pin for reproducibility
        X = np.asarray(X, dtype=np.float64)
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd[self._sd < 1e-8] = 1.0
        Xz = (X - self._mu) / self._sd

        acts = t.tensor(Xz, dtype=t.float32)
        labels = t.tensor(np.asarray(y), dtype=t.float32)
        self._probe = LRProbe.from_data(acts, labels, lr=self.lr,
                                        weight_decay=self.weight_decay,
                                        epochs=self.epochs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch as t

        X = np.asarray(X, dtype=np.float64)
        Xz = (X - self._mu) / self._sd
        acts = t.tensor(Xz, dtype=t.float32)
        with t.no_grad():
            p = self._probe(acts)
        return p.numpy().astype(np.float64)
