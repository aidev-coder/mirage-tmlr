"""
Adapter: CCS (Contrast-Consistent Search), Burns, Ye, Klein & Steinhardt 2022,
wrapped to the MIRAGE auditor's fit/score contract.

Third external probe, and the first from a different research group and a
different premise: CCS is UNSUPERVISED. It fits a direction on pairs of a
statement and its negation, asking only that the pair get complementary
probabilities and that neither collapse to 0.5. Truth labels enter at one point
only, resolving the sign of a direction the optimizer already found. So if CCS
still substitutes entity frequency for truth, the labels did not teach it that.

Lineage, stated precisely: the METHOD is Burns et al., the IMPLEMENTATION is Marks
& Tegmark's reimplementation (vendor/geometry_of_truth/probes.py), the same file
the other two adapters use. Widens the method lineage, not the code lineage.

Paired input: every other probe takes a single [n, d] matrix. Rather than change a
contract the released auditor depends on, this takes a horizontally stacked
[n, 2d] — statement acts then negation acts — and splits it internally, so
adversarial_split and the cross-fitting loop index rows exactly as before.

Normalization is PER SIDE, not shared: x+ and x- are each z-scored with their own
mean and standard deviation. This is Burns et al.'s published procedure and it is
load-bearing, not cosmetic — the statement and its negation differ by a large
constant "contains not" direction, and a probe left free to read that direction
separates prompt form instead of truth. An earlier version of this adapter shared
the statistics across both halves and CCS came back anti-correlated with truth
(in-distribution AUROC 0.21-0.50 across most layers of every model), which is that
shortcut winning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "vendor" / "geometry_of_truth"))


def stack(X_pos: np.ndarray, X_neg: np.ndarray) -> np.ndarray:
    X_pos, X_neg = np.asarray(X_pos), np.asarray(X_neg)
    if X_pos.shape != X_neg.shape:
        raise ValueError(f"shape mismatch: {X_pos.shape} vs {X_neg.shape}")
    return np.hstack([X_pos, X_neg])


class CCSProbeAdapter:
    """fit(X, y) -> self; score(X) -> [n] P(true). X is the [n, 2d] stack."""

    paired = True

    def __init__(self, seed: int = 0, epochs: int = 1000,
                 lr: float = 0.001, weight_decay: float = 0.1):
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self._probe = None
        self._stats = None

    @staticmethod
    def _fit_stats(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = A.mean(axis=0), A.std(axis=0)
        sd[sd < 1e-8] = 1.0
        return mu, sd

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] % 2:
            raise ValueError(f"CCS expects a stacked [n, 2d] matrix, got d={X.shape[1]}")
        d = X.shape[1] // 2
        return X[:, :d], X[:, d:]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CCSProbeAdapter":
        import torch as t

        from probes import CCSProbe

        t.manual_seed(self.seed)
        pos, neg = self._split(X)
        self._stats = (self._fit_stats(pos), self._fit_stats(neg))
        (mu_p, sd_p), (mu_n, sd_n) = self._stats

        acts = t.tensor((pos - mu_p) / sd_p, dtype=t.float32)
        neg_acts = t.tensor((neg - mu_n) / sd_n, dtype=t.float32)
        labels = t.tensor(np.asarray(y, dtype=float), dtype=t.float32)
        self._probe = CCSProbe.from_data(acts, neg_acts, labels=labels, lr=self.lr,
                                         weight_decay=self.weight_decay,
                                         epochs=self.epochs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch as t

        pos, _ = self._split(X)
        (mu_p, sd_p), _ = self._stats
        acts = t.tensor((pos - mu_p) / sd_p, dtype=t.float32)
        with t.no_grad():
            p = self._probe(acts)
        return p.numpy().astype(np.float64)
