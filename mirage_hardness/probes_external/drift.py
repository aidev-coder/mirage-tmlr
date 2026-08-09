"""
Adapter: DRIFT (Hussain & Kantarcioglu, "PARALLAX", arXiv 2605.17028, 2026),
wrapped to the MIRAGE auditor's fit/score contract.

Fifth external probe, and the one that breaks the shape of every other probe this
project has audited. MMProbe, LRProbe, CCS, TTPD and our own SAPLMA are all LINEAR
READOUTS OF A SINGLE TOKEN AT A SINGLE LAYER, which is the standing limitation:
the confound we report might be a property of that shape rather than of the
representation. DRIFT is neither. It taps four upper-depth layers, mean-pools each
over all tokens, and its features are the DIFFERENCES between layer pairs, so it
reads how the residual stream CHANGES with depth rather than where it sits.

Per unordered pair (a, b) of the four taps, with h mean-pooled per layer:

    phi_ab = [ h_b - h_a ,  cos(h_a, h_b) ,  ||h_b - h_a||_2 ]   in R^(d+2)

concatenated over all 6 pairs into R^(6(d+2)), then an L2-regularised logistic
regression. PARALLAX reports DRIFT statistically indistinguishable from SAPLMA on
their live-generation benchmark, so it is a strong probe, not a strawman.

DRIFT-concat is their own ablation and is implemented here too: the same four taps
mean-pooled and concatenated with NO differencing. It isolates whether the
inter-layer differencing does any work, or whether four taps alone explain the
result. If DRIFT and DRIFT-concat behave alike on our corpus, depth-differencing
is not what matters.

Input layout, declared here and nowhere else:
    X = [ tap_0 (d) | tap_1 (d) | tap_2 (d) | tap_3 (d) ]
Row indexing is unchanged, so adversarial_split and the cross-fitting loop need no
special case. `stack()` below builds it and is the only place the layout is fixed.
"""
from __future__ import annotations

import itertools

import numpy as np

TAP_FRACTIONS = (0.60, 0.70, 0.80, 0.85)


def stack(taps: list[np.ndarray]) -> np.ndarray:
    if len({t.shape for t in taps}) != 1:
        raise ValueError(f"taps must share a shape, got {[t.shape for t in taps]}")
    return np.hstack(taps)


class DriftProbe:
    """fit(X, y) -> self; score(X) -> [n] P(true). X is the [n, 4d] tap stack."""

    multi_layer = True

    def __init__(self, seed: int = 0, n_taps: int = 4, differencing: bool = True,
                 C: float = 0.01):
        self.seed = seed
        self.n_taps = n_taps
        self.differencing = differencing
        self.C = C
        self._clf = None
        self._mu = None
        self._sd = None

    def _split(self, X: np.ndarray) -> list[np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] % self.n_taps:
            raise ValueError(f"width {X.shape[1]} not divisible by {self.n_taps} taps")
        d = X.shape[1] // self.n_taps
        return [X[:, i * d:(i + 1) * d] for i in range(self.n_taps)]

    def _features(self, X: np.ndarray) -> np.ndarray:
        taps = self._split(X)
        if not self.differencing:
            return np.hstack(taps)
        blocks = []
        for a, b in itertools.combinations(range(len(taps)), 2):
            ha, hb = taps[a], taps[b]
            diff = hb - ha
            num = (ha * hb).sum(axis=1)
            den = np.linalg.norm(ha, axis=1) * np.linalg.norm(hb, axis=1)
            cos = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
            blocks.append(np.column_stack([diff, cos, np.linalg.norm(diff, axis=1)]))
        return np.hstack(blocks)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DriftProbe":
        from sklearn.linear_model import LogisticRegression

        F = self._features(X)
        self._mu = F.mean(axis=0)
        self._sd = F.std(axis=0)
        self._sd[self._sd < 1e-8] = 1.0
        self._clf = LogisticRegression(C=self.C, max_iter=2000,
                                       random_state=self.seed)
        self._clf.fit((F - self._mu) / self._sd, np.asarray(y).astype(int))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        F = (self._features(X) - self._mu) / self._sd
        return self._clf.predict_proba(F)[:, 1].astype(np.float64)


class DriftConcatProbe(DriftProbe):
    """PARALLAX's own ablation: same taps, mean-pooled and concatenated, no
    differencing. Isolates what the inter-layer differencing contributes."""

    def __init__(self, seed: int = 0, n_taps: int = 4, C: float = 0.01):
        super().__init__(seed=seed, n_taps=n_taps, differencing=False, C=C)
