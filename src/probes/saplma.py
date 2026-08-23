"""
SAPLMA-style probe (Azaria & Mitchell 2023, "The Internal State of an LLM
Knows When It's Lying"): a small feedforward classifier on the last-token
hidden state of a chosen layer.

Faithful-reproduction notes ( — audit the strongest real version):
  - architecture matches the paper: 3 ReLU hidden layers (256, 128, 64), sigmoid out;
  - trained per-layer; §4.3 requires sweeping ALL layers and reporting the curve.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..stats import auroc_with_ci


class SaplmaProbe:
    """Feedforward truth probe on one layer's last-token hidden state."""

    def __init__(self, hidden_sizes=(256, 128, 64), max_iter: int = 400,
                 seed: int = 0):
        self.scaler = StandardScaler()
        self.clf = MLPClassifier(hidden_layer_sizes=tuple(hidden_sizes),
                                 activation="relu", max_iter=max_iter,
                                 early_stopping=True, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SaplmaProbe":
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """P(true) per item — the probe score all Stage-3 tests consume."""
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


def layer_sweep(hidden_states: np.ndarray, y: np.ndarray,
                train_idx: np.ndarray, test_idx: np.ndarray,
                seed: int = 0, **probe_kw) -> list[dict]:
    """Train and evaluate one probe per layer; return the FULL curve (D-004).

    hidden_states: [n_items, n_layers+1, d_model].
    Never returns a single 'best layer' — cherry-picking is the confound.
    """
    out = []
    for layer in range(hidden_states.shape[1]):
        X = hidden_states[:, layer, :]
        probe = SaplmaProbe(seed=seed, **probe_kw).fit(X[train_idx], y[train_idx])
        res = auroc_with_ci(y[test_idx], probe.score(X[test_idx]), seed=seed)
        res["layer"] = layer
        out.append(res)
    return out
